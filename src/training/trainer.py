"""
RDMCA progressive stage trainer — the training loop.

Each stage trains until it passes its graduation gate (ratcheting masked-validation
perplexity) or its token budget is exhausted, shipping the best checkpoint either way.
The foundational core (cognition + values) is frozen permanently after the last ACTIVE
cognitive stage (ethics/BCF); behavioral stages then train LoRA sectors on top.

This module is the importable trainer; the CLI lives in scripts/train.py. The helper
groups live in sibling modules (curriculum / checkpoint / gates / valdata / dataload /
heads); they are imported here and re-exported so callers and tests have one entry point.
"""

from __future__ import annotations

import contextlib
import json
import math
import time
from pathlib import Path

import src.backend as backend
from src.plugins import get_stage, stage_data_dir
from src.training.checkpoint import (
    freeze_model,  # noqa: F401  (re-exported)
    load_checkpoint,
    save_checkpoint,
    write_stage_audit,
    write_stage_complete,
)
from src.training.curriculum import (
    BCF_STAGE,  # noqa: F401  (re-exported for callers/tests)
    ckpt_root,
    cosine_lr,
    is_behavioral_stage,  # noqa: F401  (re-exported for tests)
    last_cognitive_stage,  # noqa: F401  (re-exported for tests)
    load_config,  # noqa: F401  (re-exported)
    prev_active_stage,  # noqa: F401  (re-exported for tests)
    stage_name,
)
from src.training.gates import (
    evaluate_gate,  # noqa: F401  (re-exported for tests; the loop calls model_spec.evaluate)
    gate_decision,
    gate_threshold,
    stage_entry_ppl,
    validation_perplexity,  # noqa: F401  (re-exported)
)
from src.training.heads import on_stage_complete
from src.training.valdata import make_val_batches


def train_stage(stage: int, cfg: dict, resume: bool = False, plain: bool = False) -> bool:
    tcfg = cfg["training"]
    skip_gate = cfg.get("skip_gate", False)  # toy config sets this to true
    n_tokens_target = cfg["curriculum"][f"stage{stage}"]["n_tokens"]  # key-based
    root = ckpt_root(cfg)
    ckpt_dir = root / f"stage{stage}"

    def _fmt_tokens(n: int) -> str:
        if n >= 1_000_000_000:
            return f"{n / 1e9:.2f}B"
        if n >= 1_000_000:
            return f"{n / 1e6:.0f}M"
        return f"{n / 1e3:.0f}K"

    print(f"\n{'=' * 60}")
    print(f"  Stage {stage}: {stage_name(stage, cfg)}")
    print(f"  Target: {_fmt_tokens(n_tokens_target)} tokens")
    print(f"{'=' * 60}")

    # Resolve the ACTIVE model spec (text-LM by default). Every task-specific piece —
    # how to build the network, the loader, the loss and the gate — comes from this
    # spec, so the same loop trains a different kind of model when the model changes.
    from src.model.transformer import set_model_precision
    from src.training.dashboard import TrainingDashboard
    from src.training.model_spec import active_model_spec

    model_spec = active_model_spec(cfg)

    # Build the model at the trained-tokenizer vocab and load its starting weights
    # (cognitive: prev stage; behavioral: frozen core + LoRA sector). See setup.py.
    model, model_cfg, adapter, precision, seed = model_spec.build_model(stage, cfg, root)
    B = backend.current()

    # optimizer_states: bf16 (default — already in effect since the model is bf16)
    # or int8 (bitsandbytes 8-bit AdamW on CUDA, a further memory saving for big runs;
    # opt-in, falls back to bf16 with a warning when unavailable).
    optimizer = B.engine.make_optimizer(
        model, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"], states=tcfg.get("optimizer_states")
    )

    start_step = 0
    tokens_seen = 0
    if resume:
        start_step, tokens_seen = load_checkpoint(model, ckpt_dir, optimizer)

    # Re-apply precision: loading prev-stage / resume weights restores their
    # saved dtype, so cast once more before training in the configured precision.
    set_model_precision(model, precision)

    # Real data loader (from the active model spec — TextDataset for the text-LM model)
    data_loader = model_spec.build_loader(stage, cfg)

    # Derived constants
    bs = tcfg["batch_size"]
    grad_acc = tcfg["grad_accumulation"]
    clip_norm = tcfg.get("clip_grad_norm")  # global-norm grad clip (None = off)
    seq_len = model_cfg.context_len
    toks_step = bs * seq_len * grad_acc

    # On resume, fast-forward the data stream past what the interrupted run already
    # consumed — otherwise the loader restarts at token 0 and re-trains the early
    # data while never reaching the rest (issue C3). The dataset + replay draws are
    # fully seeded, so replaying start_step×grad_acc batches lands EXACTLY where the
    # run stopped. One-time re-tokenization cost (an indexed loader would skip it).
    skip_index_path = ckpt_dir / "skip_index.npz"
    if resume and start_step > 0:
        n_skip = start_step * grad_acc
        # Load the per-record token-length index saved with the checkpoint: the skip
        # then replays cached lengths instead of re-tokenizing the consumed span
        # (seconds vs minutes). Missing/stale index → exact-but-slow live skip.
        fast = data_loader.load_skip_index(skip_index_path)
        print(
            f"  [resume] fast-forwarding data stream past {n_skip:,} batches "
            f"({start_step:,} steps){' — cached lengths' if fast else ' — re-tokenizing (no index)'}…"
        )
        skipped = data_loader.skip(n_skip)
        if skipped < n_skip:
            print(f"  [resume] stream shorter than expected — skipped {skipped:,}.")
    warmup = tcfg["warmup_steps"]
    total_steps = n_tokens_target // toks_step
    # Cap re-cycling of a small corpus: training a tiny corpus toward an oversized
    # token target just re-reads it many times → overfit/parroting. Once we know
    # the corpus size (after the first pass) we lower the effective target to at
    # most `max_corpus_passes` reads, so the run completes (and the bar reaches
    # 100%) against a budget the data can actually support.
    max_passes = int(tcfg.get("max_corpus_passes", 3))
    target = n_tokens_target
    capped = False
    save_every = tcfg["save_every"]
    eval_every = tcfg["eval_every"]

    # Anchor the LR horizon up front by estimating the corpus size from disk, so
    # cosine decays toward the REAL (possibly cap-limited) end from step 1. Doing
    # this only at runtime when the cap triggers would shift total_steps mid-run
    # and make the schedule jump discontinuously (M3). The estimate is approximate
    # (bytes/≈chars-per-token); it only sets the LR horizon, not the stop point.
    stage_cfg = cfg["curriculum"][f"stage{stage}"]
    # Per-stage LR scale: the narrow late cognitive stages (arithmetic/CoT) OVERWRITE the
    # shared core at the full LR. A gentler LR makes them NUDGE the core (learn the skill)
    # instead of stamping their low-entropy format over conversation. Default is a STAGE
    # property (from the registry); the yaml may override per stage.
    lr_scale = float(stage_cfg.get("lr_scale", get_stage(stage).lr_scale))
    base_lr = tcfg["lr"] * lr_scale
    min_lr = tcfg.get("lr_min", 3e-5) * lr_scale
    data_dir = Path(stage_data_dir(stage, cfg))
    corpus_bytes = (
        sum(f.stat().st_size for f in data_dir.glob("*.jsonl")) if data_dir.exists() else 0
    )
    est_corpus = int(corpus_bytes / 3.5)  # ≈ chars/token (prose-weighted)
    if est_corpus:
        eff_target = min(n_tokens_target, max_passes * est_corpus)
        total_steps = max(eff_target // toks_step, 1)

    step = start_step
    dash_interval = 10  # update dashboard (and measure tok/s) every N steps
    t_dash = time.time()
    last_tps = 0.0
    # NaN/Inf guard: a single unstable batch (fp16 underflow, a bad sample, an LR
    # spike) can blow the loss to NaN; without a guard the optimizer then writes
    # NaN into every weight and the run silently continues training garbage for
    # hours. We SKIP the update on a non-finite loss and abort if it persists.
    _NAN_ABORT = 20  # consecutive non-finite losses ⇒ the run has diverged
    nan_streak = 0

    # Best-checkpoint tracking (anti-divergence). Stages whose data is narrow and
    # format-shifting (esp. stage 6 memory) MEMORIZE then DIVERGE as they re-cycle
    # the corpus. We remember the lowest-val-perplexity point, and at stage end (or
    # after `early_stop_patience` evals with no improvement) we RESTORE it instead of
    # shipping the diverged tail. patience=0 disables the in-loop early stop but
    # best-restore at stage end still applies. Independent of the graduation gate.
    patience = int(tcfg.get("early_stop_patience", 4))
    min_delta = float(tcfg.get("early_stop_min_delta", 0.002))  # 0.2% PPL gain = "meaningful"
    # (only for plateau; any gain still saves)
    best_path = ckpt_dir / "best.npz"
    best_meta = ckpt_dir / "best.json"
    best_score = float("inf")
    best_step = start_step
    best_tokens = tokens_seen
    best_loss = 0.0
    stale = 0
    # Plateau reference for early-stop. We compare CUMULATIVE improvement against this
    # reference (which only moves on a meaningful drop), NOT step-to-step — because late
    # gains shrink toward 0, a per-step min_delta would declare a plateau while the model
    # is still slowly improving. Every strict improvement is still SAVED as the best.
    plateau_ref = float("inf")
    restored_best = False  # set when the diverged tail is rolled back to best
    # Anti-divergence RESTORE point: the lowest val-PPL seen REGARDLESS of the floor.
    # Distinct from `best` — a point above the floor is NOT a graduation best (it never
    # becomes best.npz), but we still keep its weights so a budget-exhausted run that
    # never reached the floor ships its lowest point instead of a diverged tail.
    low_path = ckpt_dir / "restore.npz"
    low_score = float("inf")
    low_step = start_step
    low_tokens = tokens_seen
    low_loss = 0.0
    # PERSISTENT monotonic best: the best checkpoint is the bar to beat ACROSS runs, not
    # just within one. On --resume we load the historical best so a later run REPLACES it
    # only on a genuine improvement — a worse checkpoint is discarded and never clobbers
    # the best. The absolute graduation gate is separate.
    if resume and best_meta.exists() and best_path.exists():
        try:
            bm = json.loads(best_meta.read_text())
            best_score, best_step = float(bm["score"]), int(bm["step"])
            best_tokens, best_loss = int(bm["tokens"]), float(bm["loss"])
            # The historical best is, by construction, a floor-clearing candidate and
            # thus also the lowest seen — carry it into the restore tracker too.
            low_score, low_step = best_score, best_step
            low_tokens, low_loss = best_tokens, best_loss
            plateau_ref = best_score  # measure further progress from here
            print(
                f"  [best] resuming against historical best val PPL {best_score:.2f} "
                f"(step {best_step:,}) — only an improvement replaces it"
            )
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            pass

    def _save_best(score: float, step: int, tokens: int, loss: float) -> None:
        """Persist the new best weights + its metadata so the bar survives across runs."""
        B.engine.save_weights(model, str(best_path))
        with contextlib.suppress(OSError):
            best_meta.write_text(
                json.dumps(
                    {
                        "score": score,
                        "step": step,
                        "tokens": tokens,
                        "loss": loss,
                        "timestamp": time.time(),
                    },
                    indent=2,
                )
            )

    # Training objective from the active model spec. For the text-LM model this is the
    # MRL completion-only CE plus the MoE load-balance aux term (aux_loss() is 0.0 with
    # no routing, so it's a no-op on cognitive stages); other models supply their own.
    loss_and_grad_fn = B.engine.value_and_grad(model, model_spec.objective)

    # Fixed validation batches, sampled ONCE up front and reused for every gate eval (a
    # frozen set keeps the metric stable/comparable). Prefer a true HELD-OUT split
    # (`*.val.jsonl`); fall back to the training stream when no split was prepared.
    val_batches = make_val_batches(stage, cfg, data_loader, n=8)

    # Inherited-baseline (entry) PP — measured once on the loaded checkpoint before
    # training, so stage progress can be read as the offset-corrected Δ from here. This
    # is perplexity telemetry (it calls model.eval_ce); a model with no such head simply
    # reports no baseline (the Δ display is then skipped downstream).
    if model_spec.gate_metric == "perplexity" and hasattr(model, "eval_ce"):
        entry_ppl = stage_entry_ppl(
            model, ckpt_dir, val_batches, resume, log=lambda m: print(m) if plain else None
        )
    else:
        entry_ppl = float("inf")

    # Persist the full training CONTEXT (config, data provenance, rehearsal mix,
    # model geometry, env/git) for post-hoc audits — see write_stage_audit.
    audit = write_stage_audit(
        ckpt_dir,
        stage=stage,
        cfg=cfg,
        model=model,
        model_cfg=model_cfg,
        tcfg=tcfg,
        data_loader=data_loader,
        target=target,
        total_steps=total_steps,
        precision=precision,
        seed=seed,
        hparams_extra={
            "rehearsal_fraction": getattr(data_loader, "replay_fraction", 0.0),
            "early_stop_patience": patience,
            "early_stop_min_delta": min_delta,
        },
    )

    # The dashboard's per-token perplexity divides the COMPOSITE loss by its CE-unit
    # weight: the MRL head mean is 1 CE-unit, each MTP head adds `mtp_loss_weight`.
    # Without this, exp(loss) reports a wildly inflated PP (e.g. 184K at init).
    n_mtp = int(getattr(model_cfg, "n_mtp_heads", 0) or 0)
    mtp_w = float(getattr(model_cfg, "mtp_loss_weight", 0.0) or 0.0)
    loss_ce_weight = 1.0 + n_mtp * mtp_w

    dash = TrainingDashboard(
        stage,
        n_tokens_target,
        resume_step=start_step,
        resume_tokens=tokens_seen,
        params=model.count_params(),
        n_layers=model.cfg.n_layers,
        d_model=model.cfg.d_model,
        plain=plain,
        log_path=ckpt_dir / "train.log",
        loss_ce_weight=loss_ce_weight,
        append=resume,  # fresh run truncates log+metrics; --resume appends
        gate_baseline=entry_ppl,
    )

    with dash:
        dash.print(f"Stage {stage} | {model.count_params() / 1e6:.1f}M params | real data")
        # Echo the audit context into train.log so the text log is self-contained.
        dash.print(
            f"[audit] full context → {ckpt_dir / 'audit.json'} | git {audit['git_commit']} "
            f"| seed {seed} | precision {precision}"
        )
        dash.print(
            "[data] sources: "
            + (
                ", ".join(
                    f"{s['source']}={s.get('tokens', 0) / 1e6:.1f}M{'*' if s.get('exhausted') else ''}"
                    for s in audit["data"]["sources"]
                )
                or "(none)"
            )
        )
        if audit["rehearsal"]["dirs"]:
            dash.print(
                f"[rehearsal] {audit['rehearsal']['fraction']:.0%} replay over "
                f"{len(audit['rehearsal']['dirs'])} stage(s), weights%="
                f"{audit['rehearsal']['weights_pct']}"
            )
        lr_tag = f"lr={base_lr:.2e}" + (f" (×{lr_scale:g} stage scale)" if lr_scale != 1.0 else "")
        dash.print(
            f"[hparams] {lr_tag} bs={bs} warmup={warmup} "
            f"max_passes={max_passes} early_stop=patience{patience}/δ{min_delta}"
        )

        # Seed loop-carried values so the final dashboard update is safe even when the
        # loop never runs — e.g. --resume on a stage that ALREADY hit its budget.
        acc_loss = 0.0
        lr = cosine_lr(step, base_lr, min_lr, warmup, total_steps)
        while tokens_seen < target:
            # Update learning rate (per-stage scaled base/min — see lr_scale)
            lr = cosine_lr(step, base_lr, min_lr, warmup, total_steps)
            B.engine.set_lr(optimizer, lr)

            # Gradient accumulation — TRUE accumulation: sum the per-micro-batch
            # gradients (mean over grad_acc) so the effective batch is really
            # bs×grad_acc. ga==1 stays on a zero-overhead fast path.
            acc_loss = 0.0
            if grad_acc == 1:
                toks_np, mask_np = data_loader.next_batch()
                batch = (B.ops.array(toks_np), B.ops.array(mask_np))
                loss, grads = loss_and_grad_fn(model, batch)
                B.engine.eval(loss)
                acc_loss = B.engine.item(loss)
            else:
                running = None
                for _ in range(grad_acc):
                    toks_np, mask_np = data_loader.next_batch()
                    batch = (B.ops.array(toks_np), B.ops.array(mask_np))
                    loss, g = loss_and_grad_fn(model, batch)
                    B.engine.eval(loss)
                    acc_loss += B.engine.item(loss)
                    running = B.engine.accumulate_grads(running, g, model)
                grads = B.engine.finalize_grads(running, 1.0 / grad_acc, model)

            # NaN/Inf guard: skip the optimizer update on a non-finite loss so the
            # blow-up never reaches the weights; abort if it keeps happening (the run
            # has diverged — lower the LR or inspect the data). acc_loss is already
            # synced (item()), so this check is free.
            if not math.isfinite(acc_loss):
                nan_streak += 1
                dash.print(
                    f"  [nan] non-finite loss ({acc_loss}) at step {step} — "
                    f"skipped update ({nan_streak}/{_NAN_ABORT})"
                )
                if nan_streak >= _NAN_ABORT:
                    raise RuntimeError(
                        f"Training diverged: {_NAN_ABORT} consecutive non-finite "
                        "losses. Aborting before the checkpoint is corrupted — lower "
                        "the learning rate or check the data/precision."
                    )
                step += 1
                continue
            nan_streak = 0

            # Pre-clip gradient norm — sampled only on dashboard-refresh steps
            # (it forces a device sync, so we don't pay it every step). Shown as a
            # stability metric: sustained spikes mean the LR is too high.
            grad_norm_val = None
            if (step + 1) % dash_interval == 0:
                grad_norm_val = B.engine.grad_norm(model, grads)

            # Clip gradients to a global norm (config: training.clip_grad_norm) to
            # tame the loss spikes that come from the occasional high-norm batch.
            if clip_norm:
                grads = B.engine.clip_grads(model, grads, float(clip_norm))

            B.engine.optimizer_step(optimizer, model, grads)

            step += 1
            tokens_seen += toks_step

            # After the first full pass, cap the effective target to the corpus
            # size × max_passes (only if that's *below* the configured target).
            if not capped and data_loader.epoch_tokens:
                capped = True
                cap = max_passes * data_loader.epoch_tokens
                if cap < n_tokens_target:
                    target = cap  # stop point; LR horizon was set up front
                    dash.set_target(target)
                    dash.print(
                        f"[corpus] {data_loader.epoch_tokens / 1e6:.1f}M tokens/pass "
                        f"— capping at {max_passes}× ({_fmt_tokens(cap)}) to avoid "
                        f"overfitting the configured {_fmt_tokens(n_tokens_target)} target"
                    )

            # Dashboard refresh every dash_interval steps (smooth). Throughput is
            # measured over this same short window so the Speed field is populated
            # within the first dash_interval steps — otherwise it reads 0.0 tok/s
            # until step 100, which looks like a hung run (especially on the torch
            # MPS backend, whose cold first-step kernel compile is already slow).
            if step % dash_interval == 0:
                elapsed = time.time() - t_dash
                if elapsed > 0:
                    last_tps = (dash_interval * toks_step) / elapsed
                t_dash = time.time()
                dash.update(
                    step,
                    tokens_seen,
                    acc_loss / grad_acc,
                    lr,
                    last_tps,
                    grad_norm=grad_norm_val,
                    passes=data_loader.passes,
                    replay=getattr(data_loader, "last_was_replay", False),
                )

            # Checkpoint. Route the [ckpt]/[gate] detail through dash.print so it
            # lands INSIDE the dashboard's log region (raw print() would draw over
            # the pinned panel — the "broken UI" / stray lines outside the log box).
            if step % save_every == 0:
                save_checkpoint(
                    model,
                    step,
                    stage,
                    tokens_seen,
                    acc_loss / grad_acc,
                    ckpt_dir,
                    optimizer,
                    log=dash.print,
                )
                # Persist the data-stream length index alongside the checkpoint so a
                # later --resume can fast-forward without re-tokenizing the span.
                data_loader.save_skip_index(skip_index_path)
                dash.set_checkpoint(step)

            # RATCHETING gate. Each eval the score must BEAT the best seen so far to
            # "pass" — the best is the moving bar, so a worse checkpoint is marked
            # not-passed and DISCARDED. The stage GRADUATES when it plateaus (no new
            # best for `patience` evals), shipping that best — but only if it clears
            # the absolute quality bar.
            if step % eval_every == 0:
                # meets_bar = absolute quality floor; quiet log in-loop (we print the
                # ratchet line below). skip_gate levels keep the plain informational line.
                score, meets_bar = model_spec.evaluate(
                    model,
                    stage,
                    val_batches,
                    cfg,
                    log=(dash.print if skip_gate else (lambda *a, **k: None)),
                    step=step,
                )
                thr = gate_threshold(stage, cfg)
                # Ratchet: is_candidate = cleared the floor; is_new_best = candidate AND
                # strictly beats the best (ANY gain → saved); is_meaningful = gain beyond
                # min_delta (drives plateau only).
                is_candidate, is_new_best, _ = gate_decision(score, best_score, thr, min_delta)
                # Progress vs the STARTING ppl (inherited baseline): arrow = direction
                # (↓ improved, ↑ worse), so there is no +/- sign to second-guess.
                if math.isfinite(entry_ppl) and entry_ppl > 0:
                    delta = (score - entry_ppl) / entry_ppl * 100
                    dentry = (
                        f" · start {entry_ppl:.1f} {'↓' if delta < 0 else '↑'}{abs(delta):.0f}%"
                    )
                else:
                    dentry = ""
                dash.set_gate_result(
                    score,
                    is_new_best if not skip_gate else meets_bar,
                    threshold=thr,
                    best=best_score,
                )
                # Plateau is measured against the cumulative reference (only moves on a
                # meaningful drop), so shrinking late gains don't trigger a false plateau.
                meaningful = is_new_best and score < plateau_ref * (1.0 - min_delta)

                # Anti-divergence restore point: lowest seen REGARDLESS of the floor. Kept
                # so a run that never reaches the floor still ships its lowest point, not a
                # diverged tail. This is NOT "the best" (separate file, never loaded as best).
                if math.isfinite(score) and score < low_score:
                    low_score, low_step = score, step
                    low_tokens, low_loss = tokens_seen, acc_loss / grad_acc
                    B.engine.save_weights(model, str(low_path))

                if is_new_best:
                    # Cleared the floor AND beat the prior best → new best (the moving
                    # bar). ALWAYS saved, even on a sub-min_delta gain — a better
                    # checkpoint is never discarded. The gain's size only affects plateau.
                    if not skip_gate:
                        prev = "∞" if best_score == float("inf") else f"{best_score:.2f}"
                        if meaningful:
                            dash.print(
                                f"[gate] step={step:,} | val ppl {score:.2f} < best "
                                f"{prev}{dentry} → PASSED, new best (bar↓)"
                            )
                        else:
                            ref = "∞" if plateau_ref == float("inf") else f"{plateau_ref:.2f}"
                            dash.print(
                                f"[gate] step={step:,} | val ppl {score:.2f} < best "
                                f"{prev} → new best, SAVED — but cumulative gain since "
                                f"{ref} < {min_delta:.1%} min_delta, plateauing "
                                f"({stale + 1}/{patience})"
                            )
                    best_score, best_step, best_tokens = score, step, tokens_seen
                    best_loss = acc_loss / grad_acc
                    _save_best(best_score, best_step, best_tokens, best_loss)
                    if meaningful:
                        plateau_ref, stale = score, 0
                    else:
                        stale += 1
                else:
                    # Not a new best — either still above the floor (not viable yet, NOT a
                    # best per the gate) or no better than the running best (discarded).
                    if not skip_gate:
                        if not is_candidate:
                            bar = (
                                "no best yet"
                                if best_score == float("inf")
                                else f"best {best_score:.2f}"
                            )
                            dash.print(
                                f"[gate] step={step:,} | val ppl {score:.2f} > floor "
                                f"{thr:.1f} → not viable yet, NOT a best ({bar})"
                            )
                        elif math.isfinite(best_score):
                            dash.print(
                                f"[gate] step={step:,} | val ppl {score:.2f} ≥ best "
                                f"{best_score:.2f} → no improvement, discarded "
                                f"({stale + 1}/{patience})"
                            )
                    if math.isfinite(best_score):
                        stale += 1

                # Plateau (counts only once a floor-clearing best exists) → ship the best.
                # The best is, by construction, a floor-clearing candidate, so it graduates.
                if math.isfinite(best_score) and patience and stale >= patience:
                    B.engine.load_weights(model, str(best_path))
                    save_checkpoint(
                        model, best_step, stage, best_tokens, best_loss, ckpt_dir, log=dash.print
                    )
                    B.engine.save_weights(model, str(ckpt_dir / "final.npz"))  # graduated
                    write_stage_complete(
                        ckpt_dir,
                        stage=stage,
                        step=best_step,
                        tokens_seen=best_tokens,
                        gate_score=best_score,
                        gate_threshold=thr,
                        met_bar=True,
                        entry_ppl=entry_ppl,
                        checkpoint=str(ckpt_dir / f"step_{best_step:08d}.npz"),
                        early_stopped=True,
                    )
                    dash.print(
                        f"✓ Stage {stage} COMPLETE — best val ppl "
                        f"{best_score:.2f} (≤ floor {thr:.1f}), plateaued"
                    )
                    on_stage_complete(model, stage, cfg, root, ckpt_dir, precision, adapter)
                    return True

        # Final dashboard update so it shows 100%
        dash.update(step, tokens_seen, acc_loss / grad_acc, lr, last_tps)

        # Budget exhausted — ship the lowest-val-PPL point, not a memorized/diverged tail
        # (stage 6's 0.4→4.4 climb). Prefer the graduation best (a floor-clearing
        # candidate); if the floor was NEVER reached, fall back to the lowest-seen restore
        # point so we still avoid the diverged tail.
        if best_score < float("inf") and best_path.exists():
            rs_path, rs_score, rs_step, rs_tokens, rs_loss = (
                best_path,
                best_score,
                best_step,
                best_tokens,
                best_loss,
            )
        elif low_score < float("inf") and low_path.exists():
            rs_path, rs_score, rs_step, rs_tokens, rs_loss = (
                low_path,
                low_score,
                low_step,
                low_tokens,
                low_loss,
            )
        else:
            rs_path = None
        if rs_path is not None and rs_step != step:
            dash.print(
                f"[best] restoring lowest-PPL checkpoint (val PPL {rs_score:.2f} @ "
                f"step {rs_step:,}) over the tail before finishing"
            )
            B.engine.load_weights(model, str(rs_path))
            step, tokens_seen, acc_loss = rs_step, rs_tokens, rs_loss * grad_acc
            restored_best = True

        # Budget exhausted
        save_checkpoint(
            model,
            step,
            stage,
            tokens_seen,
            acc_loss / grad_acc,
            ckpt_dir,
            optimizer,
            log=dash.print,
        )

        if skip_gate:
            # Smoke-test run (e.g. profile=test) — graduation gate not required
            B.engine.save_weights(model, str(ckpt_dir / "final.npz"))
            ckpt_file = str(ckpt_dir / f"step_{step:08d}.npz")
            write_stage_complete(
                ckpt_dir,
                stage=stage,
                step=step,
                tokens_seen=tokens_seen,
                gate_score=None,
                checkpoint=ckpt_file,
                best_val_ppl=(best_score if best_score < float("inf") else None),
                best_step=(best_step if best_score < float("inf") else None),
                restored_best=restored_best,
                entry_ppl=entry_ppl,
                skip_gate=True,
            )
            dash.print(f"✓ Stage {stage} COMPLETE (gate skipped)")
            on_stage_complete(model, stage, cfg, root, ckpt_dir, precision, adapter)
            return True

        score, passed = model_spec.evaluate(model, stage, val_batches, cfg)
        dash.set_gate_result(score, passed)
        if passed:
            B.engine.save_weights(model, str(ckpt_dir / "final.npz"))  # graduated model
            write_stage_complete(
                ckpt_dir,
                stage=stage,
                step=step,
                tokens_seen=tokens_seen,
                gate_score=score,
                gate_threshold=gate_threshold(stage, cfg),
                met_bar=True,
                restored_best=restored_best,
                entry_ppl=entry_ppl,
                checkpoint=str(ckpt_dir / f"step_{step:08d}.npz"),
            )
            dash.print(f"✓ Stage {stage} COMPLETE — best val ppl {score:.4f} (budget reached)")
            on_stage_complete(model, stage, cfg, root, ckpt_dir, precision, adapter)
        else:
            need = gate_threshold(stage, cfg)
            dash.print(
                f"Budget exhausted. Best val ppl {score:.2f} (need ≤ {need:.1f}) "
                f"— run --resume to continue, or --skip-gate to accept the best"
            )
    return passed
