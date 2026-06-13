#!/usr/bin/env python3
import sys, os
try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    venv_py = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           ".venv", "bin", "python")
    if os.path.exists(venv_py) and os.path.abspath(sys.executable) != os.path.abspath(venv_py):
        os.execv(venv_py, [venv_py] + sys.argv)
    print("ERROR: dependencies not found. Run: source .venv/bin/activate")
    sys.exit(1)

"""
RDMCA Progressive Stage Trainer with Checkpoint-Resume
=======================================================
Usage:
  # Start Stage 1 fresh at a given level
  python train_stage.py --level 1 --stage 1

  # Resume Stage 1 after a pause
  python train_stage.py --level 1 --stage 1 --resume

  # After Stage 1 gate passes, start Stage 2
  python train_stage.py --level 1 --stage 2

Each stage must pass its graduation gate before the next can begin.
The foundational core (cognition + values) is frozen permanently after the last
ACTIVE cognitive stage — ethics/BCF (6) when present, else the highest base stage
(e.g. reasoning (5) at level 1). Behavioral stages 7-9 then train on top as LoRA
sectors (loaded + frozen core), so they never overwrite language/reasoning.
"""
import os
import sys
import json
import math
import time
import argparse
import yaml
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import src.backend as backend
from src.config import require_backend, get_precision, SUPPORTED_PRECISIONS

# NOTE: model/data/dashboard modules are imported lazily (inside the functions
# below) — only AFTER require_backend() has selected the compute backend — so
# their classes bind to the configured backend (mlx | torch). Importing them at
# module load would bind to the default backend before selection.


# ---------------------------------------------------------------------------
# Stage gates / names / freeze point — single source of truth (shared with the
# dashboard) lives in src/training/stages.py so the two can't diverge.
from src.training.stages import (STAGE_GATES, STAGE_NAMES, BCF_STAGE,
                                  STAGE_REHEARSAL, DEFAULT_REHEARSAL,
                                  STAGE_LR_SCALE, DEFAULT_LR_SCALE)


def last_cognitive_stage(cfg: dict) -> int | None:
    """Highest ACTIVE stage that is part of the frozen cognitive base (≤ BCF_STAGE).
    The core is frozen right after this stage. Every level now carries the full
    cognitive curriculum (1..7), so the freeze happens after ethics/BCF (7) at
    EVERY level. Behavioral stages (8-10: tool/MCP/skills) then add LoRA sectors."""
    active = [int(k.replace("stage", "")) for k in (cfg.get("curriculum") or {})]
    base = [s for s in active if s <= BCF_STAGE]
    return max(base) if base else None


def is_behavioral_stage(stage: int) -> bool:
    """Behavioral stages (tool/MCP/skills, > BCF_STAGE) train as LoRA sectors on
    the frozen core so they never overwrite language/reasoning."""
    return stage > BCF_STAGE


def stage_name(stage: int, cfg: dict | None = None) -> str:
    """Stage label — prefers the config's per-stage `name`, then STAGE_NAMES,
    then a generic fallback. Keeps new stages working with no code change."""
    if cfg:
        sc = cfg.get("curriculum", {}).get(f"stage{stage}")
        if sc and sc.get("name"):
            return sc["name"]
    return STAGE_NAMES.get(stage, f"Stage {stage}")


def prev_active_stage(stage: int, cfg: dict) -> int | None:
    """Highest curriculum stage below `stage` declared in this config (the real
    predecessor — stages can be non-contiguous, e.g. {1,2,3,6}), or None."""
    below = [int(k.replace("stage", "")) for k in cfg.get("curriculum", {})
             if int(k.replace("stage", "")) < stage]
    return max(below) if below else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_config(path: str) -> dict:
    # Single implementation (deep-merges the shared configs/levels/_base.yaml so levels
    # declare only their diffs) lives in src.config — delegate so the two never diverge.
    from src.config import load_config as _load_config
    return _load_config(path)


def ckpt_root(cfg: dict) -> Path:
    """Checkpoint root, namespaced by level so levels never collide."""
    level = cfg.get("level")                        # NB: level 0 is valid → use `is None`
    return Path("dist/checkpoints") if level is None else Path("dist/checkpoints") / f"level{level}"


def cosine_lr(step: int, base_lr: float, min_lr: float,
              warmup: int, total: int) -> float:
    if step < warmup:
        return base_lr * step / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))


def save_checkpoint(model, step: int, stage: int,
                    tokens_seen: int, loss: float, ckpt_dir: Path, optimizer=None,
                    log=print):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fname = ckpt_dir / f"step_{step:08d}.npz"
    backend.current().engine.save_weights(model, str(fname))
    if optimizer is not None:                   # warm-resume: save AdamW moments too,
        # per-checkpoint (step_NNNN.opt) so a rollback to any step restores its
        # optimizer moments, not just the latest. Costs ~2× the weights per ckpt.
        backend.current().engine.save_optimizer(optimizer, str(fname.with_suffix(".opt")))
    state = {
        "step": step, "stage": stage,
        "tokens_seen": tokens_seen, "loss": round(loss, 6),
        "timestamp": time.time(), "checkpoint": str(fname),
    }
    tmp = ckpt_dir / "latest.json.tmp"          # atomic pointer update
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, ckpt_dir / "latest.json")
    log(f"[ckpt] step={step:,} | {tokens_seen/1e6:.1f}M tokens | "
        f"loss={loss:.4f} -> {fname.name}")


def _git_commit() -> str:
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=str(Path(__file__).parent),
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def write_stage_audit(ckpt_dir: Path, *, stage: int, cfg: dict, model, model_cfg,
                      tcfg: dict, data_loader, target: int, total_steps: int,
                      precision: str, seed, hparams_extra: dict) -> dict:
    """Persist the COMPLETE training CONTEXT for this stage to `audit.json`, so a run
    is fully auditable/reproducible after the fact: exact hyperparameters, data
    provenance (per-source tokens + 'exhausted' flags), the rehearsal mix and its
    size-weights, model geometry, and env/git. The run TIMELINE (loss curve, gates,
    early-stop, COMPLETE) lives in train.log; the OUTCOME in stage_complete.json."""
    from datetime import datetime
    ddir = Path(cfg["curriculum"][f"stage{stage}"].get("data_dir", ""))
    sources = []
    for m in sorted(ddir.glob("*.meta.json")):
        try:
            sources.append({"source": m.name[:-len(".meta.json")], **json.loads(m.read_text())})
        except (OSError, json.JSONDecodeError):
            pass
    rw = getattr(data_loader, "_replay_weights", None)
    rec = {
        "stage": stage, "level": cfg.get("level"), "stage_name": stage_name(stage, cfg),
        "started": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(), "backend": cfg.get("backend"),
        "precision": precision, "seed": seed,
        "command": "python " + " ".join(sys.argv),
        "model": {"params": int(model.count_params()), "d_model": model_cfg.d_model,
                  "n_layers": model_cfg.n_layers, "n_heads": model_cfg.n_heads,
                  "vocab_size": model_cfg.vocab_size, "context_len": model_cfg.context_len,
                  "mrl_dims": list(model_cfg.mrl_dims)},
        "hparams": {"lr": tcfg.get("lr"), "lr_min": tcfg.get("lr_min"),
                    "batch_size": tcfg.get("batch_size"),
                    "grad_accumulation": tcfg.get("grad_accumulation"),
                    "warmup_steps": tcfg.get("warmup_steps"),
                    "max_corpus_passes": tcfg.get("max_corpus_passes"),
                    "clip_grad_norm": tcfg.get("clip_grad_norm"),
                    "save_every": tcfg.get("save_every"), "eval_every": tcfg.get("eval_every"),
                    **hparams_extra},
        "data": {"dir": str(ddir), "target_tokens": int(target),
                 "lr_horizon_steps": int(total_steps), "sources": sources},
        "rehearsal": {"fraction": getattr(data_loader, "replay_fraction", 0.0),
                      "dirs": getattr(data_loader, "replay_dirs", []),
                      "weights_pct": ([round(100 * w / sum(rw), 1) for w in rw] if rw else [])},
    }
    try:
        (ckpt_dir / "audit.json").write_text(json.dumps(rec, indent=2))
    except OSError:
        pass
    return rec


def load_checkpoint(model, ckpt_dir: Path, optimizer=None):
    latest = ckpt_dir / "latest.json"
    if not latest.exists():
        return 0, 0
    with open(latest) as f:
        state = json.load(f)
    backend.current().engine.load_weights(model, state["checkpoint"])
    warm = False
    if optimizer is not None:                   # restore AdamW moments if saved (the
        opt_path = Path(state["checkpoint"]).with_suffix(".opt")   # step's own .opt
        warm = backend.current().engine.load_optimizer(optimizer, str(opt_path))
    print(f"  [resume] step={state['step']:,} | "
          f"{state['tokens_seen']/1e6:.1f}M tokens | loss={state['loss']:.4f}"
          f" | optimizer={'warm' if warm else 'cold (no .opt — will spike briefly)'}")
    return state["step"], state["tokens_seen"]


def _val_split_batches(stage: int, cfg: dict, n: int):
    """`n` (tokens, mask) batches from a stage's held-out split (`*.val.jsonl`), or []
    if there is no usable split (missing / empty / sub-one-batch). Completion-masked."""
    from src.modalities.text import TextTokenizer
    from src.data.loader import DataLoader
    try:
        vloader = DataLoader.from_config(stage, cfg, TextTokenizer(), val=True,
                                         with_mask=True)
        return [vloader.next_batch() for _ in range(n)]
    except (FileNotFoundError, KeyError, StopIteration):
        return []


def _make_val_batches(stage: int, cfg: dict, train_loader, n: int = 8):
    """Return fixed validation batches as (tokens, mask) pairs. The mask is the
    completion-only loss mask, so the gate measures perplexity on the SAME (assistant)
    tokens training optimizes — not the user/system context the model never learns to
    predict (which otherwise inflates val ppl ~7×).

    RETENTION GATE: for a later cognitive stage (2..BCF) the val set FOLDS IN held-out
    CONVERSATION (stage 1) alongside the stage's own data. Without this the gate measures
    only the narrow new skill — stages 2..7 'pass' at ppl ~1 by memorizing templated data
    (arithmetic, CoT, ethics) while the shared core forgets how to converse (the observed
    'hi'→'2' / 'The answer is N' collapse). Mixing conversation in makes a stage that
    erodes it ratchet/fail instead of passing, so the best checkpoint kept is one that
    learned the new skill WITHOUT forgetting. Prefers held-out splits; falls back to the
    training stream for the stage's own slice."""
    own = _val_split_batches(stage, cfg, n)
    src_own = "held-out split" if own else None
    if not own:
        # The training loader already yields (tokens, mask) pairs — mask matches training.
        own = [train_loader.next_batch() for _ in range(n)]
        src_own = "training stream (no *.val.jsonl — run prepare_data for a disjoint gate)"

    if is_behavioral_stage(stage) or stage <= 1:
        print(f"  [val] {src_own} — {len(own)} batches (completion-masked)")
        return own

    # Retention: half conversation (stage 1), half the stage's own skill. Conversation
    # is the priority skill and the one most eroded, so it anchors the gate.
    half = max(n // 2, 1)
    conv = _val_split_batches(1, cfg, half)
    if not conv:
        print(f"  [val] {src_own} — {len(own)} batches; NO stage-1 split for retention "
              f"(run prepare_data on stage 1) — gate measures the new skill only")
        return own
    mixed = conv + own[:half]
    print(f"  [val] RETENTION gate — {len(conv)} conversation (stage 1) + {len(own[:half])} "
          f"stage-{stage} batches (completion-masked); a stage that forgets conversation fails")
    return mixed


def build_data_loader(stage: int, cfg: dict):
    """
    Build a real DataLoader from stage data. Exits with actionable instructions
    if the tokenizer or the stage corpus is missing (no random-batch fallback).
    """
    from src.modalities.text import TextTokenizer
    from src.data.loader import DataLoader
    tokenizer = TextTokenizer()
    if not tokenizer.ready:
        print("ERROR: tokenizer not found at dist/tokenizer/rdmca_spm.model")
        print("  Run: python scripts/train_tokenizer.py --level <N>")
        sys.exit(1)
    # Rehearsal: cognitive stages after the first mix in a fraction of earlier
    # base stages' data, so learning a new faculty (e.g. reasoning) does not erode
    # earlier ones (esp. conversation) before the core is frozen. Behavioral stages
    # need none — they train sectors on the already-frozen core.
    replay_dirs: list[str] = []
    frac = 0.0
    if not is_behavioral_stage(stage):
        # Per-stage override > global default. Format-shifting stages (e.g. stage 6
        # memory, all short factual Q&A) can request MORE rehearsal so they don't
        # erode conversation; see curriculum.stageN.rehearsal_fraction.
        scfg = cfg.get("curriculum", {}).get(f"stage{stage}", {}) or {}
        # Per-stage anti-forgetting default (applies at EVERY level — see stages.py);
        # the level's yaml may override, else the global training default, else 0.15.
        _reh_default = STAGE_REHEARSAL.get(
            stage, cfg.get("training", {}).get("rehearsal_fraction", DEFAULT_REHEARSAL))
        frac = float(scfg.get("rehearsal_fraction", _reh_default))
        if frac > 0:
            cur = cfg.get("curriculum", {})
            earlier = sorted(s for s in (int(k.replace("stage", "")) for k in cur)
                             if s < stage and s <= BCF_STAGE)
            for s in earlier:
                d = cur[f"stage{s}"].get("data_dir") or f"data/level{cfg.get('level')}/stage{s}"
                if Path(d).exists():
                    replay_dirs.append(d)
    try:
        loader = DataLoader.from_config(stage, cfg, tokenizer,
                                        replay_dirs=replay_dirs, replay_fraction=frac,
                                        with_mask=True)   # completion-only loss masking
        loader.replay_dirs = replay_dirs        # expose for the per-stage audit record
        loader.replay_fraction = frac
        data_dir = cfg["curriculum"][f"stage{stage}"].get("data_dir")   # key-based (stages may be non-contiguous)
        print(f"  [data] Real data loader: {data_dir}")
        if replay_dirs:
            print(f"  [rehearsal] mixing {frac:.0%} replay from {len(replay_dirs)} earlier "
                  f"stage(s) to retain prior skills (e.g. conversation)")
        return loader
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print(f"  Run: python scripts/prepare_data.py --level {cfg.get('level', '')} "
              f"--stage {stage}".rstrip())
        sys.exit(1)


def validation_perplexity(model, val_batches) -> float:
    """Mean validation perplexity over a FIXED set of batches (the same arrays on
    every call), so the gate metric is comparable across evals instead of jumping
    around on a fresh random slice each time. Batches are (tokens, mask) pairs; the
    mask makes the CE COMPLETION-ONLY, matching the training objective (bare token
    arrays without a mask are still accepted for back-compat)."""
    B = backend.current()
    losses = []
    for batch in val_batches:
        if isinstance(batch, tuple):
            toks, mask = batch
            loss = model.eval_ce(B.ops.array(toks), mask=B.ops.array(mask))
        else:
            loss = model.eval_ce(B.ops.array(batch))
        B.engine.eval(loss)
        losses.append(B.engine.item(loss))
    return float(np.exp(np.mean(losses)))


# Proxy perplexity gates per stage until task-specific benchmarks
# (BLiMP / ARC / GSM8K / COPA / BCF probes) are wired in. Overridable via
# cfg["gate"]["max_perplexity"][stage].
#
# These are the STARTING POINTS (viability floors), not the quality target. The gate
# RATCHETS: a checkpoint must beat the BEST seen so far to "pass", so once the model
# first drops under the floor the bar tightens to each new best and worse checkpoints
# are discarded. The floor only (a) gates the FIRST pass (a model still above it isn't
# learning yet) and (b) is the minimum the best must clear to graduate. The ratchet —
# not the floor — drives quality, so the floor stays lenient and never soft-locks a
# working model (a working stage-1 sits ~15 masked, well under 50, then ratchets down).
DEFAULT_GATE_PPL = {1: 50.0, 2: 45.0, 3: 40.0, 4: 38.0, 5: 36.0,
                    6: 36.0, 7: 35.0}   # 6 = memory, 7 = ethics/BCF


def gate_threshold(stage: int, cfg: dict = None) -> float:
    """The stage's STARTING-POINT floor (cfg.gate.max_perplexity > default): the bar the
    first pass must clear and the minimum the best must clear to graduate. The in-loop
    gate then RATCHETS below it (beat-your-own-best); see gate_decision."""
    gate_cfg = (cfg or {}).get("gate", {})
    return gate_cfg.get("max_perplexity", {}).get(stage, DEFAULT_GATE_PPL.get(stage, 40.0))


def gate_decision(score: float, best_score: float, threshold: float,
                  min_delta: float = 0.002) -> tuple:
    """Ratcheting graduation gate. Returns (is_candidate, is_new_best, is_meaningful):
      • is_candidate  — `score` clears the absolute floor (`threshold`, the starting
        point), so it is ELIGIBLE to be a best at all. A checkpoint ABOVE the floor is
        not viable yet and can NEVER become the best — no matter how much it improves on
        a worse above-floor attempt it is not "the best" (the user's rule: it isn't the
        best if it didn't even pass the default gate).
      • is_new_best   — is_candidate AND STRICTLY beats the running best. ANY genuine
        improvement is a new best and is SAVED (the ratchet bar moves down). So a 16.11
        always replaces a 16.14 — a better checkpoint is NEVER discarded (the user's
        confusion: '16.11 ≥ best 16.14 → discarded' was wrong; min_delta did that).
      • is_meaningful — is_new_best AND the gain exceeds `min_delta` (relative). This does
        NOT gate saving; it only governs PLATEAU/early-stop: a string of sub-min_delta
        improvements still saves each new best but counts toward the plateau so the stage
        eventually graduates instead of chasing noise. min_delta is small (0.2%) so real
        late-stage gains (which shrink with steps) still count as meaningful progress.
    Example (floor 50): 55→(F,F,F) above floor; 35 from ∞→(T,T,T) bar 35; 30→(T,T,T) bar
    30; 29.97 vs 30→(T,T,F) saved (new bar 29.97) but a plateauing tick; 31→(T,F,F)
    worse, discarded."""
    is_candidate  = math.isfinite(score) and score <= threshold
    is_new_best   = is_candidate and score < best_score
    is_meaningful = is_new_best and score < best_score * (1.0 - min_delta)
    return is_candidate, is_new_best, is_meaningful


def evaluate_gate(model, stage: int,
                  val_batches=None, cfg: dict = None, log=print, step=None) -> tuple:
    """
    Absolute graduation check. Operative metric is real validation perplexity (a proxy
    that actually measures the model); task-specific benchmarks (BLiMP, ARC, GSM8K,
    COPA, BCF probes) should replace the per-stage threshold as they are wired in. The
    ethics/BCF stage additionally checks BCF probe accuracy when a probe set is
    available. Returns (score, meets_bar) where meets_bar = ppl ≤ the absolute minimum.

    NB this is the ABSOLUTE check (good enough?). The training loop ALSO applies a
    RATCHET each eval — a checkpoint "passes" only if it beats the best seen so far, so
    a worse checkpoint is marked not-passed and discarded; the best is the moving bar.
    """
    # Post-base behavioral stages (tool use / MCP / skills / reasoning) have no
    # benchmark gate — fall back to the perplexity-only proxy with a generic label.
    gate = STAGE_GATES.get(stage)
    desc = gate[2] if gate else stage_name(stage, cfg)
    max_ppl  = gate_threshold(stage, cfg)

    # Measure with dropout OFF (eval mode), then restore training mode — the gate
    # is called mid-training, so validation must not see dropout noise.
    B = backend.current()
    B.engine.set_eval(model)
    ppl = validation_perplexity(model, val_batches)
    B.engine.set_train(model)
    passed = ppl <= max_ppl
    # One self-identifying line (with the step) so it reads in order next to the
    # [ckpt] lines instead of two stepless lines floating between checkpoints.
    tag = f"step={step:,} | " if step is not None else ""
    log(f"[gate] {tag}val perplexity={ppl:.2f} <= {max_ppl:.1f} "
        f"-> {'PASS' if passed else 'fail'}  ({desc})")

    if stage == BCF_STAGE:
        passed = passed and _bcf_gate(model, cfg)
    return ppl, passed


def _bcf_gate(model, cfg: dict) -> bool:
    """Stage-5 BCF probe-accuracy gate (>= 0.90) when probes are available."""
    probe_path = Path("data/benchmarks/bcf_probes.jsonl")
    if not probe_path.exists():
        print("  [gate] BCF probes not found — skipping BCF accuracy check")
        return True
    from src.model.bcf import BCFHead, bcf_accuracy
    from src.modalities.text import TextTokenizer
    probes = []
    with open(probe_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            probes.append((rec["text"], int(rec["label"])))
    head = getattr(model, "bcf_head", None) or BCFHead(model.cfg.d_model)
    acc  = bcf_accuracy(model, TextTokenizer(), head, probes)
    print(f"  [gate] BCF probe accuracy={acc:.3f} | threshold>= 0.90")
    return acc >= 0.90


def train_bcf_head(model, ckpt_dir: Path, precision: str = "fp32",
                   epochs: int = 30, batch: int = 16) -> None:
    """
    Train the Behavioral Constraint head on the probe set over frozen-core
    features (§15.3). Runs only if data/benchmarks/bcf_probes.jsonl exists;
    the trained head is stored on model.bcf_head and saved beside the stage.
    """
    probe_path = Path("data/benchmarks/bcf_probes.jsonl")
    if not probe_path.exists():
        print("  [bcf] No probe set — skipping BCF head training "
              "(expected data/benchmarks/bcf_probes.jsonl)")
        return
    from src.model.bcf import BCFHead, bcf_train_step, bcf_accuracy
    from src.modalities.text import TextTokenizer
    probes = []
    with open(probe_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                probes.append((rec["text"], int(rec["label"])))

    head = BCFHead(model.cfg.d_model)
    backend.current().engine.set_precision(head, precision)   # match model device/dtype
    model.bcf_head = head                       # attach for gate + pipeline use
    opt = backend.current().engine.make_optimizer(head, lr=1e-3, weight_decay=0.0)
    tok = TextTokenizer()
    print(f"  [bcf] Training BCF head on {len(probes)} probes, {epochs} epochs")
    for ep in range(epochs):
        np.random.shuffle(probes)
        for i in range(0, len(probes), batch):
            bcf_train_step(model, tok, head, probes[i:i + batch], opt)
    acc = bcf_accuracy(model, tok, head, probes)
    print(f"  [bcf] final probe accuracy={acc:.3f}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    backend.current().engine.save_weights(head, str(ckpt_dir / "bcf_head.npz"))


def freeze_model(model, ckpt_dir: Path):
    """Permanently freeze the foundational core after the ethics/BCF stage."""
    print("\n" + "=" * 60)
    print("  FREEZING FOUNDATIONAL CORE — Theta_F locked forever")
    backend.current().engine.freeze_all(model)   # excludes params from trainable set
    n = model.count_params()
    print(f"  {n/1e6:.1f}M parameters frozen")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    backend.current().engine.save_weights(model, str(ckpt_dir / "theta_f_frozen.npz"))
    with open(ckpt_dir / "frozen.json", "w") as f:
        json.dump({"frozen": True, "params": n,
                   "timestamp": time.time()}, f, indent=2)
    print(f"  Saved: {ckpt_dir}/theta_f_frozen.npz")
    print("=" * 60 + "\n")


def _on_stage_complete(model, stage: int, cfg: dict, root: Path, ckpt_dir: Path,
                       precision: str, adapter=None) -> None:
    """Side effects when a stage finishes: a behavioral stage persists its trained
    sector; the last cognitive stage trains the BCF head (if ethics is active)
    and freezes the foundational core. This is the single freeze/sector seam."""
    from src.model import sector_io
    if is_behavioral_stage(stage):
        if adapter is not None:
            print(f"  Behavioral sector saved: {sector_io.save_sector(adapter, root, stage)}")
        return
    # Cognitive stage finished: train the conversation mood head on this checkpoint's
    # core so the stage is chat-ready with mood tracking — no separate script needed.
    _maybe_train_mood_head(model, stage, cfg, ckpt_dir, precision)
    if stage == last_cognitive_stage(cfg):
        if stage == BCF_STAGE:
            train_bcf_head(model, ckpt_dir, precision)
        freeze_model(model, root / "foundational")


def _maybe_train_mood_head(model, stage: int, cfg: dict, ckpt_dir: Path,
                           precision: str) -> None:
    """Train + save the conversation mood head beside this stage's checkpoint. Best-
    effort and OFF the critical path: gated by `training.mood_head` (default on) and
    silently skipped if the labeled data is unavailable (e.g. offline) — it must never
    fail a finished stage."""
    if not cfg.get("training", {}).get("mood_head", True):
        return
    try:
        from src.model.mood import train_mood_head as _train_mood
        from src.modalities.text import TextTokenizer
        _train_mood(model, TextTokenizer(), ckpt_dir,
                    level=cfg.get("level"), stage=stage, precision=precision)
    except Exception as e:
        print(f"  [mood] skipped ({type(e).__name__}: {e})")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train_stage(stage: int, cfg: dict, resume: bool = False, plain: bool = False) -> bool:
    tcfg      = cfg["training"]
    mcfg      = cfg["model"]
    skip_gate = cfg.get("skip_gate", False)   # toy config sets this to true
    n_tokens_target = cfg["curriculum"][f"stage{stage}"]["n_tokens"]   # key-based
    root      = ckpt_root(cfg)
    ckpt_dir  = root / f"stage{stage}"

    def _fmt_tokens(n: int) -> str:
        if n >= 1_000_000_000:
            return f"{n/1e9:.2f}B"
        if n >= 1_000_000:
            return f"{n/1e6:.0f}M"
        return f"{n/1e3:.0f}K"

    print(f"\n{'='*60}")
    print(f"  Stage {stage}: {stage_name(stage, cfg)}")
    print(f"  Target: {_fmt_tokens(n_tokens_target)} tokens")
    print(f"{'='*60}")

    # Build model — override vocab_size from trained tokenizer if available
    tok_info = Path("dist/tokenizer/tokenizer_info.json")
    if tok_info.exists():
        with open(tok_info) as f:
            info = json.load(f)
        # The unified multimodal layout reserves IDs 0..20479 (text+image+audio),
        # but the text tokenizer only ever emits IDs < text_vocab_size (8192).
        # Sizing the embedding/head to the full 20480 leaves ~60% of rows without
        # gradient and lets phantom image/audio logits steal softmax mass, which
        # produces incoherent text. Train the text head at the real text vocab.
        actual_vocab = info.get("text_vocab_size", info["vocab_size"])
        if actual_vocab != mcfg.get("vocab_size"):
            print(f"  [vocab] Using tokenizer vocab_size={actual_vocab} "
                  f"(config had {mcfg.get('vocab_size')})")
            mcfg = dict(mcfg)
            mcfg["vocab_size"] = actual_vocab

    # Backend already selected by require_backend(); import model modules now so
    # their classes bind to it.
    from src.model.transformer import RDMCAFoundational, set_model_precision
    from src.model.config import ModelConfig
    from src.training.dashboard import TrainingDashboard
    B = backend.current()

    # Reproducibility: seed every RNG (Python/numpy/backend) BEFORE weight init so a
    # run is repeatable and gates are comparable. Configurable via `seed:` (top-level
    # or training.seed); fixed default keeps runs comparable across machines.
    seed = int(cfg.get("seed", (cfg.get("training", {}) or {}).get("seed", 42)))
    B.engine.set_seed(seed)

    model_cfg = ModelConfig(**{k: v for k, v in mcfg.items()
                               if k in ModelConfig.__dataclass_fields__})
    model = RDMCAFoundational(model_cfg)
    precision = get_precision(cfg)
    set_model_precision(model, precision)
    print(f"  Model: {model.count_params()/1e6:.1f}M params | "
          f"d_model={model_cfg.d_model} | layers={model_cfg.n_layers} | "
          f"vocab={model_cfg.vocab_size} | precision={precision}")

    # Starting weights. Cognitive stages continue from the previous active stage.
    # Behavioral stages (tool/MCP/skills) instead load the FROZEN cognitive core
    # and train a LoRA sector on top of it — so language/reasoning is preserved.
    adapter = None
    if is_behavioral_stage(stage):
        from src.model import sector_io
        core = sector_io.frozen_core_path(root)
        if not core.exists():
            print(f"ERROR: behavioral stage {stage} needs the frozen cognitive core, "
                  f"but it is missing:\n  {core}")
            print(f"  Train the cognitive base first (through stage "
                  f"{last_cognitive_stage(cfg)}) — that freezes the core.")
            sys.exit(1)
        B.engine.load_weights(model, str(core))
        set_model_precision(model, precision)
        sid, adapter = sector_io.attach_for_training(model, stage)
        print(f"  Loaded frozen core; training behavioral sector S{sid} "
              f"({B.engine.param_count(adapter)/1e3:.0f}K trainable params) on the frozen base")
    else:
        prev_n = prev_active_stage(stage, cfg)
        if prev_n is not None:
            prev_ckpt = root / f"stage{prev_n}" / "latest.json"
            if prev_ckpt.exists():
                with open(prev_ckpt) as f:
                    prev_state = json.load(f)
                B.engine.load_weights(model, prev_state["checkpoint"])
                print(f"  Loaded Stage {prev_n} weights as starting point")
            else:
                print(f"  No Stage {prev_n} checkpoint found — starting from random init")

    # optimizer_states: bf16 (default — already in effect since the model is bf16)
    # or int8 (bitsandbytes 8-bit AdamW on CUDA, a further memory saving for big runs;
    # opt-in, falls back to bf16 with a warning when unavailable).
    optimizer = B.engine.make_optimizer(
        model, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"],
        states=tcfg.get("optimizer_states"))

    start_step = 0
    tokens_seen = 0
    if resume:
        start_step, tokens_seen = load_checkpoint(model, ckpt_dir, optimizer)

    # Re-apply precision: loading prev-stage / resume weights restores their
    # saved dtype, so cast once more before training in the configured precision.
    set_model_precision(model, precision)

    # Real data loader
    data_loader = build_data_loader(stage, cfg)

    # Derived constants
    bs        = tcfg["batch_size"]
    grad_acc  = tcfg["grad_accumulation"]
    clip_norm = tcfg.get("clip_grad_norm")     # global-norm grad clip (None = off)
    seq_len   = model_cfg.context_len
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
        print(f"  [resume] fast-forwarding data stream past {n_skip:,} batches "
              f"({start_step:,} steps){' — cached lengths' if fast else ' — re-tokenizing (no index)'}…")
        skipped = data_loader.skip(n_skip)
        if skipped < n_skip:
            print(f"  [resume] stream shorter than expected — skipped {skipped:,}.")
    warmup    = tcfg["warmup_steps"]
    total_steps = n_tokens_target // toks_step
    # Cap re-cycling of a small corpus: training a tiny corpus toward an oversized
    # token target just re-reads it many times → overfit/parroting. Once we know
    # the corpus size (after the first pass) we lower the effective target to at
    # most `max_corpus_passes` reads, so the run completes (and the bar reaches
    # 100%) against a budget the data can actually support.
    max_passes = int(tcfg.get("max_corpus_passes", 3))
    target     = n_tokens_target
    capped     = False
    save_every  = tcfg["save_every"]
    eval_every  = tcfg["eval_every"]

    # Anchor the LR horizon up front by estimating the corpus size from disk, so
    # cosine decays toward the REAL (possibly cap-limited) end from step 1. Doing
    # this only at runtime when the cap triggers would shift total_steps mid-run
    # and make the schedule jump discontinuously (M3). The estimate is approximate
    # (bytes/≈chars-per-token); it only sets the LR horizon, not the stop point.
    stage_cfg = cfg["curriculum"][f"stage{stage}"]
    # Per-stage LR scale: the narrow late cognitive stages (arithmetic/CoT) OVERWRITE the
    # shared core at the full LR — even with heavy rehearsal, arithmetic leaked numbers
    # into greetings ('hi'→'3'). A gentler LR makes them NUDGE the core (learn the skill)
    # instead of stamping their low-entropy format over conversation. Default is a STAGE
    # property (applies at every level — see stages.py); the yaml may override per stage.
    lr_scale = float(stage_cfg.get("lr_scale", STAGE_LR_SCALE.get(stage, DEFAULT_LR_SCALE)))
    base_lr  = tcfg["lr"] * lr_scale
    min_lr   = tcfg.get("lr_min", 3e-5) * lr_scale
    _lvl  = cfg.get("level")
    _ddir = Path(stage_cfg.get("data_dir",
                 f"data/level{_lvl}/stage{stage}" if _lvl is not None
                 else f"data/stage{stage}"))
    _bytes = sum(f.stat().st_size for f in _ddir.glob("*.jsonl")) if _ddir.exists() else 0
    _est_corpus = int(_bytes / 3.5)                      # ≈ chars/token (prose-weighted)
    if _est_corpus:
        eff_target  = min(n_tokens_target, max_passes * _est_corpus)
        total_steps = max(eff_target // toks_step, 1)

    step = start_step
    dash_interval = 10    # update dashboard (and measure tok/s) every N steps
    t_dash = time.time()
    last_tps = 0.0
    # NaN/Inf guard: a single unstable batch (fp16 underflow, a bad sample, an LR
    # spike) can blow the loss to NaN; without a guard the optimizer then writes
    # NaN into every weight and the run silently continues training garbage for
    # hours. We SKIP the update on a non-finite loss and abort if it persists.
    import math
    _NAN_ABORT  = 20      # consecutive non-finite losses ⇒ the run has diverged
    nan_streak  = 0

    # Best-checkpoint tracking (anti-divergence). Stages whose data is narrow and
    # format-shifting (esp. stage 6 memory) MEMORIZE then DIVERGE as they re-cycle
    # the corpus — train loss fell to ~0.4 then climbed to 4.4 in the reported run.
    # We remember the lowest-val-perplexity point, and at stage end (or after
    # `early_stop_patience` evals with no improvement) we RESTORE it instead of
    # shipping the diverged tail. patience=0 disables the in-loop early stop but
    # best-restore at stage end still applies. Independent of the graduation gate.
    patience    = int(tcfg.get("early_stop_patience", 4))
    min_delta   = float(tcfg.get("early_stop_min_delta", 0.002))   # 0.2% PPL gain = "meaningful"
                                                                   # (only for plateau; any gain still saves)
    best_path   = ckpt_dir / "best.npz"
    best_meta   = ckpt_dir / "best.json"
    best_score  = float("inf")
    best_step   = start_step
    best_tokens = tokens_seen
    best_loss   = 0.0
    stale       = 0
    # Plateau reference for early-stop. We compare CUMULATIVE improvement against this
    # reference (which only moves on a meaningful drop), NOT step-to-step — because late
    # gains shrink toward 0, a per-step min_delta would declare a plateau while the model
    # is still slowly improving. With a cumulative reference, many tiny-but-real gains add
    # up and keep training; only a genuine flatline (no min_delta drop over `patience`
    # evals) graduates. Every strict improvement is still SAVED as the best regardless.
    plateau_ref = float("inf")
    restored_best = False        # set when the diverged tail is rolled back to best
    # Anti-divergence RESTORE point: the lowest val-PPL seen REGARDLESS of the floor.
    # Distinct from `best` — a point above the floor is NOT a graduation best (it never
    # becomes best.npz and the use cases never load it as "best"), but we still keep its
    # weights so a budget-exhausted run that never reached the floor ships its lowest
    # point instead of a diverged tail.
    low_path    = ckpt_dir / "restore.npz"
    low_score   = float("inf")
    low_step    = start_step
    low_tokens  = tokens_seen
    low_loss    = 0.0
    # PERSISTENT monotonic best: the best checkpoint is the bar to beat ACROSS runs, not
    # just within one. On --resume we load the historical best so a later run REPLACES it
    # only on a genuine improvement — a worse checkpoint is discarded and never clobbers
    # the best (the "a new checkpoint must beat the best, else it's dropped" rule). The
    # absolute graduation gate is separate (it decides when the stage is good enough).
    if resume and best_meta.exists() and best_path.exists():
        try:
            bm = json.loads(best_meta.read_text())
            best_score, best_step = float(bm["score"]), int(bm["step"])
            best_tokens, best_loss = int(bm["tokens"]), float(bm["loss"])
            # The historical best is, by construction, a floor-clearing candidate and
            # thus also the lowest seen — carry it into the restore tracker too.
            low_score, low_step = best_score, best_step
            low_tokens, low_loss = best_tokens, best_loss
            plateau_ref = best_score          # measure further progress from here
            print(f"  [best] resuming against historical best val PPL {best_score:.2f} "
                  f"(step {best_step:,}) — only an improvement replaces it")
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            pass

    def _save_best(score: float, step: int, tokens: int, loss: float) -> None:
        """Persist the new best weights + its metadata so the bar survives across runs."""
        B.engine.save_weights(model, str(best_path))
        try:
            best_meta.write_text(json.dumps(
                {"score": score, "step": step, "tokens": tokens,
                 "loss": loss, "timestamp": time.time()}, indent=2))
        except OSError:
            pass

    # MoE load-balance aux loss: with routing active (behavioral stages) the gate
    # otherwise gets no gradient and experts collapse. aux_loss() is 0.0 when there
    # is no routing (cognitive stages), so this is a no-op there.
    aux_w = float((cfg.get("moe", {}) or {}).get("aux_loss_weight", 0.01))

    def loss_fn(mdl, batch):
        toks, mask = batch                     # (tokens, loss_mask) — completion-only CE
        return mdl.mrl_loss(toks, mask) + aux_w * mdl.aux_loss()

    loss_and_grad_fn = B.engine.value_and_grad(model, loss_fn)

    # Fixed validation batches, sampled ONCE up front and reused for every gate
    # eval (a frozen set keeps the metric stable/comparable across evals). Prefer a
    # true HELD-OUT split (`*.val.jsonl`, never trained on) so the gate measures
    # generalization; fall back to the training stream when no split was prepared.
    val_batches = _make_val_batches(stage, cfg, data_loader, n=8)

    # Persist the full training CONTEXT (config, data provenance, rehearsal mix,
    # model geometry, env/git) for post-hoc audits — see write_stage_audit.
    audit = write_stage_audit(
        ckpt_dir, stage=stage, cfg=cfg, model=model, model_cfg=model_cfg, tcfg=tcfg,
        data_loader=data_loader, target=target, total_steps=total_steps,
        precision=precision, seed=seed,
        hparams_extra={"rehearsal_fraction": getattr(data_loader, "replay_fraction", 0.0),
                       "early_stop_patience": patience, "early_stop_min_delta": min_delta})

    # The dashboard's per-token perplexity divides the COMPOSITE loss by its CE-unit
    # weight: the MRL head mean is 1 CE-unit, each MTP head adds `mtp_loss_weight`.
    # Without this, exp(loss) reports a wildly inflated PP (e.g. 184K at init).
    _n_mtp = int(getattr(model_cfg, "n_mtp_heads", 0) or 0)
    _mtp_w = float(getattr(model_cfg, "mtp_loss_weight", 0.0) or 0.0)
    loss_ce_weight = 1.0 + _n_mtp * _mtp_w

    dash = TrainingDashboard(stage, n_tokens_target,
                             resume_step=start_step,
                             resume_tokens=tokens_seen,
                             params=model.count_params(),
                             n_layers=model.cfg.n_layers,
                             d_model=model.cfg.d_model,
                             plain=plain,
                             log_path=ckpt_dir / "train.log",
                             loss_ce_weight=loss_ce_weight,
                             append=resume)   # fresh run truncates log+metrics; --resume appends

    with dash:
        dash.print(f"Stage {stage} | {model.count_params()/1e6:.1f}M params | real data")
        # Echo the audit context into train.log so the text log is self-contained.
        dash.print(f"[audit] full context → {ckpt_dir/'audit.json'} | git {audit['git_commit']} "
                   f"| seed {seed} | precision {precision}")
        dash.print("[data] sources: " + (", ".join(
            f"{s['source']}={s.get('tokens',0)/1e6:.1f}M{'*' if s.get('exhausted') else ''}"
            for s in audit["data"]["sources"]) or "(none)"))
        if audit["rehearsal"]["dirs"]:
            dash.print(f"[rehearsal] {audit['rehearsal']['fraction']:.0%} replay over "
                       f"{len(audit['rehearsal']['dirs'])} stage(s), weights%="
                       f"{audit['rehearsal']['weights_pct']}")
        _lrtag = f"lr={base_lr:.2e}" + (f" (×{lr_scale:g} stage scale)" if lr_scale != 1.0
                                        else "")
        dash.print(f"[hparams] {_lrtag} bs={bs} warmup={warmup} "
                   f"max_passes={max_passes} early_stop=patience{patience}/δ{min_delta}")

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
                dash.print(f"  [nan] non-finite loss ({acc_loss}) at step {step} — "
                           f"skipped update ({nan_streak}/{_NAN_ABORT})")
                if nan_streak >= _NAN_ABORT:
                    raise RuntimeError(
                        f"Training diverged: {_NAN_ABORT} consecutive non-finite "
                        "losses. Aborting before the checkpoint is corrupted — lower "
                        "the learning rate or check the data/precision.")
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

            step         += 1
            tokens_seen  += toks_step

            # After the first full pass, cap the effective target to the corpus
            # size × max_passes (only if that's *below* the configured target).
            if not capped and data_loader.epoch_tokens:
                capped = True
                cap = max_passes * data_loader.epoch_tokens
                if cap < n_tokens_target:
                    target = cap            # stop point; LR horizon was set up front
                    dash.set_target(target)
                    dash.print(f"[corpus] {data_loader.epoch_tokens/1e6:.1f}M tokens/pass "
                               f"— capping at {max_passes}× ({_fmt_tokens(cap)}) to avoid "
                               f"overfitting the configured {_fmt_tokens(n_tokens_target)} target")

            # Dashboard refresh every dash_interval steps (smooth). Throughput is
            # measured over this same short window so the Speed field is populated
            # within the first dash_interval steps — otherwise it reads 0.0 tok/s
            # until step 100, which looks like a hung run (especially on the torch
            # MPS backend, whose cold first-step kernel compile is already slow).
            # (The dashboard keeps its own loss moving-average from update() — no
            # separate running_loss window is needed here.)
            if step % dash_interval == 0:
                elapsed  = time.time() - t_dash
                if elapsed > 0:
                    last_tps = (dash_interval * toks_step) / elapsed
                t_dash = time.time()
                dash.update(step, tokens_seen, acc_loss / grad_acc, lr, last_tps,
                            grad_norm=grad_norm_val, passes=data_loader.passes,
                            replay=getattr(data_loader, "last_was_replay", False))

            # Checkpoint. Route the [ckpt]/[gate] detail through dash.print so it
            # lands INSIDE the dashboard's log region (raw print() would draw over
            # the pinned panel — the "broken UI" / stray lines outside the log box).
            if step % save_every == 0:
                save_checkpoint(model, step, stage, tokens_seen,
                               acc_loss / grad_acc, ckpt_dir, optimizer, log=dash.print)
                # Persist the data-stream length index alongside the checkpoint so a
                # later --resume can fast-forward without re-tokenizing the span.
                data_loader.save_skip_index(skip_index_path)
                dash.set_checkpoint(step)

            # RATCHETING gate. Each eval the score must BEAT the best seen so far to
            # "pass" — the best is the moving bar, so a worse checkpoint is marked
            # not-passed and DISCARDED (the best.npz is never clobbered by it). The
            # stage does NOT graduate on a single static-threshold crossing (that would
            # ship the first mediocre checkpoint over 50); it keeps seeking a better
            # best and GRADUATES when it plateaus (no new best for `patience` evals),
            # shipping that best — but only if the best clears the absolute quality bar.
            if step % eval_every == 0:
                # meets_bar = absolute quality floor; quiet log in-loop (we print the
                # ratchet line below). skip_gate levels keep the plain informational line.
                score, meets_bar = evaluate_gate(
                    model, stage, val_batches, cfg,
                    log=(dash.print if skip_gate else (lambda *a, **k: None)), step=step)
                thr = gate_threshold(stage, cfg)
                # Ratchet: is_candidate = cleared the floor (eligible to be a best);
                # is_new_best = candidate AND strictly beats the best (ANY gain → saved);
                # is_meaningful = gain beyond min_delta (drives plateau only). ONLY a
                # floor-clearing checkpoint can be the best — an above-floor point is not
                # viable yet and is never saved/labelled "best".
                is_candidate, is_new_best, _ = gate_decision(score, best_score, thr, min_delta)
                dash.set_gate_result(score, is_new_best if not skip_gate else meets_bar,
                                     threshold=thr, best=best_score)
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
                            dash.print(f"[gate] step={step:,} | val ppl {score:.2f} < best "
                                       f"{prev} → PASSED, new best (bar↓)")
                        else:
                            ref = "∞" if plateau_ref == float("inf") else f"{plateau_ref:.2f}"
                            dash.print(f"[gate] step={step:,} | val ppl {score:.2f} < best "
                                       f"{prev} → new best, SAVED — but cumulative gain since "
                                       f"{ref} < {min_delta:.1%} min_delta, plateauing "
                                       f"({stale + 1}/{patience})")
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
                            bar = ("no best yet" if best_score == float("inf")
                                   else f"best {best_score:.2f}")
                            dash.print(f"[gate] step={step:,} | val ppl {score:.2f} > floor "
                                       f"{thr:.1f} → not viable yet, NOT a best ({bar})")
                        elif math.isfinite(best_score):
                            dash.print(f"[gate] step={step:,} | val ppl {score:.2f} ≥ best "
                                       f"{best_score:.2f} → no improvement, discarded "
                                       f"({stale + 1}/{patience})")
                    if math.isfinite(best_score):
                        stale += 1

                # Plateau (counts only once a floor-clearing best exists) → ship the best.
                # The best is, by construction, a floor-clearing candidate, so it graduates.
                if math.isfinite(best_score) and patience and stale >= patience:
                    B.engine.load_weights(model, str(best_path))
                    save_checkpoint(model, best_step, stage, best_tokens,
                                    best_loss, ckpt_dir, log=dash.print)
                    B.engine.save_weights(model, str(ckpt_dir / "final.npz"))   # graduated
                    with open(ckpt_dir / "stage_complete.json", "w") as f:
                        json.dump({"stage": stage, "step": best_step,
                                   "tokens_seen": best_tokens, "gate_score": best_score,
                                   "gate_threshold": thr, "met_bar": True,
                                   "checkpoint": str(ckpt_dir / f"step_{best_step:08d}.npz"),
                                   "early_stopped": True, "timestamp": time.time()}, f, indent=2)
                    dash.print(f"✓ Stage {stage} COMPLETE — best val ppl "
                               f"{best_score:.2f} (≤ floor {thr:.1f}), plateaued")
                    _on_stage_complete(model, stage, cfg, root, ckpt_dir, precision, adapter)
                    return True

        # Final dashboard update so it shows 100%
        dash.update(step, tokens_seen, acc_loss / grad_acc, lr, last_tps)

        # Budget exhausted — ship the lowest-val-PPL point, not a memorized/diverged tail
        # (stage 6's 0.4→4.4 climb). Prefer the graduation best (a floor-clearing
        # candidate); if the floor was NEVER reached, fall back to the lowest-seen restore
        # point so we still avoid the diverged tail.
        if best_score < float("inf") and best_path.exists():
            rs_path, rs_score, rs_step, rs_tokens, rs_loss = (
                best_path, best_score, best_step, best_tokens, best_loss)
        elif low_score < float("inf") and low_path.exists():
            rs_path, rs_score, rs_step, rs_tokens, rs_loss = (
                low_path, low_score, low_step, low_tokens, low_loss)
        else:
            rs_path = None
        if rs_path is not None and rs_step != step:
            dash.print(f"[best] restoring lowest-PPL checkpoint (val PPL {rs_score:.2f} @ "
                       f"step {rs_step:,}) over the tail before finishing")
            B.engine.load_weights(model, str(rs_path))
            step, tokens_seen, acc_loss = rs_step, rs_tokens, rs_loss * grad_acc
            restored_best = True

        # Budget exhausted
        save_checkpoint(model, step, stage, tokens_seen,
                       acc_loss / grad_acc, ckpt_dir, optimizer, log=dash.print)

        if skip_gate:
            # Smoke-test run (e.g. profile=test) — graduation gate not required
            B.engine.save_weights(model, str(ckpt_dir / "final.npz"))
            ckpt_file = str(ckpt_dir / f"step_{step:08d}.npz")
            with open(ckpt_dir / "stage_complete.json", "w") as f:
                json.dump({"stage": stage, "step": step,
                           "tokens_seen": tokens_seen, "gate_score": None,
                           "checkpoint": ckpt_file,
                           "best_val_ppl": (best_score if best_score < float("inf") else None),
                           "best_step": (best_step if best_score < float("inf") else None),
                           "restored_best": restored_best,
                           "skip_gate": True, "timestamp": time.time()}, f, indent=2)
            dash.print(f"✓ Stage {stage} COMPLETE (gate skipped)")
            _on_stage_complete(model, stage, cfg, root, ckpt_dir, precision, adapter)
            return True

        score, passed = evaluate_gate(model, stage, val_batches, cfg)
        dash.set_gate_result(score, passed)
        if passed:
            B.engine.save_weights(model, str(ckpt_dir / "final.npz"))   # graduated model
            with open(ckpt_dir / "stage_complete.json", "w") as f:
                json.dump({"stage": stage, "step": step, "tokens_seen": tokens_seen,
                           "gate_score": score, "gate_threshold": gate_threshold(stage, cfg),
                           "met_bar": True, "restored_best": restored_best,
                           "checkpoint": str(ckpt_dir / f"step_{step:08d}.npz"),
                           "timestamp": time.time()}, f, indent=2)
            dash.print(f"✓ Stage {stage} COMPLETE — best val ppl {score:.4f} (budget reached)")
            _on_stage_complete(model, stage, cfg, root, ckpt_dir, precision, adapter)
        else:
            need = gate_threshold(stage, cfg)
            dash.print(f"Budget exhausted. Best val ppl {score:.2f} (need ≤ {need:.1f}) "
                       f"— run --resume to continue, or --skip-gate to accept the best")
    return passed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="RDMCA Progressive Stage Trainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_stage.py --level 1 --stage 1
  python train_stage.py --level 1 --stage 1 --resume
  python train_stage.py --level 2 --stage 2
        """
    )
    parser.add_argument("--stage",  type=int, required=True,
                        help="Curriculum stage number (validated against the level's config)")
    parser.add_argument("--level", type=int, default=None,
                        help="Educational level 1-5 (preescolar..universidad). "
                             "Determines model size, data and resources.")
    parser.add_argument("--config", type=str, default=None,
                        help="Explicit config path (overrides --level)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint in stage dir")
    parser.add_argument("--force", action="store_true",
                        help="Run even if the resource guard says it won't fit (risk OOM)")
    parser.add_argument("--precision", choices=SUPPORTED_PRECISIONS, default=None,
                        help="Override training precision (fp32|bf16|fp16). Lower precision "
                             "uses less memory, so a bigger level may fit on the same hardware.")
    parser.add_argument("--plain", action="store_true",
                        help="Plain scrolling logs instead of the live dashboard (selectable/"
                             "copyable, no flicker). A full train.log is written to the stage's "
                             "checkpoint dir either way (also via RDMCA_PLAIN_LOGS=1).")
    parser.add_argument("--skip-gate", dest="skip_gate", action="store_true",
                        help="Manually disable the graduation gate for this run (the gate is "
                             "ENFORCED from level 1 by default — quality first). Lets a stage "
                             "complete at its token budget without meeting the perplexity bar.")
    args = parser.parse_args()

    from src.config import resolve_config_path
    cfg_path = resolve_config_path(args.config, args.level)
    cfg = load_config(cfg_path)
    # Precision override (CLI wins over config). Set before the guard/announce so
    # the memory estimate — which is precision-aware — reflects the chosen dtype:
    # dropping fp32→bf16 roughly halves weight/grad/activation bytes, letting a
    # larger level fit on the same hardware.
    if args.precision:
        cfg.setdefault("training", {})["precision"] = args.precision
    # Manual gate override (CLI wins): force-disable the graduation gate for this run.
    if args.skip_gate:
        cfg["skip_gate"] = True
        print("  [gate] manually disabled for this run (--skip-gate)")
    level = cfg.get("level", "?")
    active_backend = require_backend(cfg)   # selects mlx|torch (falls back if unavailable)
    print(f"  Level: {level} ({cfg.get('name','custom')}) | "
          f"backend: {active_backend} | config: {cfg_path} | "
          f"precision: {get_precision(cfg)}")

    # Is this stage active at this level? (entry_level ≤ level and present)
    skey = f"stage{args.stage}"
    cur = cfg.get("curriculum", {}) or {}
    if skey not in cur:
        print(f"ERROR: Stage {args.stage} is not part of level {level}.")
        active = sorted(int(k.replace('stage','')) for k in cur)
        print(f"  Active stages at level {level}: {active}")
        sys.exit(1)
    entry = int(cur[skey].get("entry_level", 1))
    if entry > (level if isinstance(level, int) else 99):
        print(f"ERROR: Stage {args.stage} enters at level {entry}; you are at level {level}.")
        print(f"  Train it at level {entry} or higher.")
        sys.exit(1)

    # Resource guard + announce (avoid OOM mid-run; report what is being learned).
    from src import resources as R
    R.announce(cfg, mode="train", stage=args.stage)
    R.guard(cfg, mode="train", force=args.force)

    # Prerequisite check (previous active stage must be complete)
    prev_n = prev_active_stage(args.stage, cfg)
    if prev_n is not None:
        prev = ckpt_root(cfg) / f"stage{prev_n}" / "stage_complete.json"
        if not prev.exists():
            print(f"ERROR: Stage {prev_n} must complete before Stage {args.stage}.")
            print(f"  Run: python train_stage.py --level {level} --stage {prev_n}")
            sys.exit(1)
        print(f"  Stage {prev_n} prereq OK")

    passed = train_stage(args.stage, cfg, resume=args.resume, plain=args.plain)

    skip_gate = cfg.get("skip_gate", False)
    lvl_flag = f" --level {level}" if isinstance(level, int) else f" --config {cfg_path}"
    # Suggest the next active stage (curriculum may be non-contiguous, e.g. a
    # level that skips causal/ethics still trains reasoning + the behavioral stages).
    active = sorted(int(k.replace("stage", "")) for k in (cfg.get("curriculum", {}) or {}))
    later = [s for s in active if s > args.stage]
    if passed:
        if skip_gate:
            tag = "smoke test — pipeline verified" if level == 0 else "no graduation gate at this level"
            print(f"\nStage {args.stage} complete ({tag}).")
            if later:
                print(f"Next stage: python train_stage.py{lvl_flag} --stage {later[0]}")
            print(f"Or chat now: python uses/chat/run_chat.py{lvl_flag} --stage {active[-1]}")
        elif later:
            print(f"\nNext: python train_stage.py{lvl_flag} --stage {later[0]}")
        else:
            print("\nAll stages complete. Foundational core frozen.")
            print(f"Next: python consolidation_daemon.py{lvl_flag} --once")
    else:
        print(f"\nStage {args.stage} gate not passed.")
        print(f"  Options: extend corpus, adjust thresholds, or --resume")
        print(f"  See: docs/GUIDE.md")


if __name__ == "__main__":
    main()
