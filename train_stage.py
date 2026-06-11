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
# Stage gates: metric name, threshold, human description
# ---------------------------------------------------------------------------
# Natural order: cognition (1-5) → values (6, freeze) → behavioral interfaces (7-9).
STAGE_GATES = {
    1: ("blim_accuracy",      0.70, "Language — BLiMP grammaticality"),
    2: ("arc_easy_accuracy",  0.60, "Patterns — ARC easy"),
    3: ("gsm8k_accuracy",     0.15, "Abstraction — GSM8K"),
    4: ("causal_accuracy",    0.65, "Causal and procedural reasoning"),
    5: ("reasoning_accuracy", 0.20, "Reasoning — chain-of-thought (GSM8K)"),
    6: ("bcf_accuracy",       0.90, "Cognitive ethics — BCF probe"),
}

STAGE_NAMES = {
    1: "Language and communication",
    2: "Perception and pattern recognition",
    3: "Abstraction and symbolic composition",
    4: "Causal and procedural reasoning",
    5: "Reasoning",
    6: "Cognitive ethics and BCF",
    7: "Action and tool use",
    8: "Model Context Protocol (MCP)",
    9: "Skills",
}

# BCF_STAGE is the ethics stage and the LATEST possible freeze point. The actual
# freeze happens after the last ACTIVE cognitive stage (`last_cognitive_stage`),
# which is BCF_STAGE when ethics is present and an earlier stage otherwise (e.g.
# reasoning (5) at level 1). Behavioral stages (>6) then train as LoRA sectors on
# the frozen core. Change BCF_STAGE here if the curriculum order changes.
BCF_STAGE = 6


def last_cognitive_stage(cfg: dict) -> int | None:
    """Highest ACTIVE stage that is part of the frozen cognitive base (≤ BCF_STAGE).
    The core is frozen right after this stage: at L1 (no ethics) that's reasoning
    (5), at L4+ it's ethics/BCF (6). Below that, behavioral stages add sectors."""
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
    with open(path) as f:
        return yaml.safe_load(f)


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
                    tokens_seen: int, loss: float, ckpt_dir: Path):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fname = ckpt_dir / f"step_{step:08d}.npz"
    backend.current().engine.save_weights(model, str(fname))
    state = {
        "step": step, "stage": stage,
        "tokens_seen": tokens_seen, "loss": round(loss, 6),
        "timestamp": time.time(), "checkpoint": str(fname),
    }
    with open(ckpt_dir / "latest.json", "w") as f:
        json.dump(state, f, indent=2)
    print(f"  [ckpt] step={step:,} | {tokens_seen/1e6:.1f}M tokens | "
          f"loss={loss:.4f} -> {fname.name}")


def load_checkpoint(model, ckpt_dir: Path):
    latest = ckpt_dir / "latest.json"
    if not latest.exists():
        return 0, 0
    with open(latest) as f:
        state = json.load(f)
    backend.current().engine.load_weights(model, state["checkpoint"])
    print(f"  [resume] step={state['step']:,} | "
          f"{state['tokens_seen']/1e6:.1f}M tokens | loss={state['loss']:.4f}")
    return state["step"], state["tokens_seen"]


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
        frac = float(cfg.get("training", {}).get("rehearsal_fraction", 0.15))
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
                                        replay_dirs=replay_dirs, replay_fraction=frac)
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


def validation_perplexity(model, data_loader, n_batches: int = 8) -> float:
    """Mean validation perplexity over n held-out batches."""
    B = backend.current()
    losses = []
    for _ in range(n_batches):
        batch = B.ops.array(data_loader.next_batch())
        loss  = model.eval_ce(batch)
        B.engine.eval(loss)
        losses.append(B.engine.item(loss))
    return float(np.exp(np.mean(losses)))


# Proxy perplexity gates per stage until task-specific benchmarks
# (BLiMP / ARC / GSM8K / COPA / BCF probes) are wired in. Overridable via
# cfg["gate"]["max_perplexity"][stage].
DEFAULT_GATE_PPL = {1: 50.0, 2: 45.0, 3: 40.0, 4: 38.0, 5: 36.0, 6: 35.0}


def evaluate_gate(model, stage: int,
                  data_loader=None, cfg: dict = None) -> tuple:
    """
    Graduation gate. Operative metric is real validation perplexity (a proxy
    that actually measures the model); task-specific benchmarks (BLiMP, ARC,
    GSM8K, COPA, BCF probes) should replace the per-stage threshold as they
    are wired in. The ethics/BCF stage additionally checks BCF probe accuracy when a probe
    set is available. Returns (score, passed).
    """
    # Post-base behavioral stages (tool use / MCP / skills / reasoning) have no
    # benchmark gate — fall back to the perplexity-only proxy with a generic label.
    gate = STAGE_GATES.get(stage)
    desc = gate[2] if gate else stage_name(stage, cfg)
    gate_cfg = (cfg or {}).get("gate", {})
    max_ppl  = gate_cfg.get("max_perplexity", {}).get(stage, DEFAULT_GATE_PPL.get(stage, 40.0))

    ppl = validation_perplexity(model, data_loader)
    passed = ppl <= max_ppl
    print(f"  [gate] Stage {stage}: {desc}")
    print(f"  [gate] val perplexity={ppl:.2f} | threshold<= {max_ppl:.1f} "
          f"-> {'PASS' if passed else 'fail'}")

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
    if stage == last_cognitive_stage(cfg):
        if stage == BCF_STAGE:
            train_bcf_head(model, ckpt_dir, precision)
        freeze_model(model, root / "foundational")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train_stage(stage: int, cfg: dict, resume: bool = False) -> bool:
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
        actual_vocab = info["vocab_size"]
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

    optimizer = B.engine.make_optimizer(
        model, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])

    start_step = 0
    tokens_seen = 0
    if resume:
        start_step, tokens_seen = load_checkpoint(model, ckpt_dir)

    # Re-apply precision: loading prev-stage / resume weights restores their
    # saved dtype, so cast once more before training in the configured precision.
    set_model_precision(model, precision)

    # Real data loader
    data_loader = build_data_loader(stage, cfg)

    # Derived constants
    bs        = tcfg["batch_size"]
    grad_acc  = tcfg["grad_accumulation"]
    seq_len   = model_cfg.context_len
    toks_step = bs * seq_len * grad_acc
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

    step = start_step
    running_loss = 0.0
    log_interval  = 100   # interval for tps calculation
    dash_interval = 10    # update dashboard every N steps (smooth)
    t0 = time.time()
    t_dash = time.time()
    last_tps = 0.0

    def loss_fn(mdl, toks):
        return mdl.mrl_loss(toks)

    loss_and_grad_fn = B.engine.value_and_grad(model, loss_fn)

    dash = TrainingDashboard(stage, n_tokens_target,
                             resume_step=start_step,
                             resume_tokens=tokens_seen)

    with dash:
        dash.print(f"Stage {stage} | {model.count_params()/1e6:.1f}M params | real data")

        while tokens_seen < target:
            # Update learning rate
            lr = cosine_lr(step, tcfg["lr"], tcfg.get("lr_min", 3e-5),
                           warmup, total_steps)
            B.engine.set_lr(optimizer, lr)

            # Gradient accumulation
            acc_loss = 0.0
            grads = None
            for _ in range(grad_acc):
                batch = B.ops.array(data_loader.next_batch())
                loss, g = loss_and_grad_fn(model, batch)
                B.engine.eval(loss)
                acc_loss += B.engine.item(loss)
                grads = g

            B.engine.optimizer_step(optimizer, model, grads)

            step         += 1
            tokens_seen  += toks_step
            running_loss += acc_loss / grad_acc

            # After the first full pass, cap the effective target to the corpus
            # size × max_passes (only if that's *below* the configured target).
            if not capped and data_loader.epoch_tokens:
                capped = True
                cap = max_passes * data_loader.epoch_tokens
                if cap < n_tokens_target:
                    target = cap
                    dash.set_target(target)
                    dash.print(f"[corpus] {data_loader.epoch_tokens/1e6:.1f}M tokens/pass "
                               f"— capping at {max_passes}× ({_fmt_tokens(cap)}) to avoid "
                               f"overfitting the configured {_fmt_tokens(n_tokens_target)} target")

            # Recalculate tps every log_interval steps
            if step % log_interval == 0:
                elapsed  = time.time() - t0
                last_tps = (log_interval * toks_step) / elapsed
                running_loss = 0.0
                t0 = time.time()

            # Dashboard refresh every dash_interval steps (smooth)
            if step % dash_interval == 0:
                avg_loss = running_loss / max(step % log_interval or log_interval, 1)
                dash.update(step, tokens_seen, acc_loss / grad_acc, lr, last_tps)

            # Checkpoint
            if step % save_every == 0:
                save_checkpoint(model, step, stage, tokens_seen,
                               acc_loss / grad_acc, ckpt_dir)
                dash.set_checkpoint(step)
                dash.print(f"[ckpt] step {step:,}")

            # Gate evaluation. On skip_gate levels (no graduation gate) the score
            # is shown for information only — it must NOT end the stage early, or
            # training stops at the first eval_every (e.g. step 1000) far short of
            # the token budget. The stage then runs to its full budget.
            if step % eval_every == 0:
                score, passed = evaluate_gate(model, stage, data_loader, cfg)
                dash.set_gate_result(score, passed)
                if passed and not skip_gate:
                    save_checkpoint(model, step, stage, tokens_seen,
                                   acc_loss / grad_acc, ckpt_dir)
                    B.engine.save_weights(model, str(ckpt_dir / "final.npz"))
                    with open(ckpt_dir / "stage_complete.json", "w") as f:
                        json.dump({
                            "stage": stage, "step": step,
                            "tokens_seen": tokens_seen, "gate_score": score,
                            "timestamp": time.time(),
                        }, f, indent=2)
                    dash.print(f"[bold green]Stage {stage} COMPLETE — "
                               f"gate {score:.4f}[/bold green]")
                    _on_stage_complete(model, stage, cfg, root, ckpt_dir, precision, adapter)
                    return True

        # Final dashboard update so it shows 100%
        dash.update(step, tokens_seen, acc_loss / grad_acc, lr, last_tps)

        # Budget exhausted
        save_checkpoint(model, step, stage, tokens_seen,
                       acc_loss / grad_acc, ckpt_dir)

        if skip_gate:
            # Smoke-test run (e.g. profile=test) — graduation gate not required
            ckpt_file = str(ckpt_dir / f"step_{step:08d}.npz")
            with open(ckpt_dir / "stage_complete.json", "w") as f:
                json.dump({"stage": stage, "step": step,
                           "tokens_seen": tokens_seen, "gate_score": None,
                           "checkpoint": ckpt_file,
                           "skip_gate": True, "timestamp": time.time()}, f, indent=2)
            dash.print(f"[bold green]Stage {stage} COMPLETE (gate skipped)[/bold green]")
            _on_stage_complete(model, stage, cfg, root, ckpt_dir, precision, adapter)
            return True

        score, passed = evaluate_gate(model, stage, data_loader, cfg)
        dash.set_gate_result(score, passed)
        if passed:
            dash.print(f"[bold green]Stage {stage} COMPLETE — gate {score:.4f}[/bold green]")
            _on_stage_complete(model, stage, cfg, root, ckpt_dir, precision, adapter)
        else:
            need = STAGE_GATES[stage][1] if stage in STAGE_GATES else None
            need_txt = f"(need {need:.2f}) " if need is not None else ""
            dash.print(f"Budget exhausted. Gate: {score:.4f} "
                       f"{need_txt}— run --resume to continue")
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

    passed = train_stage(args.stage, cfg, resume=args.resume)

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
