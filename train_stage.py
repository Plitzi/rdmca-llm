#!/usr/bin/env python3
import sys, os
try:
    import mlx.core  # noqa: F401
except ModuleNotFoundError:
    venv_py = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           ".venv", "bin", "python")
    if os.path.exists(venv_py) and os.path.abspath(sys.executable) != os.path.abspath(venv_py):
        os.execv(venv_py, [venv_py] + sys.argv)
    print("ERROR: mlx not found. Run: source .venv/bin/activate")
    sys.exit(1)

"""
RDMCA Progressive Stage Trainer with Checkpoint-Resume
=======================================================
Usage:
  # Start Stage 1 fresh
  python train_stage.py --stage 1 --config configs/rdmca_t2.yaml

  # Resume Stage 1 after a pause
  python train_stage.py --stage 1 --config configs/rdmca_t2.yaml --resume

  # After Stage 1 gate passes, start Stage 2
  python train_stage.py --stage 2 --config configs/rdmca_t2.yaml

Each stage must pass its graduation gate before the next can begin.
Foundational core is frozen permanently after Stage 5.
"""
import os
import sys
import json
import time
import argparse
import yaml
import numpy as np
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).parent))
from src.model.transformer import RDMCAFoundational, ModelConfig
from src.modalities.text import TextTokenizer
from src.data.loader import DataLoader, TextDataset
from src.training.dashboard import TrainingDashboard


# ---------------------------------------------------------------------------
# Stage gates: metric name, threshold, human description
# ---------------------------------------------------------------------------
STAGE_GATES = {
    1: ("blim_accuracy",     0.70, "Language — BLiMP grammaticality"),
    2: ("arc_easy_accuracy", 0.60, "Patterns — ARC easy"),
    3: ("gsm8k_accuracy",    0.15, "Abstraction — GSM8K"),
    4: ("causal_accuracy",   0.65, "Causal reasoning"),
    5: ("bcf_accuracy",      0.90, "Cognitive ethics — BCF probe"),
}

STAGE_NAMES = {
    1: "Language and communication",
    2: "Perception and pattern recognition",
    3: "Abstraction and symbolic composition",
    4: "Causal and procedural reasoning",
    5: "Cognitive ethics and BCF",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


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
    mx.savez(str(fname), **dict(tree_flatten(model.parameters())))
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
    weights = mx.load(state["checkpoint"])
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    print(f"  [resume] step={state['step']:,} | "
          f"{state['tokens_seen']/1e6:.1f}M tokens | loss={state['loss']:.4f}")
    return state["step"], state["tokens_seen"]


def dummy_batch(vocab_size: int, seq_len: int, batch_size: int) -> mx.array:
    """Fallback when tokenizer or data is not yet available."""
    return mx.array(np.random.randint(1, vocab_size, (batch_size, seq_len + 1)))


def build_data_loader(stage: int, cfg: dict):
    """
    Try to build a real DataLoader from stage data.
    Falls back to dummy_batch if data or tokenizer is missing.
    """
    tokenizer = TextTokenizer()
    if not tokenizer.ready:
        print("  [data] Tokenizer not found — using dummy batches.")
        print("         Run: python scripts/train_tokenizer.py")
        return None

    stage_dirs = {
        1: "data/stage1_language",
        2: "data/stage2_patterns",
        3: "data/stage3_abstraction",
        4: "data/stage4_causal",
        5: "data/stage5_ethics",
    }
    data_dir = stage_dirs[stage]
    try:
        loader = DataLoader.from_config(stage, cfg, tokenizer)
        print(f"  [data] Real data loader: {data_dir}")
        return loader
    except FileNotFoundError as e:
        print(f"  [data] {e}")
        print(f"         Run: python scripts/prepare_data.py --stage {stage}")
        return None


def evaluate_gate(model: RDMCAFoundational, stage: int) -> tuple:
    """
    Evaluate graduation gate for this stage.
    Returns (score: float, passed: bool).

    REPLACE each branch with the real benchmark evaluation:
      Stage 1: BLiMP grammaticality (https://github.com/alexwarstadt/blimp)
      Stage 2: ARC Easy (https://allenai.org/data/arc)
      Stage 3: GSM8K (https://github.com/openai/grade-school-math)
      Stage 4: COPA / causal reasoning benchmark
      Stage 5: BCF probe set (custom, defined in tests/bcf_probes.jsonl)
    """
    metric, threshold, desc = STAGE_GATES[stage]
    print(f"  [gate] Stage {stage}: {desc}")
    print(f"  [gate] Metric: {metric} | Threshold: {threshold:.2f}")
    print(f"  [gate] TODO: plug in real benchmark here")
    # Placeholder — always returns 0 until you implement real eval
    score = 0.0
    return score, score >= threshold


def freeze_model(model: RDMCAFoundational, ckpt_dir: Path):
    """Permanently freeze all foundational parameters after Stage 5."""
    print("\n" + "=" * 60)
    print("  FREEZING FOUNDATIONAL CORE — Theta_F locked forever")
    n = model.count_params()
    print(f"  {n/1e6:.1f}M parameters frozen")
    mx.savez(str(ckpt_dir / "theta_f_frozen.npz"), **dict(tree_flatten(model.parameters())))
    with open(ckpt_dir / "frozen.json", "w") as f:
        json.dump({"frozen": True, "params": n,
                   "timestamp": time.time()}, f, indent=2)
    print(f"  Saved: {ckpt_dir}/theta_f_frozen.npz")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train_stage(stage: int, cfg: dict, resume: bool = False) -> bool:
    tcfg      = cfg["training"]
    mcfg      = cfg["model"]
    skip_gate = cfg.get("skip_gate", False)   # toy config sets this to true
    stages    = list(cfg["curriculum"].values())
    n_tokens_target = stages[stage - 1]["n_tokens"]
    ckpt_dir = Path(f"dist/checkpoints/stage{stage}")

    def _fmt_tokens(n: int) -> str:
        if n >= 1_000_000_000:
            return f"{n/1e9:.2f}B"
        if n >= 1_000_000:
            return f"{n/1e6:.0f}M"
        return f"{n/1e3:.0f}K"

    print(f"\n{'='*60}")
    print(f"  Stage {stage}: {STAGE_NAMES[stage]}")
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

    model_cfg = ModelConfig(**{k: v for k, v in mcfg.items()
                               if k in ModelConfig.__dataclass_fields__})
    model = RDMCAFoundational(model_cfg)
    print(f"  Model: {model.count_params()/1e6:.1f}M params | "
          f"d_model={model_cfg.d_model} | layers={model_cfg.n_layers} | "
          f"vocab={model_cfg.vocab_size}")

    # Load previous stage weights as starting point (stages 2-5)
    if stage > 1:
        prev_ckpt = Path(f"dist/checkpoints/stage{stage-1}/latest.json")
        with open(prev_ckpt) as f:
            prev_state = json.load(f)
        weights = mx.load(prev_state["checkpoint"])
        model.load_weights(list(weights.items()))
        mx.eval(model.parameters())
        print(f"  Loaded Stage {stage-1} weights as starting point")

    optimizer = optim.AdamW(
        learning_rate=tcfg["lr"],
        weight_decay=tcfg["weight_decay"],
    )

    start_step = 0
    tokens_seen = 0
    if resume:
        start_step, tokens_seen = load_checkpoint(model, ckpt_dir)

    # Real data loader (falls back to dummy if data not ready)
    data_loader = build_data_loader(stage, cfg)

    # Derived constants
    bs        = tcfg["batch_size"]
    grad_acc  = tcfg["grad_accumulation"]
    seq_len   = model_cfg.context_len
    toks_step = bs * seq_len * grad_acc
    warmup    = tcfg["warmup_steps"]
    total_steps = n_tokens_target // toks_step
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

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    dash = TrainingDashboard(stage, n_tokens_target,
                             resume_step=start_step,
                             resume_tokens=tokens_seen)

    with dash:
        dash.print(f"Stage {stage} | {model.count_params()/1e6:.1f}M params | "
                   f"{'real data' if data_loader else 'dummy batches'}")

        while tokens_seen < n_tokens_target:
            # Update learning rate
            lr = cosine_lr(step, tcfg["lr"], tcfg.get("lr_min", 3e-5),
                           warmup, total_steps)
            optimizer.learning_rate = lr

            # Gradient accumulation
            acc_loss = 0.0
            grads = None
            for _ in range(grad_acc):
                if data_loader is not None:
                    batch = data_loader.next_batch()
                else:
                    batch = dummy_batch(model_cfg.vocab_size, seq_len, bs)
                loss, g = loss_and_grad_fn(model, batch)
                mx.eval(loss)
                acc_loss += loss.item()
                grads = g

            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            step         += 1
            tokens_seen  += toks_step
            running_loss += acc_loss / grad_acc

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

            # Gate evaluation
            if step % eval_every == 0:
                score, passed = evaluate_gate(model, stage)
                dash.set_gate_result(score, passed)
                if passed:
                    save_checkpoint(model, step, stage, tokens_seen,
                                   acc_loss / grad_acc, ckpt_dir)
                    mx.savez(str(ckpt_dir / "final.npz"),
                             **dict(tree_flatten(model.parameters())))
                    with open(ckpt_dir / "stage_complete.json", "w") as f:
                        json.dump({
                            "stage": stage, "step": step,
                            "tokens_seen": tokens_seen, "gate_score": score,
                            "timestamp": time.time(),
                        }, f, indent=2)
                    dash.print(f"[bold green]Stage {stage} COMPLETE — "
                               f"gate {score:.4f}[/bold green]")
                    if stage == 5:
                        freeze_model(model, Path("dist/checkpoints/foundational"))
                    return True

        # Final dashboard update so it shows 100%
        dash.update(step, tokens_seen, acc_loss / grad_acc, lr, last_tps)

        # Budget exhausted
        save_checkpoint(model, step, stage, tokens_seen,
                       acc_loss / grad_acc, ckpt_dir)

        if skip_gate:
            # Toy / smoke-test run — gate not required
            ckpt_file = str(ckpt_dir / f"step_{step:08d}.npz")
            with open(ckpt_dir / "stage_complete.json", "w") as f:
                json.dump({"stage": stage, "step": step,
                           "tokens_seen": tokens_seen, "gate_score": None,
                           "checkpoint": ckpt_file,
                           "skip_gate": True, "timestamp": time.time()}, f, indent=2)
            dash.print(f"[bold green]Stage {stage} COMPLETE (toy run — gate skipped)[/bold green]")
            return True

        score, passed = evaluate_gate(model, stage)
        dash.set_gate_result(score, passed)
        if passed:
            dash.print(f"[bold green]Stage {stage} COMPLETE — gate {score:.4f}[/bold green]")
        else:
            dash.print(f"Budget exhausted. Gate: {score:.4f} "
                       f"(need {STAGE_GATES[stage][1]:.2f}) — run --resume to continue")
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
  python train_stage.py --stage 1 --config configs/rdmca_t2.yaml
  python train_stage.py --stage 1 --config configs/rdmca_t2.yaml --resume
  python train_stage.py --stage 2 --config configs/rdmca_t2.yaml
        """
    )
    parser.add_argument("--stage",  type=int, required=True,
                        choices=[1, 2, 3, 4, 5], help="Curriculum stage (1-5)")
    parser.add_argument("--config", type=str, default="configs/rdmca_t2.yaml")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint in stage dir")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Prerequisite check
    if args.stage > 1:
        prev = Path(f"dist/checkpoints/stage{args.stage-1}/stage_complete.json")
        if not prev.exists():
            print(f"ERROR: Stage {args.stage-1} must complete before Stage {args.stage}.")
            print(f"  Run: python train_stage.py --stage {args.stage-1} --config {args.config}")
            sys.exit(1)
        print(f"  Stage {args.stage-1} prereq OK")

    passed = train_stage(args.stage, cfg, resume=args.resume)

    skip_gate = cfg.get("skip_gate", False)
    if passed:
        if skip_gate:
            print(f"\nToy Stage {args.stage} complete. Pipeline verified.")
            print(f"Next: python chat.py --stage {args.stage}")
        elif args.stage < 5:
            nxt = args.stage + 1
            print(f"\nNext: python train_stage.py --stage {nxt} --config {args.config}")
        else:
            print("\nAll stages complete. Foundational core frozen.")
            print("Next: run consolidation_daemon.py to begin daily learning.")
    else:
        print(f"\nStage {args.stage} gate not passed.")
        print(f"  Options: extend corpus, adjust thresholds, or --resume")
        print(f"  See: docs/guides/training.md")


if __name__ == "__main__":
    main()
