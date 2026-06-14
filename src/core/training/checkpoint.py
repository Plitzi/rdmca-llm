"""Checkpoint I/O, the per-stage audit record, and the foundational-core freeze."""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from pathlib import Path

import src.core.backend as backend
from src.core.training.curriculum import stage_name


def save_checkpoint(
    model,
    step: int,
    stage: int,
    tokens_seen: int,
    loss: float,
    ckpt_dir: Path,
    optimizer=None,
    log=print,
):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fname = ckpt_dir / f"step_{step:08d}.npz"
    backend.current().engine.save_weights(model, str(fname))
    if optimizer is not None:  # warm-resume: save AdamW moments too, per-checkpoint
        # (step_NNNN.opt) so a rollback to any step restores its optimizer moments,
        # not just the latest. Costs ~2× the weights per ckpt.
        backend.current().engine.save_optimizer(optimizer, str(fname.with_suffix(".opt")))
    state = {
        "step": step,
        "stage": stage,
        "tokens_seen": tokens_seen,
        "loss": round(loss, 6),
        "timestamp": time.time(),
        "checkpoint": str(fname),
    }
    tmp = ckpt_dir / "latest.json.tmp"  # atomic pointer update
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, ckpt_dir / "latest.json")
    log(f"[ckpt] step={step:,} | {tokens_seen / 1e6:.1f}M tokens | loss={loss:.4f} -> {fname.name}")


def write_stage_complete(ckpt_dir: Path, **fields) -> None:
    """Write the stage's OUTCOME record (stage_complete.json). `timestamp` is added
    automatically; pass the rest (stage, step, tokens_seen, gate_score, …) as kwargs."""
    with open(ckpt_dir / "stage_complete.json", "w") as f:
        json.dump({**fields, "timestamp": time.time()}, f, indent=2)


def _git_commit() -> str:
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[2]),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def write_stage_audit(
    ckpt_dir: Path,
    *,
    stage: int,
    cfg: dict,
    model,
    model_cfg,
    tcfg: dict,
    data_loader,
    target: int,
    total_steps: int,
    precision: str,
    seed,
    hparams_extra: dict,
) -> dict:
    """Persist the COMPLETE training CONTEXT for this stage to `audit.json`, so a run
    is fully auditable/reproducible after the fact: exact hyperparameters, data
    provenance (per-source tokens + 'exhausted' flags), the rehearsal mix and its
    size-weights, model geometry, and env/git. The run TIMELINE (loss curve, gates,
    early-stop, COMPLETE) lives in train.log; the OUTCOME in stage_complete.json."""
    from datetime import datetime

    from src.plugins import stage_data_dir

    data_dir = Path(stage_data_dir(stage, cfg))
    sources = []
    for meta in sorted(data_dir.glob("*.meta.json")):
        with contextlib.suppress(OSError, json.JSONDecodeError):
            sources.append(
                {"source": meta.name[: -len(".meta.json")], **json.loads(meta.read_text())}
            )
    replay_weights = getattr(data_loader, "_replay_weights", None)
    rec = {
        "stage": stage,
        "level": cfg.get("level"),
        "stage_name": stage_name(stage, cfg),
        "started": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "backend": cfg.get("backend"),
        "precision": precision,
        "seed": seed,
        "command": "python " + " ".join(sys.argv),
        "model": {
            "params": int(model.count_params()),
            "d_model": model_cfg.d_model,
            "n_layers": model_cfg.n_layers,
            "n_heads": model_cfg.n_heads,
            "vocab_size": model_cfg.vocab_size,
            "context_len": model_cfg.context_len,
            "mrl_dims": list(model_cfg.mrl_dims),
        },
        "hparams": {
            "lr": tcfg.get("lr"),
            "lr_min": tcfg.get("lr_min"),
            "batch_size": tcfg.get("batch_size"),
            "grad_accumulation": tcfg.get("grad_accumulation"),
            "warmup_steps": tcfg.get("warmup_steps"),
            "max_corpus_passes": tcfg.get("max_corpus_passes"),
            "clip_grad_norm": tcfg.get("clip_grad_norm"),
            "save_every": tcfg.get("save_every"),
            "eval_every": tcfg.get("eval_every"),
            **hparams_extra,
        },
        "data": {
            "dir": str(data_dir),
            "target_tokens": int(target),
            "lr_horizon_steps": int(total_steps),
            "sources": sources,
        },
        "rehearsal": {
            "fraction": getattr(data_loader, "replay_fraction", 0.0),
            "dirs": getattr(data_loader, "replay_dirs", []),
            "weights_pct": (
                [round(100 * w / sum(replay_weights), 1) for w in replay_weights]
                if replay_weights
                else []
            ),
        },
    }
    with contextlib.suppress(OSError):
        (ckpt_dir / "audit.json").write_text(json.dumps(rec, indent=2))
    return rec


def load_checkpoint(model, ckpt_dir: Path, optimizer=None):
    latest = ckpt_dir / "latest.json"
    if not latest.exists():
        return 0, 0
    with open(latest) as f:
        state = json.load(f)
    backend.current().engine.load_weights(model, state["checkpoint"])
    warm = False
    if optimizer is not None:  # restore AdamW moments if saved (the step's own .opt)
        opt_path = Path(state["checkpoint"]).with_suffix(".opt")
        warm = backend.current().engine.load_optimizer(optimizer, str(opt_path))
    print(
        f"  [resume] step={state['step']:,} | "
        f"{state['tokens_seen'] / 1e6:.1f}M tokens | loss={state['loss']:.4f}"
        f" | optimizer={'warm' if warm else 'cold (no .opt — will spike briefly)'}"
    )
    return state["step"], state["tokens_seen"]


def freeze_model(model, ckpt_dir: Path):
    """Permanently freeze the foundational core after the ethics/BCF stage."""
    print("\n" + "=" * 60)
    print("  FREEZING FOUNDATIONAL CORE — Theta_F locked forever")
    backend.current().engine.freeze_all(model)  # excludes params from trainable set
    n = model.count_params()
    print(f"  {n / 1e6:.1f}M parameters frozen")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    backend.current().engine.save_weights(model, str(ckpt_dir / "theta_f_frozen.npz"))
    with open(ckpt_dir / "frozen.json", "w") as f:
        json.dump({"frozen": True, "params": n, "timestamp": time.time()}, f, indent=2)
    print(f"  Saved: {ckpt_dir}/theta_f_frozen.npz")
    print("=" * 60 + "\n")
