"""Checkpoint I/O, the per-stage audit record, and the foundational-core freeze."""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from pathlib import Path

import src.backend as backend
from src.training.curriculum import stage_name


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
        # Geometry is recorded best-effort: the audit must work for ANY model, not just
        # the text transformer — a non-text model (e.g. hand-pose) has no heads/vocab/mrl.
        "model": {
            "params": int(model.count_params()),
            "d_model": getattr(model_cfg, "d_model", None),
            "n_layers": getattr(model_cfg, "n_layers", None),
            "n_heads": getattr(model_cfg, "n_heads", None),
            "vocab_size": getattr(model_cfg, "vocab_size", None),
            "context_len": getattr(model_cfg, "context_len", None),
            "mrl_dims": list(getattr(model_cfg, "mrl_dims", []) or []),
            # Vision-model geometry (None for the text LM): lets a consumer rebuild the
            # EXACT net from the checkpoint — e.g. the camera reconstructs the hand FCN at
            # the trained arch/size instead of guessing (which silently shape-mismatched).
            "arch": getattr(model_cfg, "arch", None),
            "img_size": getattr(model_cfg, "img_size", None),
            "in_channels": getattr(model_cfg, "in_channels", None),
            "heatmap_size": getattr(model_cfg, "heatmap_size", None),
            "dims": getattr(model_cfg, "dims", None),
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
        # The audit is the FIRST thing written, before any checkpoint creates the dir —
        # ensure it exists so the context isn't silently dropped (it was, on fresh runs).
        ckpt_dir.mkdir(parents=True, exist_ok=True)
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


def _read_json(path: Path):
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except (OSError, ValueError):
        return None


def resolve_stage_checkpoint(stage_dir: Path):
    """Pick the checkpoint inference should use for ONE stage dir, ALWAYS preferring the
    BEST (ratcheted, lowest val-score) over the latest training step. This is the STANDARD
    resolution for every model — the same best→final→latest layout the trainer writes, so
    any consumer (chat, agent, camera, …) resolves a trained model the same way. Returns
    (path|None, label, meta); meta is the tracked JSON so the caller can report quality:

      1. best.npz   — the running/ratcheted best (the gate's moving bar), meta=best.json;
      2. final.npz  — the graduated model (= the best at graduation);
      3. latest.json — only when no eval-best exists yet (training just started).
    """
    best_npz, final_npz = stage_dir / "best.npz", stage_dir / "final.npz"
    if best_npz.exists():
        return best_npz, "best", _read_json(stage_dir / "best.json")
    if final_npz.exists():
        return (
            final_npz,
            "final (graduated)",
            _read_json(stage_dir / "best.json") or _read_json(stage_dir / "stage_complete.json"),
        )
    state = _read_json(stage_dir / "latest.json")
    if state and state.get("checkpoint") and Path(state["checkpoint"]).exists():
        return Path(state["checkpoint"]), "latest (in-progress)", state
    return None, "none", None


def discover_checkpoint(model: str, level: int | None = None, stage: int | None = None):
    """Auto-discover the trained checkpoint to load when the caller does NOT know the exact
    stage dir — scans dist/<model>/checkpoints/level*/stage*/ and returns the best one from
    the MOST RECENTLY TRAINED stage (optionally restricted to a level/stage). Returns the
    same (path|None, label, meta) as `resolve_stage_checkpoint`. This is what a use case runs
    with no --checkpoint: "just load whatever I last trained for this model"."""
    from src.config import model_dist_root

    root = model_dist_root(model) / "checkpoints"
    if not root.is_dir():
        return None, "none", None
    stage_dirs = [
        stage_dir
        for lvl_dir in root.glob("level*")
        if level is None or lvl_dir.name == f"level{level}"
        for stage_dir in lvl_dir.glob("stage*")
        if stage is None or stage_dir.name == f"stage{stage}"
    ]
    # Newest first, so a fresh retrain wins; resolve_stage_checkpoint then picks best/final.
    for stage_dir in sorted(stage_dirs, key=lambda d: d.stat().st_mtime, reverse=True):
        path, label, meta = resolve_stage_checkpoint(stage_dir)
        if path is not None:
            return path, label, meta
    return None, "none", None


def trained_arch(checkpoint: str | Path) -> dict:
    """The architecture the checkpoint was trained at, from its sibling audit.json (the
    `model` block: d_model, n_layers, …). The net MUST be rebuilt at these dims or the
    weights are shape-mismatched and silently stay random. Empty dict if no audit."""
    audit = Path(checkpoint).parent / "audit.json"
    rec = _read_json(audit) or {}
    arch = rec.get("model")
    return arch if isinstance(arch, dict) else {}


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
