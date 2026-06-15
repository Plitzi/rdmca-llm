#!/usr/bin/env python3
from __future__ import annotations

import os

# Auto-bootstrap: re-run with .venv/bin/python if dependencies are not available.
import sys

try:
    import numpy  # noqa: F401 — just checking the venv is active
except ModuleNotFoundError:
    _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _venv = os.path.join(_repo, ".venv", "bin", "python")
    if os.path.exists(_venv) and os.path.abspath(sys.executable) != os.path.abspath(_venv):
        os.execv(_venv, [_venv, *sys.argv])

OVERVIEW = """\
rdmca — one entry point for the whole pipeline.

Instead of remembering eight scripts, two run apps and a daemon, you run:

    rdmca <command> [args…]

Every command forwards its args to the underlying tool (so `rdmca train --help`
shows train's REAL options — there's a single source of truth per command). The
extra `rdmca info` command is model-aware: it lists the models, levels and stages
that exist and what's already prepared/trained, so you can see at a glance what to
run next.

Run `rdmca` (no args) for the grouped command list, or `rdmca <command> --help`
for a command's own options.

Select the MODEL with `--model NAME` (default `cognition`): for build/train/eval
commands it is forwarded to the tool; for a `Run` command it picks which model's use
case to launch (models/<model>/uses/<app>/). `rdmca info` lists the models.

Typical flow (model defaults to `cognition`):
    rdmca info                          # what models/levels/stages exist + status
    rdmca prepare  --level 1 --stage 1  # build that stage's corpus
    rdmca tokenizer --level 1           # train the tokenizer
    rdmca train    --level 1 --stage 1  # train the stage
    rdmca chat     --level 1 --stage 1  # talk to it
    rdmca camera --model hands_recognition   # a different model's use case
"""

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


# Each command → (group, kind, target, help). Kinds:
#   "script" — scripts/<target>; model-aware ones get `--model NAME` forwarded.
#   "run"    — a model use case at models/<model>/uses/<target>/run_<target>.py
#              (the MODEL chooses which use case exists; `--model` selects it).
#   "module" — `python -m <target>`.
#   "builtin"— handled in-process here (info).
# Args are forwarded verbatim, so each tool's own argparse is the single source of truth.
COMMANDS: dict[str, tuple[str, str, str, str]] = {
    # Data
    "prepare": ("Data", "script", "scripts/prepare_data.py", "Build a stage's training corpus"),
    "tokenizer": ("Data", "script", "scripts/train_tokenizer.py", "Train the tokenizer (+ VQ-VAE)"),
    "prepare-mm": ("Data", "script", "scripts/prepare_multimodal.py", "Build image/audio pairs"),
    # Train
    "train": ("Train", "script", "scripts/train.py", "Train a stage (gated curriculum)"),
    # Evaluate
    "bench": ("Evaluate", "script", "scripts/run_benchmarks.py", "Run external benchmarks"),
    "ood": ("Evaluate", "script", "scripts/ood_probe.py", "Out-of-distribution probe"),
    "plot": ("Evaluate", "script", "scripts/plot_metrics.py", "Plot training metrics"),
    # Run a model's use case (resolved under the selected model's uses/)
    "chat": ("Run", "run", "chat", "Interactive chat (cognition)"),
    "agent": ("Run", "run", "agent", "Agent tool-use loop (cognition)"),
    "camera": ("Run", "run", "camera", "Live camera inference (hands_recognition)"),
    "daemon": ("Run", "module", "src.consolidation.daemon", "Consolidation daemon (cognition)"),
    # Maintenance
    "purge": ("Maintenance", "script", "scripts/purge.py", "Delete generated artifacts"),
    # Discovery (built-in, handled here)
    "info": ("Discovery", "builtin", "info", "List models, levels, stages + status"),
}

GROUP_ORDER = ["Data", "Train", "Evaluate", "Run", "Maintenance", "Discovery"]

# `script` commands that accept a `--model` flag (forwarded). Others (prepare-mm) don't.
_MODEL_AWARE_SCRIPTS = {"prepare", "tokenizer", "train", "bench", "ood", "plot", "purge"}


def _print_overview() -> None:
    print(OVERVIEW.strip())
    print("\nCommands:")
    width = max(len(name) for name in COMMANDS)
    for group in GROUP_ORDER:
        print(f"\n  {group}")
        for name, (grp, _kind, _target, help_text) in COMMANDS.items():
            if grp == group:
                print(f"    {name:<{width}}  {help_text}")


def _extract_model(rest: list[str]) -> tuple[str | None, list[str]]:
    """Pull a `--model NAME` / `--model=NAME` out of the forwarded args (so the CLI can
    both route by model AND re-add it for the scripts that want it). Returns (model, rest)."""
    out: list[str] = []
    model = None
    i = 0
    while i < len(rest):
        tok = rest[i]
        if tok == "--model" and i + 1 < len(rest):
            model = rest[i + 1]
            i += 2
            continue
        if tok.startswith("--model="):
            model = tok.split("=", 1)[1]
            i += 1
            continue
        out.append(tok)
        i += 1
    return model, out


def _dispatch(name: str, rest: list[str]) -> int:
    _group, kind, target, _help = COMMANDS[name]
    model, rest = _extract_model(rest)
    if kind == "run":
        # A use case belongs to a model: models/<model>/uses/<app>/run_<app>.py
        model = model or "cognition"
        app_path = REPO / "models" / model / "uses" / target / f"run_{target}.py"
        if not app_path.exists():
            print(f"rdmca: model '{model}' has no '{name}' use case ({app_path} not found).")
            return 2
        cmd = [sys.executable, str(app_path), *rest]
    elif kind == "module":
        cmd = [sys.executable, "-m", target, *rest]
    else:  # script
        fwd = ["--model", model] if (model and name in _MODEL_AWARE_SCRIPTS) else []
        cmd = [sys.executable, str(REPO / target), *rest, *fwd]
    return subprocess.call(cmd, cwd=str(REPO))


# ── info: model-aware discovery ────────────────────────────────────────────────
def _available_models() -> list[str]:
    """Model packages under models/ (a dir with __init__.py that is not the SDK
    or the plugin system itself)."""
    models = []
    for child in sorted((REPO / "models").iterdir()):
        if not child.is_dir() or child.name in {"sdk", "__pycache__"}:
            continue
        if (child / "__init__.py").exists():
            models.append(child.name)
    return models


def _stage_prepared(number: int, level: int) -> bool:
    # Single source of truth for the corpus location is the registry (active model set
    # by the caller), so this tracks any data-layout change automatically.
    from src.plugins import stage_data_dir

    data_dir = REPO / stage_data_dir(number, {"level": level})
    return data_dir.is_dir() and any(data_dir.glob("*.jsonl"))


def _stage_trained(model: str, level: int, number: int) -> bool:
    stage_dir = REPO / "dist" / "checkpoints" / model / f"level{level}" / f"stage{number}"
    return any((stage_dir / f).exists() for f in ("final.npz", "best.npz"))


def _cmd_info(rest: list[str]) -> int:
    import argparse

    from src.config import available_levels
    from src.plugins import all_stages, bcf_stage, set_active_model

    ap = argparse.ArgumentParser(prog="rdmca info", description="List models, levels, stages.")
    ap.add_argument("--model", default=None, help="Model to inspect (default: cognition)")
    ap.add_argument(
        "--level", type=int, default=None, help="Add prepared/trained status for a level"
    )
    args = ap.parse_args(rest)

    models = _available_models()
    levels = available_levels()
    print(f"Models  ({len(models)}): " + ", ".join(models))
    print(f"Levels  ({len(levels)}): " + ", ".join(map(str, levels)))

    model = args.model or "cognition"
    if model not in models:
        print(f"\nUnknown model '{model}'. Available: {', '.join(models)}")
        return 2
    set_active_model(model)
    print(f"\nModel '{model}' — stages:")
    try:
        stages = all_stages()
    except ModuleNotFoundError:
        stages = []
    if not stages:
        print("  (no stages defined yet — this model is a TODO stub)")
        return 0
    freeze = bcf_stage()

    header = f"  {'#':>2}  {'name':<38}  {'kind':<10}  {'gate':<18}"
    if args.level is not None:
        header += f"  {f'data(L{args.level})':<9}  trained"
    print(header)
    for p in stages:
        kind = "cognitive" if p.frozen_base else "behavioral"
        gate = p.gate.metric_key if p.gate else "—"
        freeze_mark = "  ⟵ freeze point" if p.number == freeze else ""
        row = f"  {p.number:>2}  {p.name:<38}  {kind:<10}  {gate:<18}"
        if args.level is not None:
            prepared = "✓" if _stage_prepared(p.number, args.level) else "·"
            trained = "✓" if _stage_trained(model, args.level, p.number) else "·"
            row += f"  {prepared:<9}  {trained}"
        print(row + freeze_mark)
    if args.level is None:
        print("\n  Tip: `rdmca info --level 1` adds a prepared/trained column per stage.")
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help", "help"}:
        _print_overview()
        return 0
    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        print(f"rdmca: unknown command '{command}'.\n")
        _print_overview()
        return 2
    if command == "info":
        return _cmd_info(rest)
    return _dispatch(command, rest)


if __name__ == "__main__":
    sys.path.insert(0, str(REPO))
    raise SystemExit(main())
