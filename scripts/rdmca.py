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

Typical flow (model defaults to `cognition`):
    rdmca info                         # what models/levels/stages exist + status
    rdmca prepare  --level 1 --stage 1 # build that stage's corpus
    rdmca tokenizer --level 1          # train the tokenizer
    rdmca train    --level 1 --stage 1 # train the stage
    rdmca chat     --level 1 --stage 1 # talk to it
"""

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


# Each command → (group, target, one-line help). A target is ("script", relpath)
# run as a file, or ("module", dotted) run with `python -m`. Args are forwarded
# verbatim, so each tool's own argparse stays the single source of truth.
COMMANDS: dict[str, tuple[str, tuple[str, str], str]] = {
    # Data
    "prepare": ("Data", ("script", "scripts/prepare_data.py"), "Build a stage's training corpus"),
    "tokenizer": (
        "Data",
        ("script", "scripts/train_tokenizer.py"),
        "Train the tokenizer (+ VQ-VAE)",
    ),
    "prepare-mm": ("Data", ("script", "scripts/prepare_multimodal.py"), "Build image/audio pairs"),
    # Train
    "train": ("Train", ("script", "scripts/train.py"), "Train a stage (gated curriculum)"),
    # Evaluate
    "bench": ("Evaluate", ("script", "scripts/run_benchmarks.py"), "Run external benchmarks"),
    "ood": ("Evaluate", ("script", "scripts/ood_probe.py"), "Out-of-distribution probe"),
    "plot": ("Evaluate", ("script", "scripts/plot_metrics.py"), "Plot training metrics"),
    # Run (consume the trained model)
    "chat": (
        "Run",
        ("script", "models/cognition/uses/chat/run_chat.py"),
        "Interactive chat with a checkpoint",
    ),
    "agent": ("Run", ("script", "models/cognition/uses/agent/run_agent.py"), "Agent tool-use loop"),
    "daemon": (
        "Run",
        ("module", "src.core.consolidation.daemon"),
        "Consolidation daemon (learn from experience)",
    ),
    # Maintenance
    "purge": ("Maintenance", ("script", "scripts/purge.py"), "Delete generated artifacts"),
    # Discovery (built-in, handled here)
    "info": ("Discovery", ("builtin", "info"), "List models, levels, stages + status"),
}

GROUP_ORDER = ["Data", "Train", "Evaluate", "Run", "Maintenance", "Discovery"]


def _print_overview() -> None:
    print(OVERVIEW.strip())
    print("\nCommands:")
    width = max(len(name) for name in COMMANDS)
    for group in GROUP_ORDER:
        print(f"\n  {group}")
        for name, (grp, _target, help_text) in COMMANDS.items():
            if grp == group:
                print(f"    {name:<{width}}  {help_text}")


def _dispatch(name: str, rest: list[str]) -> int:
    _group, (kind, target), _help = COMMANDS[name]
    if kind == "script":
        cmd = [sys.executable, str(REPO / target), *rest]
    else:  # module
        cmd = [sys.executable, "-m", target, *rest]
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


def _stage_prepared(model: str, stage_pkg: str, level: int) -> bool:
    data_dir = REPO / "models" / model / stage_pkg / "data" / f"level{level}"
    return data_dir.is_dir() and any(data_dir.glob("*.jsonl"))


def _stage_trained(model: str, level: int, number: int) -> bool:
    stage_dir = REPO / "dist" / "checkpoints" / model / f"level{level}" / f"stage{number}"
    return any((stage_dir / f).exists() for f in ("final.npz", "best.npz"))


def _cmd_info(rest: list[str]) -> int:
    import argparse

    from src.core.config import available_levels
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
            prepared = "✓" if _stage_prepared(model, p.package, args.level) else "·"
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
