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

Select the MODEL with `--model NAME` (default `cognition`): build/train/eval commands
forward it to the tool. USE CASES are NOT global commands — each model declares its own
under models/<model>/uses/<app>/, discovered at runtime (cognition has chat/agent,
hands_recognition has camera). There is NO `rdmca chat`: you always go through `uses` —
`rdmca uses [--model NAME]` lists them, `rdmca uses <app> [--model NAME]` launches one.
`rdmca info` lists the models.

Typical flow (model defaults to `cognition`):
    rdmca info                          # what models/levels/stages exist + status
    rdmca prepare  --level 1 --stage 1  # build that stage's corpus
    rdmca tokenizer --level 1           # train the tokenizer
    rdmca train    --level 1 --stage 1  # train the stage
    rdmca uses                          # this model's use cases (how to run it)
    rdmca uses chat --level 1 --stage 1            # a cognition use case
    rdmca uses camera --model hands_recognition    # a hands_recognition use case
"""

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


# Each FRAMEWORK command → (group, kind, target, help). Kinds:
#   "script" — scripts/<target>; model-aware ones get `--model NAME` forwarded.
#   "module" — `python -m <target>`.
#   "builtin"— handled in-process here (info, uses).
# Args are forwarded verbatim, so each tool's own argparse is the single source of truth.
# NOTE: use cases (chat, agent, camera, …) are NOT listed here — they belong to a model,
# not the framework, and are discovered at runtime from models/<model>/uses/ (see
# `_model_uses`). A bare `rdmca <app>` that isn't a framework command is resolved there.
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
    # Runtime
    "daemon": ("Runtime", "module", "src.consolidation.daemon", "Consolidation daemon"),
    # Maintenance
    "purge": ("Maintenance", "script", "scripts/purge.py", "Delete generated artifacts"),
    # Discovery (built-in, handled here)
    "info": ("Discovery", "builtin", "info", "List models, levels, stages + status"),
    "uses": ("Discovery", "builtin", "uses", "List a model's use cases (how to run it)"),
}

GROUP_ORDER = ["Data", "Train", "Evaluate", "Runtime", "Maintenance", "Discovery"]

# `script` commands that accept a `--model` flag (forwarded). Others (prepare-mm) don't.
_MODEL_AWARE_SCRIPTS = {"prepare", "tokenizer", "train", "bench", "ood", "plot", "purge"}


def _print_overview() -> None:
    print(OVERVIEW.strip())
    print("\nCommands:")
    # The fixed framework commands fit a stable column; the discovered use cases print as
    # `uses <app>` (the only launch form), so size the column to whichever is longest.
    all_uses = _all_model_uses()
    width = max(len(name) for name in (*COMMANDS, *(f"uses {a}" for a in all_uses)))
    for group in GROUP_ORDER:
        print(f"\n  {group}")
        for name, (grp, _kind, _target, help_text) in COMMANDS.items():
            if grp == group:
                print(f"    {name:<{width}}  {help_text}")
    # Use cases are per-model and discovered at runtime — list every model's apps so
    # `rdmca --help` answers "what can I run?" without hardcoding chat/agent/camera. They
    # are launched via `rdmca uses <app>`, never as bare commands.
    if all_uses:
        print("\n  Use cases (per model — launch with `rdmca uses <app>`)")
        for name, (model, summary) in all_uses.items():
            launch = f"uses {name}"
            print(f"    {launch:<{width}}  {summary}  ({model})")


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
    """Run a FRAMEWORK command (script or module). Use cases are NOT here — they are
    launched through `rdmca uses <app>` (see `_launch_use`), never as a bare command."""
    _group, kind, target, _help = COMMANDS[name]
    model, rest = _extract_model(rest)
    if kind == "module":
        cmd = [sys.executable, "-m", target, *rest]
    else:  # script
        fwd = ["--model", model] if (model and name in _MODEL_AWARE_SCRIPTS) else []
        cmd = [sys.executable, str(REPO / target), *rest, *fwd]
    return subprocess.call(cmd, cwd=str(REPO))


def _launch_use(model: str | None, app: str, app_args: list[str]) -> int:
    """Launch a use case via `rdmca uses <app>`: models/<model>/uses/<app>/run_<app>.py.
    When no `--model` is given the model is INFERRED from the app — an app owned by a
    single model (e.g. `camera` → hands_recognition) needs no `--model`. If SEVERAL models
    declare the same app name it is ambiguous and `--model` is REQUIRED (nothing is assumed,
    not even cognition). Use cases belong to the model, so there is no global command."""
    owners = [m for m, apps in _all_uses_by_model().items() if app in apps]
    if model is None:
        if len(owners) == 1:
            model = owners[0]  # unambiguous → infer it, no --model needed
        elif len(owners) > 1:
            print(f"rdmca uses: '{app}' is declared by several models: {', '.join(owners)}")
            print(f"  Ambiguous — pick one:  rdmca uses {app} --model {owners[0]}")
            return 2
    if model and app in _model_uses(model):
        return subprocess.call(
            [sys.executable, str(_model_uses(model)[app]), *app_args], cwd=str(REPO)
        )
    # Not found in the chosen/default model — point at whoever owns it (if anyone).
    print(f"rdmca uses: model '{model or 'cognition'}' has no '{app}' use case.")
    if owners:
        print(f"  '{app}' belongs to: {', '.join(owners)}")
        print(f"  Try:  rdmca uses {app} --model {owners[0]}")
    else:
        print(f"  Run `rdmca uses --model {model or 'cognition'}` to see its use cases.")
    return 2


# ── use-case discovery (per model) — delegated to the framework's single source ──
def _model_uses(model: str) -> dict[str, Path]:
    from src.plugins import model_uses

    return model_uses(model)


def _use_summary(path: Path) -> str:
    """First non-empty line of the run module's docstring (a short label for listings)."""
    import ast

    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return ""
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            text = node.value.value
            if isinstance(text, str):
                for line in text.strip().splitlines():
                    if line.strip():
                        return line.strip()
                break
    return ""


def _all_uses_by_model() -> dict[str, dict[str, Path]]:
    """{model: {app: run_path}} across every model — for cross-model hints + the overview."""
    return {m: _model_uses(m) for m in _available_models()}


def _all_model_uses() -> dict[str, tuple[str, str]]:
    """Flat {app: (model, summary)} across all models, for the `--help` use-case section.
    If two models share an app name the first (alphabetical) model wins the summary line;
    the per-model `rdmca uses` view always shows each model's own."""
    flat: dict[str, tuple[str, str]] = {}
    for model, apps in _all_uses_by_model().items():
        for app, path in apps.items():
            flat.setdefault(app, (model, _use_summary(path)))
    return flat


# ── info: model-aware discovery ────────────────────────────────────────────────
def _available_models() -> list[str]:
    from src.plugins import available_models

    return available_models()


def _stage_prepared(number: int, level: int) -> bool:
    # Single source of truth for the corpus location is the registry (active model set
    # by the caller), so this tracks any data-layout change automatically.
    from src.plugins import stage_data_dir

    data_dir = REPO / stage_data_dir(number, {"level": level})
    return data_dir.is_dir() and any(data_dir.glob("*.jsonl"))


def _stage_trained(model: str, level: int, number: int) -> bool:
    stage_dir = REPO / "dist" / model / "checkpoints" / f"level{level}" / f"stage{number}"
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
    model = args.model or "cognition"
    if model not in models:
        print(f"Models  ({len(models)}): " + ", ".join(models))
        print(f"\nUnknown model '{model}'. Available: {', '.join(models)}")
        return 2
    set_active_model(model)
    # Levels are PER-MODEL — list this model's ladder (a model may use a single --config
    # instead of numbered levels, e.g. hands_recognition → none).
    levels = available_levels(model)
    print(f"Models  ({len(models)}): " + ", ".join(models))
    print(f"Levels of '{model}'  ({len(levels)}): " + (", ".join(map(str, levels)) or "—"))
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


_USES_HELP = """\
rdmca uses — discover and launch a model's use cases.

  rdmca uses                       list every model's use cases
  rdmca uses --model M             list a single model's use cases
  rdmca uses <app> [--model M] …   LAUNCH a use case (args after <app> are forwarded;
                                   add --help for the use case's own options)

Use cases belong to the model (cognition: chat, agent · hands_recognition: camera),
so there is no global `rdmca chat` — you always go through `uses`. The model is INFERRED
from the app, so an app only one model has (e.g. `camera`) needs no --model; --model is
only required to pick among models that share an app name."""


def _cmd_uses(rest: list[str]) -> int:
    """Discover + launch use cases. No positional → list; `uses <app> …` → launch it.
    This is the ONLY way to run a use case (no bare `rdmca chat`), so the entry point
    is model-aware and a model with no apps simply has nothing to launch."""
    model, rest = _extract_model(rest)
    # Launch form: the first remaining token is an app name (not a flag). Pass the model
    # as given (None ⇒ `_launch_use` infers it from the app).
    if rest and not rest[0].startswith("-"):
        return _launch_use(model, rest[0], rest[1:])
    # Listing form (optionally `-h`): no app to launch.
    if rest and rest[0] in {"-h", "--help"}:
        print(_USES_HELP)
        return 0

    models = _available_models()
    if model and model not in models:
        print(f"Unknown model '{model}'. Available: {', '.join(models)}")
        return 2
    selected = [model] if model else models
    # An app only one model owns can launch without --model, so only show --model in the
    # listing when the name is shared (ambiguous) across models.
    owners = _all_uses_by_model()
    all_apps = {app for apps in owners.values() for app in apps}
    shared = {app for app in all_apps if sum(app in apps for apps in owners.values()) > 1}
    # Pre-compute every launch string so the summaries line up in one column.
    rows: list[tuple[str, str, str]] = []  # (model, launch, summary)
    for name in selected:
        for app, path in _model_uses(name).items():
            suffix = f" --model {name}" if app in shared else ""
            rows.append((name, f"rdmca uses {app}{suffix}", _use_summary(path)))
    width = max((len(launch) for _m, launch, _s in rows), default=0)
    for name in selected:
        model_rows = [r for r in rows if r[0] == name]
        print(f"\n{name} — {len(model_rows)} use case(s):")
        if not model_rows:
            print("  (none — this model ships no runnable uses/ app yet)")
        for _m, launch, summary in model_rows:
            print(f"  {launch:<{width}}  {summary}")
    if rows:
        print("\nLaunch one with `rdmca uses <app> [--model M]` (add --help for its options).")
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help", "help"}:
        _print_overview()
        return 0
    command, rest = argv[0], argv[1:]
    if command == "info":
        return _cmd_info(rest)
    if command == "uses":
        return _cmd_uses(rest)
    if command in COMMANDS:
        return _dispatch(command, rest)
    return _unknown_command(command)


def _unknown_command(command: str) -> int:
    """Unknown top-level word. If it's actually a use case (e.g. `chat`, `camera`), teach
    the `rdmca uses <app>` form instead of pretending it's a global command."""
    elsewhere = [m for m, apps in _all_uses_by_model().items() if command in apps]
    print(f"rdmca: unknown command '{command}'.")
    if elsewhere:
        print(f"  '{command}' is a use case of {', '.join(elsewhere)} — launch it via `uses`:")
        print(f"    rdmca uses {command} --model {elsewhere[0]}")
    else:
        print("  Run `rdmca` for commands, or `rdmca uses` for model use cases.")
    return 2


if __name__ == "__main__":
    sys.path.insert(0, str(REPO))
    raise SystemExit(main())
