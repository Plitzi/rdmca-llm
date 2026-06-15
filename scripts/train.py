#!/usr/bin/env python3
import os
import sys

try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    venv_py = os.path.join(_repo, ".venv", "bin", "python")
    if os.path.exists(venv_py) and os.path.abspath(sys.executable) != os.path.abspath(venv_py):
        os.execv(venv_py, [venv_py, *sys.argv])
    print("ERROR: dependencies not found. Run: source .venv/bin/activate")
    sys.exit(1)

"""
RDMCA Progressive Stage Trainer — CLI.

Usage (via the unified CLI):
  rdmca train --level 1 --stage 1            # start a stage fresh
  rdmca train --level 1 --stage 1 --resume   # resume after a pause
  rdmca train --level 1 --stage 2            # next stage (prereq enforced)

Each stage must pass its graduation gate before the next can begin. The foundational
core is frozen permanently after the last ACTIVE cognitive stage (ethics/BCF);
behavioral stages then train LoRA sectors on the frozen core. The trainer itself lives
in src/training/ (trainer/gates/checkpoint/dataload/heads/curriculum).
"""
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on path

from src.config import SUPPORTED_PRECISIONS, get_precision, require_backend
from src.training.curriculum import ckpt_root, load_config, prev_active_stage, stage_enabled
from src.training.trainer import train_stage


def _run_hint(model_name: str) -> str:
    """Model-agnostic 'now run it' line. Use cases belong to the model (each declares its
    own under uses/), so point at `rdmca uses` for THIS model instead of assuming `chat`."""
    from src.plugins import available_models, model_uses

    apps = list(model_uses(model_name))
    if not apps:
        return f"Run it:  rdmca uses --model {model_name}"
    # Suggest launching the first use case the model actually declares (no hardcoded
    # `chat`). Only add --model if another model shares the app name — an app owned by a
    # single model (e.g. `camera`) is inferred, so --model would be noise.
    app = apps[0]
    shared = sum(app in model_uses(m) for m in available_models()) > 1
    suffix = f" --model {model_name}" if shared else ""
    return f"Run it:  rdmca uses {app}{suffix}   (all: rdmca uses --model {model_name})"


def main():
    parser = argparse.ArgumentParser(
        description="RDMCA Progressive Stage Trainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  rdmca train --level 1 --stage 1
  rdmca train --level 1 --stage 1 --resume
  rdmca train --level 2 --stage 2
        """,
    )
    parser.add_argument(
        "--stage",
        type=int,
        required=True,
        help="Curriculum stage number (validated against the level's config)",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=None,
        help="Educational level 1-5 (preescolar..universidad). Determines model size, data and resources.",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Explicit config path (overrides --level)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model to train (a package under models/, e.g. cognition). "
        "Overrides the config's model_name; defaults to cognition.",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from latest checkpoint in stage dir"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even if the resource guard says it won't fit (risk OOM)",
    )
    parser.add_argument(
        "--precision",
        choices=SUPPORTED_PRECISIONS,
        default=None,
        help="Override training precision (fp32|bf16|fp16). Lower precision uses less "
        "memory, so a bigger level may fit on the same hardware.",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Plain scrolling logs instead of the live dashboard (selectable/copyable, "
        "no flicker). A full train.log is written to the stage's checkpoint dir either "
        "way (also via RDMCA_PLAIN_LOGS=1).",
    )
    parser.add_argument(
        "--skip-gate",
        dest="skip_gate",
        action="store_true",
        help="Manually disable the graduation gate for this run (the gate is ENFORCED "
        "from level 1 by default — quality first). Lets a stage complete at its token "
        "budget without meeting the perplexity bar.",
    )
    args = parser.parse_args()

    from src.config import resolve_config_path

    cfg_path = resolve_config_path(args.config, args.level)
    cfg = load_config(cfg_path)
    # Select the active model (CLI --model wins; registry default = cognition) before
    # anything touches the stage registry, so it discovers THIS model's stage plugins.
    from src.config import select_model

    model_name = select_model(cfg, args.model)
    # Precision override (CLI wins over config). Set before the guard/announce so the
    # precision-aware memory estimate reflects the chosen dtype.
    if args.precision:
        cfg.setdefault("training", {})["precision"] = args.precision
    # Manual gate override (CLI wins): force-disable the graduation gate for this run.
    if args.skip_gate:
        cfg["skip_gate"] = True
        print("  [gate] manually disabled for this run (--skip-gate)")
    level = cfg.get("level", "?")
    active_backend = require_backend(cfg)  # selects mlx|torch (falls back if unavailable)
    print(
        f"  Level: {level} ({cfg.get('name', 'custom')}) | "
        f"backend: {active_backend} | config: {cfg_path} | "
        f"precision: {get_precision(cfg)}"
    )

    # Is this stage active at this level? (entry_level ≤ level and present)
    stage_key = f"stage{args.stage}"
    curriculum = cfg.get("curriculum", {}) or {}
    if stage_key not in curriculum:
        print(f"ERROR: Stage {args.stage} is not part of level {level}.")
        active = sorted(int(k.replace("stage", "")) for k in curriculum)
        print(f"  Active stages at level {level}: {active}")
        sys.exit(1)
    entry = int(curriculum[stage_key].get("entry_level", 1))
    if entry > (level if isinstance(level, int) else 99):
        print(f"ERROR: Stage {args.stage} enters at level {entry}; you are at level {level}.")
        print(f"  Train it at level {entry} or higher.")
        sys.exit(1)
    if not stage_enabled(args.stage, cfg):
        print(
            f"ERROR: Stage {args.stage} is disabled (plugin `enabled=False` or "
            f"curriculum.{stage_key}.enabled: false). Re-enable it to train."
        )
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
            print(f"  Run: rdmca train --level {level} --stage {prev_n}")
            sys.exit(1)
        print(f"  Stage {prev_n} prereq OK")

    passed = train_stage(args.stage, cfg, resume=args.resume, plain=args.plain)

    skip_gate = cfg.get("skip_gate", False)
    lvl_flag = f" --level {level}" if isinstance(level, int) else f" --config {cfg_path}"
    # Suggest the next active stage (curriculum may be non-contiguous).
    active = sorted(int(k.replace("stage", "")) for k in (cfg.get("curriculum", {}) or {}))
    later = [s for s in active if s > args.stage]
    if passed:
        if skip_gate:
            tag = (
                "smoke test — pipeline verified"
                if level == 0
                else "no graduation gate at this level"
            )
            print(f"\nStage {args.stage} complete ({tag}).")
            if later:
                print(f"Next stage: rdmca train{lvl_flag} --stage {later[0]}")
            print(f"Or run it now: {_run_hint(model_name)}")
        elif later:
            print(f"\nNext: rdmca train{lvl_flag} --stage {later[0]}")
        else:
            print("\nAll stages complete. Foundational core frozen.")
            print(_run_hint(model_name))
    else:
        print(f"\nStage {args.stage} gate not passed.")
        print("  Options: extend corpus, adjust thresholds, or --resume")
        print(f"  See: models/{model_name}/docs/GUIDE.md")


if __name__ == "__main__":
    main()
