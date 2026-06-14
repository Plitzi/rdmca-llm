"""Build the training DataLoader for a stage, including the anti-forgetting rehearsal
mix (a fraction of earlier base stages' data) for cognitive stages."""

from __future__ import annotations

import sys
from pathlib import Path

from src.stages import get_stage, stage_data_dir
from src.training.curriculum import is_behavioral_stage

# A stage with this rehearsal fraction has "no specific profile" — it then defers to
# the level's training.rehearsal_fraction (matching the old STAGE_REHEARSAL.get default).
DEFAULT_REHEARSAL = 0.15


def build_data_loader(stage: int, cfg: dict):
    """Build a real DataLoader from stage data. Exits with actionable instructions if
    the tokenizer or the stage corpus is missing (no random-batch fallback)."""
    from src.data.loader import DataLoader
    from src.modalities.text import TextTokenizer

    tokenizer = TextTokenizer()
    if not tokenizer.ready:
        print("ERROR: tokenizer not found at dist/tokenizer/rdmca_spm.model")
        print("  Run: python scripts/train_tokenizer.py --level <N>")
        sys.exit(1)
    # Rehearsal: cognitive stages after the first mix in a fraction of earlier base
    # stages' data, so learning a new faculty does not erode earlier ones (esp.
    # conversation) before the core is frozen. Behavioral stages need none.
    replay_dirs: list[str] = []
    frac = 0.0
    if not is_behavioral_stage(stage):
        stage_cfg = cfg.get("curriculum", {}).get(f"stage{stage}", {}) or {}
        # Per-stage anti-forgetting default lives on the stage plugin (applies at EVERY
        # level). A stage left at the plain default defers to the level's training
        # default; the level's yaml may still override per stage.
        profiled = get_stage(stage).rehearsal_fraction
        rehearsal_default = (
            profiled
            if profiled != DEFAULT_REHEARSAL
            else cfg.get("training", {}).get("rehearsal_fraction", DEFAULT_REHEARSAL)
        )
        frac = float(stage_cfg.get("rehearsal_fraction", rehearsal_default))
        if frac > 0:
            curriculum = cfg.get("curriculum", {})
            earlier = sorted(
                s
                for s in (int(k.replace("stage", "")) for k in curriculum)
                if s < stage and not is_behavioral_stage(s)  # only replay frozen-base stages
            )
            for earlier_stage in earlier:
                data_dir = stage_data_dir(earlier_stage, cfg)
                if Path(data_dir).exists():
                    replay_dirs.append(data_dir)
    try:
        loader = DataLoader.from_config(
            stage, cfg, tokenizer, replay_dirs=replay_dirs, replay_fraction=frac, with_mask=True
        )  # completion-only loss masking
        loader.replay_dirs = replay_dirs  # expose for the per-stage audit record
        loader.replay_fraction = frac
        data_dir = cfg["curriculum"][f"stage{stage}"].get("data_dir")  # key-based (non-contiguous)
        print(f"  [data] Real data loader: {data_dir}")
        if replay_dirs:
            print(
                f"  [rehearsal] mixing {frac:.0%} replay from {len(replay_dirs)} earlier "
                f"stage(s) to retain prior skills (e.g. conversation)"
            )
        return loader
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print(
            f"  Run: python scripts/prepare_data.py --level {cfg.get('level', '')} "
            f"--stage {stage}".rstrip()
        )
        sys.exit(1)
