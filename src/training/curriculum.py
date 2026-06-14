"""Curriculum helpers — stage ordering, naming, the freeze point and the LR schedule.

These read a level's `curriculum` config and the stage registry; they are the leaf
utilities the trainer and its helper modules build on. Stage metadata (name, freeze
point, behavioral kind) comes from `src.stages`, the single source of truth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.stages import bcf_stage, get_stage, has_stage

# Latest possible freeze point (= the ethics/BCF stage), from the registry.
BCF_STAGE = bcf_stage()


def last_cognitive_stage(cfg: dict) -> int | None:
    """Highest ACTIVE stage that is part of the frozen cognitive base (≤ BCF_STAGE).
    The core is frozen right after this stage; behavioral stages then add LoRA sectors."""
    active = [int(k.replace("stage", "")) for k in (cfg.get("curriculum") or {})]
    base = [s for s in active if s <= BCF_STAGE]
    return max(base) if base else None


def is_behavioral_stage(stage: int) -> bool:
    """Behavioral stages (tool/MCP/skills, > BCF_STAGE) train as LoRA sectors on the
    frozen core so they never overwrite language/reasoning."""
    return stage > BCF_STAGE


def stage_name(stage: int, cfg: dict | None = None) -> str:
    """Stage label — prefers the config's per-stage `name`, then the registry, then a
    generic fallback. Keeps new stages working with no code change."""
    if cfg:
        stage_cfg = cfg.get("curriculum", {}).get(f"stage{stage}")
        if stage_cfg and stage_cfg.get("name"):
            return stage_cfg["name"]
    return get_stage(stage).name if has_stage(stage) else f"Stage {stage}"


def stage_enabled(stage: int, cfg: dict | None = None) -> bool:
    """Whether a stage is switched ON. A stage can be disabled two ways, both honored
    here (the same pattern for every stage): its plugin's `enabled` flag, or a per-level
    `curriculum.stageN.enabled: false` override in the config."""
    if has_stage(stage) and not get_stage(stage).enabled:
        return False
    if cfg:
        stage_cfg = cfg.get("curriculum", {}).get(f"stage{stage}")
        if stage_cfg is not None and not stage_cfg.get("enabled", True):
            return False
    return True


def prev_active_stage(stage: int, cfg: dict) -> int | None:
    """Highest curriculum stage below `stage` declared in this config (the real
    predecessor — stages can be non-contiguous, e.g. {1,2,3,6}), or None."""
    below = [
        int(k.replace("stage", ""))
        for k in cfg.get("curriculum", {})
        if int(k.replace("stage", "")) < stage
    ]
    return max(below) if below else None


def ckpt_root(cfg: dict) -> Path:
    """Checkpoint root, namespaced by level so levels never collide."""
    level = cfg.get("level")  # NB: level 0 is valid → use `is None`
    return Path("dist/checkpoints") if level is None else Path("dist/checkpoints") / f"level{level}"


def cosine_lr(step: int, base_lr: float, min_lr: float, warmup: int, total: int) -> float:
    """Linear warmup then cosine decay from `base_lr` to `min_lr` over `total` steps."""
    if step < warmup:
        return base_lr * step / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))


def load_config(path: str) -> dict:
    # Single implementation (deep-merges configs/levels/_base.yaml so levels declare
    # only their diffs) lives in src.config — delegate so the two never diverge.
    from src.config import load_config as _load_config

    return _load_config(path)
