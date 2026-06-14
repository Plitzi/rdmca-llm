"""
DEPRECATED compatibility shim — the single source of truth for stage metadata is
now the per-stage plugins under `src/plugins/` (discovered by `src.plugins.registry`).

This module rebuilds the old constants (`STAGE_GATES`, `STAGE_NAMES`,
`STAGE_REHEARSAL`, `STAGE_LR_SCALE`, `MOOD_TRAIN_STAGES`, `BCF_STAGE`) FROM the
registry so existing importers keep working unchanged. New code should import from
`src.plugins` directly (e.g. `from src.plugins import get_stage, bcf_stage`).
"""

from __future__ import annotations

from src.plugins import all_stages, bcf_stage, mood_stages

# Latest possible freeze point (= the BCF/ethics stage).
BCF_STAGE = bcf_stage()

# stage -> (metric_key, threshold, label); only stages that declare a gate.
STAGE_GATES = {
    p.number: (p.gate.metric_key, p.gate.threshold, p.gate.label)
    for p in all_stages()
    if p.gate is not None
}

# stage -> human name (every stage).
STAGE_NAMES = {p.number: p.name for p in all_stages()}

# Per-stage anti-forgetting profile. Kept as dicts holding only the stages that
# DIFFER from the defaults, matching the old hand-written tables (so callers that
# do `.get(stage, DEFAULT)` see identical values).
DEFAULT_REHEARSAL = 0.15
DEFAULT_LR_SCALE = 1.0
STAGE_REHEARSAL = {
    p.number: p.rehearsal_fraction
    for p in all_stages()
    if p.rehearsal_fraction != DEFAULT_REHEARSAL
}
STAGE_LR_SCALE = {p.number: p.lr_scale for p in all_stages() if p.lr_scale != DEFAULT_LR_SCALE}

# Stages whose completion (re)trains the mood head.
MOOD_TRAIN_STAGES = mood_stages()


def is_mood_stage(stage: int) -> bool:
    """Whether the mood head should be (re)trained at this stage's completion."""
    return stage in MOOD_TRAIN_STAGES
