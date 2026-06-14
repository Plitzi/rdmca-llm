"""
RDMCA curriculum stages, as self-contained plugins.

Each stage lives in its own `stageNN_<slug>` package carrying its metadata and its
data sources; the registry discovers them automatically. Import the registry API
from here:

    from src.stages import active_stages, get_stage, bcf_stage, stream_source
"""

from __future__ import annotations

from src.stages.base import SourceBuilder, StageGate, StageKind, StagePlugin
from src.stages.registry import (
    active_stages,
    all_stages,
    bcf_stage,
    enabled_stages,
    get_stage,
    has_stage,
    is_behavioral,
    mood_stages,
    owns_source,
    stage_data_dir,
    stream_source,
)

__all__ = [
    "SourceBuilder",
    "StageGate",
    "StageKind",
    "StagePlugin",
    "active_stages",
    "all_stages",
    "bcf_stage",
    "enabled_stages",
    "get_stage",
    "has_stage",
    "is_behavioral",
    "mood_stages",
    "owns_source",
    "stage_data_dir",
    "stream_source",
]
