"""
RDMCA models — training scenarios as self-contained models.

A **model** is one scenario under `src/models/<name>/` (e.g. `cognition`, the
conversational/agentic LLM; `hands_recognition`, a VR hand-pose model — TODO). Each
model holds its own `stageNN_<slug>` plugins, discovered automatically by the registry
for the ACTIVE model (`set_active_model`, driven by `cfg["model_name"]`).

The plugin SYSTEM (this package) — `base`, `registry`, `sdk` — is shared by every model
and is NOT itself a stage plugin. Import the registry API from here:

    from src.models import active_stages, get_stage, bcf_stage, stream_source
"""

from __future__ import annotations

from src.models.base import ModelSpec, SourceBuilder, StageGate, StageKind, StagePlugin
from src.models.registry import (
    active_model,
    active_stages,
    all_stages,
    bcf_stage,
    enabled_stages,
    get_stage,
    has_stage,
    is_behavioral,
    mood_stages,
    owns_source,
    set_active_model,
    stage_data_dir,
    stream_source,
)

__all__ = [
    "ModelSpec",
    "SourceBuilder",
    "StageGate",
    "StageKind",
    "StagePlugin",
    "active_model",
    "active_stages",
    "all_stages",
    "bcf_stage",
    "enabled_stages",
    "get_stage",
    "has_stage",
    "is_behavioral",
    "mood_stages",
    "owns_source",
    "set_active_model",
    "stage_data_dir",
    "stream_source",
]
