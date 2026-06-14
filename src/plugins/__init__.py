"""
RDMCA plugins — training scenarios as self-contained domains.

A **domain** is one scenario under `src/plugins/<domain>/` (e.g. `cognition`, the
conversational/agentic LLM; `hands_recognition`, a VR hand-pose model — TODO). Each
domain holds its own `stageNN_<slug>` plugins, discovered automatically by the
registry for the ACTIVE domain (`set_domain`, driven by `cfg["domain"]`).

The plugin SYSTEM (this package) — `base`, `registry`, `sdk` — is shared by every
domain and is NOT itself a plugin. Import the registry API from here:

    from src.plugins import active_stages, get_stage, bcf_stage, stream_source
"""

from __future__ import annotations

from src.plugins.base import DomainSpec, SourceBuilder, StageGate, StageKind, StagePlugin
from src.plugins.registry import (
    active_domain,
    active_stages,
    all_stages,
    bcf_stage,
    enabled_stages,
    get_stage,
    has_stage,
    is_behavioral,
    mood_stages,
    owns_source,
    set_domain,
    stage_data_dir,
    stream_source,
)

__all__ = [
    "DomainSpec",
    "SourceBuilder",
    "StageGate",
    "StageKind",
    "StagePlugin",
    "active_domain",
    "active_stages",
    "all_stages",
    "bcf_stage",
    "enabled_stages",
    "get_stage",
    "has_stage",
    "is_behavioral",
    "mood_stages",
    "owns_source",
    "set_domain",
    "stage_data_dir",
    "stream_source",
]
