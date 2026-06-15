"""Plugin SDK — the stable, single-import contract a stage plugin is written against.

A stage (`src/models/<model>/stageNN_<slug>/`) imports EVERYTHING it needs from here and
NOTHING else from the framework, so it stays a pure, droppable unit the framework
merely *consumes* (discovered by the registry). Deleting a stage can never break the
framework, and adding one only requires this SDK.

Surface:
  • contract types — StagePlugin, StageGate, StageKind, SourceBuilder
  • data-stream tooling — blend, interleave, cycle_records
  • text utilities — stable_hash, passes_filter, flesch_kincaid_grade
  • conversational shaping — persona_for, prepend_system, hash01, STORY_PROMPTS
  • tool/agent transcript helpers — hermes_events, hermes_to_transcript, AGENTIC_SYSTEM_PROMPT
  • tokenizer specials a plugin may need — REASONING_SPECIALS

The SDK itself bridges to the framework core; plugins never reach past it. Anything
MODEL-specific (e.g. cognition's moods/emotions) is NOT here — it lives with the model
and is imported intra-model (see src/models/cognition/mood).
"""

from __future__ import annotations

from src.core.modalities.vocab import REASONING_SPECIALS
from src.models.base import SourceBuilder, StageGate, StageKind, StagePlugin
from src.models.sdk.agentic import (
    AGENTIC_SYSTEM_PROMPT,
    hermes_events,
    hermes_to_transcript,
    hermes_tools,
)
from src.models.sdk.persona import (
    STORY_PROMPTS,
    SYSTEM_PERSONAS,
    hash01,
    persona_for,
    prepend_system,
)
from src.models.sdk.streams import blend, cycle_records, interleave
from src.models.sdk.textfilter import flesch_kincaid_grade, passes_filter, stable_hash

__all__ = [
    "AGENTIC_SYSTEM_PROMPT",
    "REASONING_SPECIALS",
    "STORY_PROMPTS",
    "SYSTEM_PERSONAS",
    "SourceBuilder",
    "StageGate",
    "StageKind",
    "StagePlugin",
    "blend",
    "cycle_records",
    "flesch_kincaid_grade",
    "hash01",
    "hermes_events",
    "hermes_to_transcript",
    "hermes_tools",
    "interleave",
    "passes_filter",
    "persona_for",
    "prepend_system",
    "stable_hash",
]
