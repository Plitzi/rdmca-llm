"""Stage 8 — Action and tool use. Behavioral: trains a LoRA sector on the frozen
core (real tool-use loop, Claude-style JSON), so it never overwrites the base."""

from __future__ import annotations

from src.stages.base import StageKind, StagePlugin
from src.stages.stage08_tools.sources import SOURCES

PLUGIN = StagePlugin(
    number=8,
    slug="tools",
    name="Action and tool use",
    entry_level=0,
    kind=StageKind.BEHAVIORAL,
    sources=SOURCES,
)
