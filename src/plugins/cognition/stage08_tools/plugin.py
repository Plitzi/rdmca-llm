"""Stage 8 — Action and tool use. Behavioral: trains a LoRA sector on the frozen
core (real tool-use loop, Claude-style JSON), so it never overwrites the base."""

from __future__ import annotations

from src.plugins.cognition.stage08_tools.sources import SOURCES
from src.plugins.sdk import StagePlugin

PLUGIN = StagePlugin(
    number=8,
    slug="tools",
    name="Action and tool use",
    entry_level=0,
    frozen_base=False,
    sources=SOURCES,
)
