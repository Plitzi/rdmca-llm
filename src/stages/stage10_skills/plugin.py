"""Stage 10 — Skills. Behavioral: LoRA sector for Claude-style SKILL.md skills."""

from __future__ import annotations

from src.stages.base import StageKind, StagePlugin
from src.stages.stage10_skills.sources import SOURCES

PLUGIN = StagePlugin(
    number=10,
    slug="skills",
    name="Skills",
    entry_level=0,
    kind=StageKind.BEHAVIORAL,
    sources=SOURCES,
)
