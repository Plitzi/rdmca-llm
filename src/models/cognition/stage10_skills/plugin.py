"""Stage 10 — Skills. Behavioral: LoRA sector for Claude-style SKILL.md skills."""

from __future__ import annotations

from src.models.cognition.stage10_skills.sources import SOURCES
from src.models.sdk import StagePlugin

PLUGIN = StagePlugin(
    number=10,
    slug="skills",
    name="Skills",
    entry_level=0,
    frozen_base=False,
    sources=SOURCES,
)
