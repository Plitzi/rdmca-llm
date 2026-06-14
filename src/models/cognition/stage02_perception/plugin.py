"""Stage 2 — Perception and pattern recognition. Narrow pattern data; rehearsed and
LR-scaled down so it nudges, not overwrites, the conversational core."""

from __future__ import annotations

from src.models.cognition.stage02_perception.sources import SOURCES
from src.models.sdk import StageGate, StagePlugin

PLUGIN = StagePlugin(
    number=2,
    slug="perception",
    name="Perception and pattern recognition",
    entry_level=1,
    frozen_base=True,
    rehearsal_fraction=0.35,
    lr_scale=0.7,
    gate=StageGate("arc_easy_accuracy", 0.60, "Patterns — ARC easy"),
    sources=SOURCES,
)
