"""Stage 2 — Perception and pattern recognition. Narrow pattern data; rehearsed and
LR-scaled down so it nudges, not overwrites, the conversational core."""

from __future__ import annotations

from src.stages.base import StageGate, StageKind, StagePlugin
from src.stages.stage02_perception.sources import SOURCES

PLUGIN = StagePlugin(
    number=2,
    slug="perception",
    name="Perception and pattern recognition",
    entry_level=1,
    kind=StageKind.COGNITIVE,
    rehearsal_fraction=0.35,
    lr_scale=0.7,
    gate=StageGate("arc_easy_accuracy", 0.60, "Patterns — ARC easy"),
    sources=SOURCES,
)
