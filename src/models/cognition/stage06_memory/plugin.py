"""Stage 6 — Memory management. A COGNITIVE faculty: the frozen core itself learns
to consume recalled memory (<mem> recall + episodic), so it sits inside the base,
right before ethics/BCF."""

from __future__ import annotations

from src.models.cognition.stage06_memory.sources import SOURCES
from src.models.sdk import StageGate, StagePlugin

PLUGIN = StagePlugin(
    number=6,
    slug="memory",
    name="Memory management",
    entry_level=1,
    frozen_base=True,
    rehearsal_fraction=0.35,
    lr_scale=0.7,
    gate=StageGate("memory_accuracy", 0.50, "Memory — recall and use of injected memory"),
    sources=SOURCES,
)
