"""Stage 4 — Causal and procedural reasoning. Real e-CARE seed + synthetic
cause→effect fill."""

from __future__ import annotations

from models.cognition.stages.stage04_causal.sources import SOURCES
from src.plugins.sdk import StageGate, StagePlugin

PLUGIN = StagePlugin(
    number=4,
    slug="causal",
    name="Causal and procedural reasoning",
    entry_level=1,
    frozen_base=True,
    rehearsal_fraction=0.35,
    lr_scale=0.7,
    gate=StageGate("causal_accuracy", 0.65, "Causal and procedural reasoning"),
    sources=SOURCES,
)
