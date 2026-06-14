"""Stage 4 — Causal and procedural reasoning. Real e-CARE seed + synthetic
cause→effect fill."""

from __future__ import annotations

from src.stages.base import StageGate, StageKind, StagePlugin
from src.stages.stage04_causal.sources import SOURCES

PLUGIN = StagePlugin(
    number=4,
    slug="causal",
    name="Causal and procedural reasoning",
    entry_level=1,
    kind=StageKind.COGNITIVE,
    rehearsal_fraction=0.35,
    lr_scale=0.7,
    gate=StageGate("causal_accuracy", 0.65, "Causal and procedural reasoning"),
    sources=SOURCES,
)
