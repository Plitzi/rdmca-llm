"""Stage 7 — Cognitive ethics and BCF. The freeze point: the core's values are
trained here, then the whole cognitive base is frozen. The mood head is retrained
at this final frozen state (what the shipped model and behavioral stages use)."""

from __future__ import annotations

from src.stages.base import StageGate, StagePlugin
from src.stages.stage07_ethics.sources import SOURCES

PLUGIN = StagePlugin(
    number=7,
    slug="ethics",
    name="Cognitive ethics and BCF",
    entry_level=1,
    frozen_base=True,
    rehearsal_fraction=0.35,
    lr_scale=0.7,
    gate=StageGate("bcf_accuracy", 0.90, "Cognitive ethics — BCF probe"),
    trains_mood=True,
    is_freeze_point=True,
    sources=SOURCES,
)
