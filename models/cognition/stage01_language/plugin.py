"""Stage 1 — Language and communication. The conversational core; trained at full
LR and lightly rehearsed because everything else builds on it."""

from __future__ import annotations

from models.cognition.stage01_language.sources import SOURCES
from src.plugins.sdk import StageGate, StagePlugin

PLUGIN = StagePlugin(
    number=1,
    slug="language",
    name="Language and communication",
    entry_level=1,
    frozen_base=True,
    rehearsal_fraction=0.15,
    lr_scale=1.0,
    gate=StageGate("blim_accuracy", 0.70, "Language — BLiMP grammaticality"),
    trains_mood=True,
    sources=SOURCES,
)
