"""Stage 3 — Abstraction and symbolic composition (arithmetic). The narrowest,
lowest-entropy stage: strongest rehearsal and gentlest LR so it can't stamp
"The answer is N" over conversation."""

from __future__ import annotations

from src.models.cognition.stage03_abstraction.sources import SOURCES
from src.models.sdk import StageGate, StagePlugin

PLUGIN = StagePlugin(
    number=3,
    slug="abstraction",
    name="Abstraction and symbolic composition",
    entry_level=1,
    frozen_base=True,
    rehearsal_fraction=0.45,
    lr_scale=0.5,
    gate=StageGate("gsm8k_accuracy", 0.15, "Abstraction — GSM8K"),
    sources=SOURCES,
)
