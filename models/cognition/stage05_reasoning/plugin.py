"""Stage 5 — Reasoning (chain-of-thought). Real GSM8K CoT seed + synthetic <think>
fill. Narrow eroder → strong rehearsal, gentle LR."""

from __future__ import annotations

from models.cognition.stage05_reasoning.sources import SOURCES
from src.plugins.sdk import StageGate, StagePlugin

PLUGIN = StagePlugin(
    number=5,
    slug="reasoning",
    name="Reasoning",
    entry_level=0,
    frozen_base=True,
    rehearsal_fraction=0.45,
    lr_scale=0.5,
    gate=StageGate("reasoning_accuracy", 0.20, "Reasoning — chain-of-thought (GSM8K)"),
    sources=SOURCES,
)
