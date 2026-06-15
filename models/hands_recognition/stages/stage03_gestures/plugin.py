"""Stage 3 — hand gestures (a BEHAVIORAL head on the frozen multi-hand backbone).

Trains a small classifier on each detected hand's predicted 21×3 keypoints to recognize a
gesture (e.g. thumbs-up). `frozen_base=False` → behavioral: the backbone (and the stage-2 hand
head) stay frozen; only the gesture head trains. Its data is a labelled gesture dataset (a
HaGRID subset) downloaded by the model's `prepare_stage` hook into `dataset.gesture_root`.
Gate: `gesture_err` (1 − classification accuracy).
"""

from __future__ import annotations

from src.plugins import StageGate, StagePlugin

PLUGIN = StagePlugin(
    number=3,
    slug="gestures",
    name="Hand gestures",
    entry_level=0,  # present at every level (the model's curriculum)
    frozen_base=False,  # behavioral: a head on the frozen backbone
    rehearsal_fraction=0.0,
    gate=StageGate(metric_key="gesture_err", threshold=0.3, label="Gestures — 1−accuracy"),
    sources={},  # data comes from the gesture dataset via prepare_stage, not prepared to disk
)
