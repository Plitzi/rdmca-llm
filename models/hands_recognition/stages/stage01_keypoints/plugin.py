"""Stage 1 — hand keypoint detection (the only stage of hands_recognition).

A single non-conversational stage and the model's WHOLE curriculum (it is present at every
level — levels differ only in model size). It trains the heatmap FCN on real FreiHAND hands
(or the synthetic MLP when no dataset is present). No text sources (the ModelSpec's loader
provides frames), no mood, and it IS the frozen base (no behavioral LoRA sectors). Gate
metric: mpjpe.
"""

from __future__ import annotations

from src.plugins import StageGate, StagePlugin

PLUGIN = StagePlugin(
    number=1,
    slug="keypoints",
    name="Hand keypoint detection",
    entry_level=0,  # the curriculum stage is active at every level (0 and 1)
    frozen_base=True,
    rehearsal_fraction=0.0,  # single stage → nothing earlier to rehearse
    gate=StageGate(metric_key="mpjpe", threshold=0.12, label="Hand pose — mean keypoint error"),
    # No mood, no freeze point (single-stage model: nothing to freeze a core against).
    sources={},  # data comes from the ModelSpec loader (FreiHAND / synthetic), not prepared to disk
)
