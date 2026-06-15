"""Stage 2 — handedness + finger state (a BEHAVIORAL head on the frozen multi-hand backbone).

After stage 1 detects up to N hands (positional slots) and freezes, this stage trains a small
head on each present slot's predicted 21×3 keypoints to read: which hand it is (left/right) and
whether each finger is extended or curled. `frozen_base=False` → the trainer treats it as a
behavioral stage (the backbone stays frozen; only the head trains). Labels are free: handedness
from the loader's mirror flag, finger state from the 3D keypoint geometry. Gate: `handstate_err`
(1 − mean accuracy of handedness + finger state). Data comes from the ModelSpec loader (no
prepared corpus on disk).
"""

from __future__ import annotations

from src.plugins import StageGate, StagePlugin

PLUGIN = StagePlugin(
    number=2,
    slug="handstate",
    name="Handedness + finger state",
    entry_level=0,  # present at every level (the model's curriculum)
    frozen_base=False,  # behavioral: a head on the frozen stage-1 backbone
    rehearsal_fraction=0.0,
    gate=StageGate(
        metric_key="handstate_err", threshold=0.3, label="Hand state — 1−accuracy (L/R + fingers)"
    ),
    sources={},  # data comes from the ModelSpec loader (FreiHAND), not prepared to disk
)
