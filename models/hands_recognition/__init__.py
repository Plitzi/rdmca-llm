"""Model: **hands_recognition** — a hand-pose regressor (21 keypoints from a frame).

The second model, proving the framework is task-agnostic: it is NOT a transformer and
NOT text. It regresses the 21 standard hand landmarks (wrist + four joints per finger)
and, via `HAND_CONNECTIONS`, reconstructs the articulated hand SKELETON (every phalanx).
The engine builds/trains/evaluates it through this package's `ModelSpec` (`build_spec`,
see pose.py), selected with `cfg["model_name"] = "hands_recognition"` (or
`rdmca --model hands_recognition`). Data is synthetic, so it runs with no download.

Layout (same shape as any model):
  • pose.py             — HandPoseNet + landmarks/skeleton + ModelSpec (build/loader/eval)
  • stages/             — its curriculum stages (here, the single keypoint-regression one)
  • uses/camera/        — the live-camera use case overlaying the skeleton (`rdmca camera`)

Moods/emotions don't apply here (a detector has no conversational state), so configs set
`moods: false`. Lower `mpjpe` (mean keypoint error) is better.
"""

from __future__ import annotations

from models.hands_recognition.pose import build_spec

__all__ = ["build_spec"]
