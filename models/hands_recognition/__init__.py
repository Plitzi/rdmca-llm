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
  • uses/camera/        — the live-camera use case overlaying the skeleton (`rdmca uses camera`)

Moods/emotions don't apply here (a detector has no conversational state), so configs set
`moods: false`. Lower `mpjpe` (mean keypoint error) is better.
"""

from __future__ import annotations

from models.hands_recognition.pose import build_spec

__all__ = ["build_spec", "prepare_stage"]


def prepare_stage(stage: int, cfg: dict, langs: list[str], limit_mb: int | None = None) -> None:
    """Data-preparation hook for hands_recognition (discovered via `model_hook` — the same
    pattern as `post_stage`). `rdmca prepare` delegates here instead of the text-corpus flow,
    and dispatches by stage: stages 1-2 (detector + hand state) need the real FreiHAND dataset;
    stage 3 (gestures) needs the labelled gesture dataset. With no dataset root the model trains
    on the synthetic stream, so there is nothing to prepare. `langs`/`limit_mb` don't apply."""
    dcfg = cfg.get("dataset", {}) or {}
    if stage >= 3:
        gesture_root = dcfg.get("gesture_root")
        if not gesture_root:
            print(f"  Stage {stage}: no dataset.gesture_root — nothing to download.")
            return
        from models.hands_recognition.data_gestures import download_gestures

        download_gestures(gesture_root)
        return
    root = dcfg.get("root")
    if not root:
        print(f"  Stage {stage}: synthetic data (no dataset.root) — nothing to download.")
        return
    from models.hands_recognition.data_freihand import download_freihand

    download_freihand(root)
