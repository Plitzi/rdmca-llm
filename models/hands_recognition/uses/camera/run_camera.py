#!/usr/bin/env python3
from __future__ import annotations

import os

# Auto-bootstrap: re-run with .venv/bin/python if dependencies are not available.
import sys

try:
    import numpy  # noqa: F401 — just checking the venv is active
except ModuleNotFoundError:
    _repo = os.path.dirname(  # models/hands_recognition/uses/camera/run_camera.py → repo (5 up)
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
    )
    venv_py = os.path.join(_repo, ".venv", "bin", "python")
    if os.path.exists(venv_py) and os.path.abspath(sys.executable) != os.path.abspath(venv_py):
        os.execv(venv_py, [venv_py, *sys.argv])
    print("ERROR: dependencies not found and .venv/bin/python not available.")
    sys.exit(1)

"""
hands_recognition — live camera use case.

Runs the HandPoseNet on webcam frames and overlays the 21 predicted hand keypoints.
This is the model's CONSUMER (the equivalent of cognition's chat), living with its
model. With a trained checkpoint the points track a hand; with random weights it still
proves the capture→preprocess→infer→draw pipeline.

Usage:
  python models/hands_recognition/uses/camera/run_camera.py --selftest     # headless, no camera
  python models/hands_recognition/uses/camera/run_camera.py                # webcam (needs opencv)
  python models/hands_recognition/uses/camera/run_camera.py --checkpoint dist/checkpoints/hands_recognition/level1/stage1/best.npz
  rdmca camera --model hands_recognition --selftest
"""
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repo root on path

import numpy as np

import src.backend as backend
from models.hands_recognition.pose import (
    IMG_SIZE,
    N_KEYPOINTS,
    build_pose_net,
    mean_keypoint_error,
    synth_batch,
)


def _load_net(checkpoint: str | None):
    """Build the net and (optionally) load trained weights; else random (a plumbing demo)."""
    net = build_pose_net()
    if checkpoint and Path(checkpoint).exists():
        backend.current().engine.load_weights(net, checkpoint)
        print(f"  Loaded weights: {checkpoint}")
    elif checkpoint:
        print(f"  [warn] checkpoint not found ({checkpoint}); using random weights")
    else:
        print("  Using random weights (train the model for meaningful keypoints).")
    backend.current().engine.set_eval(net)
    return net


def _predict(net, frame_flat: np.ndarray) -> np.ndarray:
    """frame_flat [_IN] in [0,1] → keypoints [N_KEYPOINTS, 2] in [0,1]."""
    ops = backend.current().ops
    out = net(ops.array(frame_flat.reshape(1, -1).astype(np.float32)))
    backend.current().engine.eval(out)
    return np.array(ops.to_numpy(out)).reshape(N_KEYPOINTS, 2)


def selftest(net) -> int:
    """Headless check: run the net on a synthetic frame and report the keypoint error —
    proves the model + pipeline work without a camera or opencv."""
    frames, targets = synth_batch(1, seed=1)
    ops = backend.current().ops
    pred = net(ops.array(frames))
    err = mean_keypoint_error(pred, ops.array(targets))
    pts = _predict(net, frames[0])
    print(f"  selftest: predicted {N_KEYPOINTS} keypoints | mean error vs target = {err:.4f}")
    print(f"  first 3 keypoints (x,y): {[tuple(round(v, 3) for v in p) for p in pts[:3]]}")
    print("  OK — model runs end-to-end.")
    return 0


def run_camera(net, camera_index: int) -> int:
    try:
        import cv2
    except ModuleNotFoundError:
        print("opencv is not installed. Install it with:")
        print("  .venv/bin/python -m pip install opencv-python")
        print("Or run headless:  rdmca camera --model hands_recognition --selftest")
        return 1

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Could not open camera {camera_index}.")
        return 1
    print("  Camera open — press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        pts = _predict(net, small.reshape(-1))
        h, w = frame.shape[:2]
        for x, y in pts:
            cv2.circle(frame, (int(x * w), int(y * h)), 4, (0, 255, 0), -1)
        cv2.imshow("hands_recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="hands_recognition live camera use case")
    ap.add_argument("--checkpoint", default=None, help="Trained weights .npz (else random)")
    ap.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index")
    ap.add_argument(
        "--selftest", action="store_true", help="Headless synthetic-frame check (no camera)"
    )
    args = ap.parse_args()

    print("Loading hand-pose model…")
    net = _load_net(args.checkpoint)
    if args.selftest:
        return selftest(net)
    return run_camera(net, args.camera_index)


if __name__ == "__main__":
    raise SystemExit(main())
