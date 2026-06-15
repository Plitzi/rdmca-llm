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
  python models/hands_recognition/uses/camera/run_camera.py --checkpoint dist/hands_recognition/checkpoints/level1/stage1/best.npz
  rdmca uses camera --selftest     # model inferred (only hands_recognition has `camera`)
"""
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repo root on path

import numpy as np

import src.backend as backend
from models.hands_recognition.pose import (
    HAND_CONNECTIONS,
    IMG_SIZE,
    N_KEYPOINTS,
    build_pose_net,
    mean_keypoint_error,
    synth_batch,
)

_DEFAULT_HIDDEN = 256  # build_pose_net's default width (used when nothing was trained)


def _load_net(checkpoint: str | None, hidden: int = _DEFAULT_HIDDEN):
    """Build the net AT THE CHECKPOINT'S WIDTH and load its weights; else random (a plumbing
    demo). `hidden` MUST match how the checkpoint was trained, or the weights are shape-
    mismatched and silently stay random — the "trained but acts untrained" trap. The caller
    gets `hidden` from the framework's `trained_arch` (the checkpoint's audit.json)."""
    net = build_pose_net(hidden)
    if checkpoint and Path(checkpoint).exists():
        backend.current().engine.load_weights(net, checkpoint)
        print(f"  Loaded weights (d_model={hidden}): {checkpoint}")
    elif checkpoint:
        print(f"  [warn] checkpoint not found ({checkpoint}); using random weights")
    else:
        print("  No trained checkpoint found — using random weights.")
        print("  Train first:  rdmca train --model hands_recognition --level 0")
    backend.current().engine.set_eval(net)
    return net


def _predict(net, frame_flat: np.ndarray) -> np.ndarray:
    """frame_flat [_IN] in [0,1] → keypoints [N_KEYPOINTS, 2] in [0,1]."""
    ops = backend.current().ops
    out = net(ops.array(frame_flat.reshape(1, -1).astype(np.float32)))
    backend.current().engine.eval(out)
    return np.array(ops.to_numpy(out)).reshape(N_KEYPOINTS, 2)


def _draw_skeleton(cv2, frame, pts: np.ndarray) -> None:
    """Overlay the articulated hand: a line per bone/phalanx (HAND_CONNECTIONS) and a
    dot per joint, so the 21 landmarks read as a hand skeleton, not a scatter of points."""
    h, w = frame.shape[:2]
    px = [(int(x * w), int(y * h)) for x, y in pts]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, px[a], px[b], (0, 200, 255), 2)  # bones (orange)
    for x, y in px:
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)  # joints (green)


def selftest(net) -> int:
    """Headless check: run the net on a synthetic frame and report the keypoint error +
    that the full skeleton (every phalanx) is reconstructed — proves the model + pipeline
    work without a camera or opencv."""
    frames, targets = synth_batch(1, seed=1)
    ops = backend.current().ops
    pred = net(ops.array(frames))
    err = mean_keypoint_error(pred, ops.array(targets))
    pts = _predict(net, frames[0])
    print(f"  selftest: predicted {N_KEYPOINTS} keypoints | mean error vs target = {err:.4f}")
    print(f"  skeleton: {len(HAND_CONNECTIONS)} bones/phalanges connecting the landmarks")
    print(f"  first 3 keypoints (x,y): {[tuple(round(v, 3) for v in p) for p in pts[:3]]}")
    print("  OK — model runs end-to-end.")
    return 0


def _draw_hud(cv2, frame, fps: float, target_fps: int) -> None:
    """Top-left HUD: the measured frame rate vs the selected target (30/60)."""
    cv2.putText(
        frame,
        f"{fps:4.1f} FPS  (target {target_fps})",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )


def run_camera(net, camera_index: int, target_fps: int = 30) -> int:
    try:
        import cv2
    except ModuleNotFoundError:
        print("opencv is not installed. Install it with:")
        print("  .venv/bin/python -m pip install opencv-python")
        print("Or run headless:  rdmca uses camera --selftest")
        return 1

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Could not open camera {camera_index}.")
        return 1
    cap.set(cv2.CAP_PROP_FPS, target_fps)  # ask the device for the chosen rate
    frame_budget_ms = max(1, int(1000 / target_fps))  # pace the loop to the target
    print(f"  Camera open — target {target_fps} FPS — press 'q' to quit.")
    fps = float(target_fps)  # smoothed (EMA) measured rate, seeded at the target
    last = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        pts = _predict(net, small.reshape(-1))
        _draw_skeleton(cv2, frame, pts)
        # Measured FPS from the real frame interval, exponentially smoothed.
        now = time.perf_counter()
        dt = now - last
        last = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        _draw_hud(cv2, frame, fps, target_fps)
        cv2.imshow("hands_recognition — hand skeleton (press q)", frame)
        # Pace to the target by waiting only the time LEFT in the frame budget — never the
        # full budget on top of the capture+infer+draw work (that double-counts and roughly
        # halved the rate, e.g. 30→15). waitKey also pumps the GUI loop and catches 'q'.
        spent_ms = (time.perf_counter() - now) * 1000.0
        wait_ms = max(1, int(frame_budget_ms - spent_ms))
        if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="hands_recognition live camera use case")
    ap.add_argument(
        "--checkpoint", default=None, help="Trained weights .npz (else auto-discovered)"
    )
    ap.add_argument("--level", type=int, default=None, help="Restrict auto-discovery to a level")
    ap.add_argument("--stage", type=int, default=None, help="Restrict auto-discovery to a stage")
    ap.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index")
    ap.add_argument(
        "--fps", type=int, default=30, choices=(30, 60), help="Target frame rate (30 or 60)"
    )
    ap.add_argument(
        "--selftest", action="store_true", help="Headless synthetic-frame check (no camera)"
    )
    args = ap.parse_args()

    print("Loading hand-pose model…")
    # Standard framework resolution: an explicit --checkpoint wins, else auto-discover this
    # model's best/final checkpoint. trained_arch recovers the width it was trained at so the
    # net is rebuilt to match (no silent shape-mismatch → random).
    from src.training.checkpoint import discover_checkpoint, trained_arch

    checkpoint = args.checkpoint
    if not checkpoint:
        checkpoint, label, _meta = discover_checkpoint("hands_recognition", args.level, args.stage)
        if checkpoint:
            print(f"  Auto-discovered checkpoint [{label}]")
    hidden = (
        int(trained_arch(checkpoint).get("d_model", _DEFAULT_HIDDEN))
        if checkpoint
        else _DEFAULT_HIDDEN
    )
    net = _load_net(str(checkpoint) if checkpoint else None, hidden)
    if args.selftest:
        return selftest(net)
    return run_camera(net, args.camera_index, target_fps=args.fps)


if __name__ == "__main__":
    raise SystemExit(main())
