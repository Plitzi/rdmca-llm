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
    build_heatmap_net,
    build_pose_net,
    mean_keypoint_error,
    soft_argmax,
    synth_batch,
)

_DEFAULT_HIDDEN = 256  # synthetic MLP's default width (used when nothing was trained)


def _build_from_arch(arch: dict):
    """Construct the net that MATCHES a checkpoint, from its `trained_arch` metadata. A real
    heatmap checkpoint (`arch="heatmap"`) → HandHeatmapNet at the trained geometry; anything
    else → the synthetic MLP. Building the wrong shape would silently load nothing."""
    if (arch or {}).get("arch") == "heatmap":
        img_size = int(arch.get("img_size") or 128)
        return build_heatmap_net(
            {
                "img_size": img_size,
                "in_channels": int(arch.get("in_channels") or 3),
                "d_model": int(arch.get("d_model") or 128),
                "heatmap_size": int(arch.get("heatmap_size") or img_size // 4),
            }
        )
    return build_pose_net(int((arch or {}).get("d_model") or _DEFAULT_HIDDEN))


def _load_net(checkpoint: str | None, arch: dict | None = None):
    """Build the net for the checkpoint's architecture (see `_build_from_arch`) and load its
    weights; else random (a plumbing demo). `arch` comes from the framework's `trained_arch`
    (the checkpoint's audit.json), so the camera reconstructs the EXACT trained net."""
    net = _build_from_arch(arch or {})
    kind = getattr(net.cfg, "arch", "mlp")
    if checkpoint and Path(checkpoint).exists():
        backend.current().engine.load_weights(net, checkpoint)
        print(f"  Loaded weights ({kind}, d_model={net.cfg.d_model}): {checkpoint}")
    elif checkpoint:
        print(f"  [warn] checkpoint not found ({checkpoint}); using random weights")
    else:
        print("  No trained checkpoint found — using random weights.")
        print(
            "  Train a real-hand model:  rdmca train "
            "--config models/hands_recognition/configs/hands2d.yaml"
        )
    backend.current().engine.set_eval(net)
    return net


def _zero_input(net) -> np.ndarray:
    """A correctly-shaped zero input for the net (heatmap image [C,H,W] or MLP flat [_IN]) —
    used to warm up the backend before the live loop."""
    cfg = net.cfg
    if getattr(cfg, "arch", None) == "heatmap":
        return np.zeros((cfg.in_channels, cfg.img_size, cfg.img_size), dtype=np.float32)
    return np.zeros(IMG_SIZE * IMG_SIZE, dtype=np.float32)


def _preprocess(net, frame_bgr: np.ndarray, cv2) -> np.ndarray:
    """A BGR webcam frame → the net's input, matching how it was TRAINED. Heatmap FCN: the
    FULL frame resized to img_size, RGB (or gray) in [0,1] as [C,H,W] — the net localizes the
    hand within it. MLP: grayscale IMG_SIZE flattened [_IN]."""
    cfg = net.cfg
    if getattr(cfg, "arch", None) == "heatmap":
        if cfg.in_channels == 1:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (cfg.img_size, cfg.img_size)).astype(np.float32) / 255.0
            return small[None, :, :]  # [1,H,W]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (cfg.img_size, cfg.img_size)).astype(np.float32) / 255.0
        return np.transpose(small, (2, 0, 1))  # [3,H,W]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return (cv2.resize(gray, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0).reshape(-1)


def _predict(net, model_input: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Model input (flat [_IN], image [C,H,W], or already-batched) → (keypoints [21,2] in
    [0,1], depth [21] | None). The heatmap FCN emits spatial heatmaps + depth → soft-argmax
    gives the (x,y) anywhere in the frame and the depth branch the z; the MLP gives only xy.
    Adds the batch dim so callers can pass a single unbatched sample."""
    eng, ops = backend.current().engine, backend.current().ops
    x = np.asarray(model_input, dtype=np.float32)
    if x.ndim in (1, 3):  # [_IN] or [C,H,W] → add batch
        x = x[None]
    out = net(ops.array(x))
    if getattr(net.cfg, "arch", None) == "heatmap":
        heatmaps, z = out
        eng.eval(heatmaps)
        eng.eval(z)
        pts = soft_argmax(np.asarray(ops.to_numpy(heatmaps)))[0]  # [21,2] localized anywhere
        return pts, np.asarray(ops.to_numpy(z))[0]  # [21] root-relative depth
    eng.eval(out)
    return np.array(ops.to_numpy(out)).reshape(N_KEYPOINTS, 2), None


def _depth_color(z_i: float) -> tuple[int, int, int]:
    """Map a root-relative depth (≈[-1,1]; <0 = nearer than the wrist) to a BGR colour:
    near = red, far = blue. So the overlay reads the hand's 3D shape, not just its 2D outline."""
    t = max(0.0, min(1.0, (z_i + 1.0) / 2.0))  # → [0,1]
    return (int(255 * t), 0, int(255 * (1.0 - t)))  # BGR: far→blue, near→red


def _draw_skeleton(cv2, frame, pts: np.ndarray, z: np.ndarray | None = None) -> None:
    """Overlay the articulated hand: a line per bone/phalanx (HAND_CONNECTIONS) and a dot per
    joint, so the 21 landmarks read as a hand skeleton. When depth `z` is available the joints
    are coloured + sized by it (near=red/large, far=blue/small) — a simple 3D cue."""
    h, w = frame.shape[:2]
    px = [(int(x * w), int(y * h)) for x, y in pts]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, px[a], px[b], (0, 200, 255), 2)  # bones (orange)
    for i, (x, y) in enumerate(px):
        if z is not None:
            t = max(0.0, min(1.0, (float(z[i]) + 1.0) / 2.0))
            cv2.circle(frame, (x, y), 3 + int(4 * (1.0 - t)), _depth_color(float(z[i])), -1)
        else:
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)  # joints (green)


def selftest(net) -> int:
    """Headless check that the model + pipeline run without a camera or opencv. For the
    synthetic MLP it reports the keypoint error on a synthetic frame; for the real heatmap
    FCN (no synthetic frames exist) it runs a forward on a zero image and reports the
    localized keypoints + the depth range."""
    if getattr(net.cfg, "arch", None) == "heatmap":
        pts, z = _predict(net, _zero_input(net))
        print(
            f"  selftest: heatmap FCN ({net.cfg.in_channels}ch {net.cfg.img_size}px, "
            f"hs={net.cfg.heatmap_size}) → {N_KEYPOINTS} keypoints + depth"
        )
        print(f"  skeleton: {len(HAND_CONNECTIONS)} bones/phalanges connecting the landmarks")
        print(f"  first 3 keypoints (x,y): {[tuple(round(v, 3) for v in p) for p in pts[:3]]}")
        print(f"  depth z (root-relative): min={z.min():.3f} max={z.max():.3f}")
        print("  OK — model runs end-to-end (feed a real hand via the camera).")
        return 0
    frames, targets = synth_batch(1, seed=1)
    ops = backend.current().ops
    pred = net(ops.array(frames))
    err = mean_keypoint_error(pred, ops.array(targets))
    pts, _z = _predict(net, frames[0])
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

    print("  Opening camera… (first open can take a few seconds on macOS)")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Could not open camera {camera_index}.")
        return 1
    # Keep the loop real-time. A default capture is often 1080p delivered at ~15 FPS over
    # USB; a modest resolution lets the device hit 30/60. BUFFERSIZE=1 always grabs the
    # freshest frame (a backlog adds latency AND starves the key-poll, which made 'q' miss).
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, target_fps)  # ask the device for the chosen rate
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    _predict(net, _zero_input(net))  # warm up the backend (cold compile) before the loop
    frame_budget_ms = max(1, int(1000 / target_fps))  # pace the loop to the target
    print(f"  Camera open — target {target_fps} FPS — press 'q' or ESC to quit.")
    fps = float(target_fps)  # smoothed (EMA) measured rate, seeded at the target
    last = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pts, z = _predict(net, _preprocess(net, frame, cv2))  # preprocess matches training
        _draw_skeleton(cv2, frame, pts, z)
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
        # halved the rate, e.g. 30→15). waitKey also pumps the GUI loop and catches keys.
        spent_ms = (time.perf_counter() - now) * 1000.0
        wait_ms = max(1, int(frame_budget_ms - spent_ms))
        if (cv2.waitKey(wait_ms) & 0xFF) in (ord("q"), 27):  # 'q' or ESC
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
    # model's best/final checkpoint. trained_arch recovers the EXACT architecture it was
    # trained at (heatmap FCN vs MLP, img_size, channels, heatmap_size) so the net is rebuilt
    # to match (no silent shape-mismatch → random).
    from src.training.checkpoint import discover_checkpoint, trained_arch

    checkpoint = args.checkpoint
    if not checkpoint:
        checkpoint, label, _meta = discover_checkpoint("hands_recognition", args.level, args.stage)
        if checkpoint:
            print(f"  Auto-discovered checkpoint [{label}]")
    arch = trained_arch(checkpoint) if checkpoint else {}
    net = _load_net(str(checkpoint) if checkpoint else None, arch)
    if args.selftest:
        return selftest(net)
    return run_camera(net, args.camera_index, target_fps=args.fps)


if __name__ == "__main__":
    raise SystemExit(main())
