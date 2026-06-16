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

Runs the multi-hand model on webcam frames and overlays, for each detected hand, the 21-point
skeleton (coloured by depth) plus — when the loaded checkpoint trained them — its handedness +
extended-finger count (stage 2) and gesture (stage 3). This is the model's CONSUMER (the
equivalent of cognition's chat). With a trained checkpoint it tracks real hands anywhere in the
frame; with random weights it still proves the capture→preprocess→infer→draw pipeline. Capture
resolution is configurable (720p/1080p+) and decoupled from the model input, so a powerful
camera stays real-time (the model resizes internally).

Usage:
  python models/hands_recognition/uses/camera/run_camera.py --selftest     # headless, no camera
  python models/hands_recognition/uses/camera/run_camera.py                # webcam (needs opencv)
  python models/hands_recognition/uses/camera/run_camera.py --fps 60 --resolution 1920x1080
  rdmca uses camera --selftest     # model inferred (only hands_recognition has `camera`)
"""
import argparse
import re
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repo root on path

import numpy as np

import src.backend as backend
from models.hands_recognition.pose import (
    HAND_CONNECTIONS,
    IMG_SIZE,
    N_KEYPOINTS,
    build_gesture_head,
    build_heatmap_net,
    build_pose_net,
    build_state_head,
    mean_keypoint_error,
    predict_hands,
    synth_batch,
)

_DEFAULT_HIDDEN = 256  # synthetic MLP's default width (used when nothing was trained)


def _build_from_arch(arch: dict, stage: int = 1):
    """Construct the net that MATCHES a checkpoint, from its `trained_arch` metadata and the
    trained `stage`. A real heatmap checkpoint (`arch="heatmap"`) → HandHeatmapNet at the
    trained geometry, plus the behavioral heads the trained stage added (stage ≥ 2 → the
    handedness/finger head). Anything else → the synthetic MLP. Building the wrong shape would
    silently load nothing; the heads load via strict=False (absent keys keep their init)."""
    if (arch or {}).get("arch") == "heatmap":
        img_size = int(arch.get("img_size") or 128)
        net = build_heatmap_net(
            {
                "img_size": img_size,
                "in_channels": int(arch.get("in_channels") or 3),
                "d_model": int(arch.get("d_model") or 128),
                "heatmap_size": int(arch.get("heatmap_size") or img_size // 4),
                "n_hands": int(arch.get("n_hands") or 2),
            }
        )
        if stage >= 2:  # stage-2 checkpoint carries the handedness + finger-state head
            net.state_head = build_state_head()
        if stage >= 3:  # stage-3 checkpoint also carries the gesture head (fixed vocabulary)
            from models.hands_recognition.data_gestures import n_gestures

            net.gesture_head = build_gesture_head(int(arch.get("n_gestures") or n_gestures()))
        return net
    return build_pose_net(int((arch or {}).get("d_model") or _DEFAULT_HIDDEN))


def _load_net(checkpoint: str | None, arch: dict | None = None, stage: int = 1):
    """Build the net for the checkpoint's architecture + stage (see `_build_from_arch`) and
    load its weights; else random (a plumbing demo). `arch` comes from the framework's
    `trained_arch` (the checkpoint's audit.json), so the camera reconstructs the EXACT net."""
    net = _build_from_arch(arch or {}, stage)
    kind = getattr(net.cfg, "arch", "mlp")
    if checkpoint and Path(checkpoint).exists():
        backend.current().engine.load_weights(net, checkpoint)
        print(f"  Loaded weights ({kind}, d_model={net.cfg.d_model}): {checkpoint}")
    elif checkpoint:
        print(f"  [warn] checkpoint not found ({checkpoint}); using random weights")
    else:
        print("  No trained checkpoint found — using random weights.")
        print(
            "  Train a real-hand model:  rdmca prepare --model hands_recognition --level 1"
            "  &&  rdmca train --model hands_recognition --level 1"
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
    hand within it. MLP: grayscale IMG_SIZE flattened [_IN]. NOTE: resizing is coordinate-
    PRESERVING — a landmark at normalized (x,y) stays at (x,y) after the resize, so mapping the
    model's [0,1] output back with (x·w, y·h) lands on the right pixel REGARDLESS of the capture
    aspect ratio (no letterbox needed; the only inaccuracy is the model's own keypoint error)."""
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


def _predict(net, model_input: np.ndarray, presence_thresh: float = 0.5) -> list[dict]:
    """Model input (flat [_IN], image [C,H,W], or already-batched) → a LIST of detected hands,
    one dict per hand: {"pts": [21,2], "z": [21]|None, "slot": int, and (stage ≥ 2) "handed"
    0/1 + "fingers" [5]}. The multi-hand heatmap FCN localizes up to n_hands hands (a slot is
    kept when presence > threshold); a stage-2 head adds handedness + finger state per hand;
    the synthetic MLP returns a single hand. Adds the batch dim for an unbatched sample."""
    ops = backend.current().ops
    x = np.asarray(model_input, dtype=np.float32)
    if x.ndim in (1, 3):  # [_IN] or [C,H,W] → add batch
        x = x[None]
    if getattr(net.cfg, "arch", None) != "heatmap":
        out = net(ops.array(x))
        backend.current().engine.eval(out)
        return [
            {"pts": np.array(ops.to_numpy(out)).reshape(N_KEYPOINTS, 2), "z": None, "slot": 0,
             "conf": 1.0}
        ]  # fmt: skip

    kpts, z, presence = predict_hands(net, x)  # [1,nh,21,2], [1,nh,21], [1,nh]
    state = getattr(net, "state_head", None)
    gesture = getattr(net, "gesture_head", None)
    hands = []
    for s in range(net.cfg.n_hands):
        if (
            presence[0, s] <= presence_thresh
        ):  # below confidence → not a hand, skip (overlay clears)
            continue
        hand = {"pts": kpts[0, s], "z": z[0, s], "slot": s, "conf": float(presence[0, s])}
        feat = (
            np.concatenate([kpts[0, s], z[0, s][:, None]], axis=-1)
            .reshape(1, -1)
            .astype(np.float32)
        )
        if state is not None:  # stage-2 head: handedness + per-finger extended/curled
            hl, fl = state(ops.array(feat))
            hand["handed"] = int(np.asarray(ops.to_numpy(hl)).argmax())
            hand["fingers"] = (np.asarray(ops.to_numpy(fl))[0] > 0).astype(int)
        if gesture is not None:  # stage-3 head: gesture class
            gl = gesture(ops.array(feat))
            hand["gesture"] = int(np.asarray(ops.to_numpy(gl)).argmax())
        hands.append(hand)
    return hands


def _depth_color(z_i: float) -> tuple[int, int, int]:
    """Map a root-relative depth (≈[-1,1]; <0 = nearer than the wrist) to a BGR colour:
    near = red, far = blue. So the overlay reads the hand's 3D shape, not just its 2D outline."""
    t = max(0.0, min(1.0, (z_i + 1.0) / 2.0))  # → [0,1]
    return (int(255 * t), 0, int(255 * (1.0 - t)))  # BGR: far→blue, near→red


def _draw_skeleton(cv2, frame, pts: np.ndarray, z: np.ndarray | None = None) -> None:
    """Overlay the articulated hand: a line per bone/phalanx (HAND_CONNECTIONS) and a dot per
    joint, so the 21 landmarks read as a hand skeleton. `pts` are normalized [0,1] coords scaled
    to the frame (resize is coordinate-preserving, so this lands on the hand at any aspect). When
    depth `z` is available the joints are coloured + sized by it (near=red/large, far=blue)."""
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


def _draw_hand_label(cv2, frame, hand: dict) -> None:
    """Label a detected hand near its wrist: the tracking confidence (the presence probability —
    how sure the model is this is a hand), then handedness (L/R) + extended-finger count
    (stage 2) and the recognized gesture (stage 3) when their heads provided them."""
    parts = [f"{round(hand['conf'] * 100)}%"]  # confidence it's a hand
    if "handed" in hand:
        parts.append(f"{'L' if hand['handed'] == 1 else 'R'} {int(hand['fingers'].sum())}f")
    if "gesture" in hand:
        from models.hands_recognition.data_gestures import GESTURES

        parts.append(GESTURES[hand["gesture"]] if hand["gesture"] < len(GESTURES) else "?")
    h, w = frame.shape[:2]
    wx, wy = int(hand["pts"][0][0] * w), int(hand["pts"][0][1] * h)
    cv2.putText(
        frame, "  ".join(parts), (wx, max(0, wy - 8)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
    )  # fmt: skip


def selftest(net) -> int:
    """Headless check that the model + pipeline run without a camera or opencv. For the
    synthetic MLP it reports the keypoint error on a synthetic frame; for the real heatmap
    FCN (no synthetic frames exist) it runs a forward on a zero image and reports the
    localized keypoints + the depth range."""
    if getattr(net.cfg, "arch", None) == "heatmap":
        hands = _predict(
            net, _zero_input(net), presence_thresh=-1.0
        )  # force all slots for the check
        print(
            f"  selftest: multi-hand heatmap FCN ({net.cfg.in_channels}ch {net.cfg.img_size}px, "
            f"hs={net.cfg.heatmap_size}, n_hands={net.cfg.n_hands}) → {N_KEYPOINTS} keypoints + depth/slot"
        )
        print(f"  skeleton: {len(HAND_CONNECTIONS)} bones/phalanges connecting the landmarks")
        for hand in hands:
            kp = [tuple(round(v, 3) for v in p) for p in hand["pts"][:3]]
            print(
                f"  slot {hand['slot']}: first 3 keypoints {kp} | "
                f"depth z min={hand['z'].min():.3f} max={hand['z'].max():.3f}"
            )
        print("  OK — model runs end-to-end (feed real hands via the camera).")
        return 0
    frames, targets = synth_batch(1, seed=1)
    ops = backend.current().ops
    pred = net(ops.array(frames))
    err = mean_keypoint_error(pred, ops.array(targets))
    pts = _predict(net, frames[0])[0]["pts"]
    print(f"  selftest: predicted {N_KEYPOINTS} keypoints | mean error vs target = {err:.4f}")
    print(f"  skeleton: {len(HAND_CONNECTIONS)} bones/phalanges connecting the landmarks")
    print(f"  first 3 keypoints (x,y): {[tuple(round(v, 3) for v in p) for p in pts[:3]]}")
    print("  OK — model runs end-to-end.")
    return 0


def _parse_resolution(spec: str) -> tuple[int, int] | None:
    """`--resolution` → the (width, height) to request from the camera. 'auto' (default)
    asks for 1280×720, a good real-time baseline that most webcams support; 'WxH' (e.g.
    1920x1080) requests that exactly. The device may honour the nearest mode it supports;
    EITHER WAY the model resizes to img_size internally, so inference cost is unchanged."""
    if not spec or spec.lower() == "auto":
        return (1280, 720)
    try:
        w, h = (int(v) for v in spec.lower().split("x", 1))
        return (w, h)
    except ValueError:
        print(f"  [warn] bad --resolution '{spec}' (expected WxH or 'auto'); using auto.")
        return (1280, 720)


def _draw_hud(cv2, frame, fps: float, target_fps: int, cap_wh: tuple[int, int]) -> None:
    """Top-left HUD: measured frame rate vs the selected target, and the capture resolution."""
    cv2.putText(
        frame,
        f"{fps:4.1f} FPS  (target {target_fps})  {cap_wh[0]}x{cap_wh[1]}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )


def run_camera(
    net,
    camera_index: int,
    target_fps: int = 30,
    resolution: tuple[int, int] = (1280, 720),
    min_confidence: float = 0.5,
) -> int:
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
    # Capture at the requested resolution (default 720p; 1080p+ fine). The model ALWAYS
    # downscales to its img_size in _preprocess, so inference cost is CONSTANT regardless of
    # capture resolution — that decoupling is what keeps a powerful camera at 30/60 FPS.
    # BUFFERSIZE=1 always grabs the freshest frame (a backlog adds latency AND starves the
    # key-poll, which made 'q' miss).
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap.set(cv2.CAP_PROP_FPS, target_fps)  # ask the device for the chosen rate
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # What the device actually gave us (it may snap to the nearest supported mode).
    cap_wh = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or resolution[0]),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or resolution[1]),
    )
    _predict(net, _zero_input(net), min_confidence)  # warm up the backend (cold compile)
    frame_budget_ms = max(1, int(1000 / target_fps))  # pace the loop to the target
    print(
        f"  Camera open — {cap_wh[0]}x{cap_wh[1]} — target {target_fps} FPS — "
        "press 'q' or ESC to quit."
    )
    fps = float(target_fps)  # smoothed (EMA) measured rate, seeded at the target
    last = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        hands = _predict(net, _preprocess(net, frame, cv2), min_confidence)  # gated by confidence
        for hand in hands:  # draw every detected hand (up to n_hands)
            _draw_skeleton(cv2, frame, hand["pts"], hand["z"])
            _draw_hand_label(cv2, frame, hand)  # confidence% + L/R + fingers + gesture
        # Measured FPS from the real frame interval, exponentially smoothed.
        now = time.perf_counter()
        dt = now - last
        last = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        _draw_hud(cv2, frame, fps, target_fps, cap_wh)
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
        "--resolution",
        default="auto",
        help="Capture resolution 'WxH' (e.g. 1920x1080) or 'auto' (=1280x720). The model "
        "resizes internally, so higher res doesn't slow inference.",
    )
    ap.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Min presence probability to draw a hand (0-1). Raise it if the skeleton appears "
        "on an empty scene; lower it if real hands are missed. Default 0.5.",
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
    # The trained STAGE (from the checkpoint's .../stageN/ path) tells the camera which
    # behavioral heads to rebuild (stage ≥ 2 → handedness/finger head) so they load too.
    stage_match = re.search(r"stage(\d+)", str(checkpoint or ""))
    stage = int(stage_match.group(1)) if stage_match else 1
    net = _load_net(str(checkpoint) if checkpoint else None, arch, stage)
    if args.selftest:
        return selftest(net)
    return run_camera(
        net,
        args.camera_index,
        target_fps=args.fps,
        resolution=_parse_resolution(args.resolution),
        min_confidence=args.min_confidence,
    )


if __name__ == "__main__":
    raise SystemExit(main())
