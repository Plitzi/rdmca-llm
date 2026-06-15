# hands_recognition — guide

A compact, **non-text, non-transformer** model that proves the RDMCA framework is
task-agnostic: it regresses the **21 standard hand landmarks** (wrist + four joints per
finger) from a downscaled grayscale frame, and — via `HAND_CONNECTIONS` — reconstructs
the **articulated hand skeleton** (every phalanx) for a live camera overlay.

It trains and evaluates through the *same* `ModelSpec` seam as `cognition`
([../pose.py](../pose.py)): only the metric changes — **lower `mpjpe`** (mean keypoint
error) is better. Data is **synthetic** (a Gaussian blob + a fixed, anatomically-plausible
landmark constellation), so nothing is downloaded.

## Layout

```
models/hands_recognition/
  pose.py             HandPoseNet + LANDMARK_NAMES + HAND_CONNECTIONS + build_spec (ModelSpec)
  stages/stage01_keypoints/   the single curriculum stage (keypoint regression)
  uses/camera/run_camera.py   live-camera use case (skeleton overlay + FPS HUD)
  data/               prepared corpora (none — data is generated on the fly)
  docs/GUIDE.md       this file
```

## Use it now — the camera

```bash
# Headless self-test (no webcam, no opencv needed) — proves the model + pipeline run:
rdmca uses camera --selftest

# Live webcam overlay (needs opencv: .venv/bin/python -m pip install opencv-python):
rdmca uses camera                 # 30 FPS (default)
rdmca uses camera --fps 60        # 60 FPS
rdmca uses camera --checkpoint dist/hands_recognition/checkpoints/level0/stage1/best.npz
```

The window overlays the **hand skeleton** (a line per bone/phalanx, a dot per joint) and a
**HUD showing the measured FPS** against the selected target. `--fps {30,60}` both asks the
device for that rate and paces the loop to it. Press `q` to quit. With random weights the
points won't track a real hand — train the model first for meaningful keypoints.

## Train it (you run the training)

The hand-pose model needs **no data prep and no tokenizer** — its `ModelSpec` loader
generates synthetic frames. Steps:

```bash
rdmca info  --model hands_recognition            # confirm the stage is discovered
rdmca train --model hands_recognition --level 0  # train stage 1 (keypoint regression)
rdmca uses camera --checkpoint dist/hands_recognition/checkpoints/level0/stage1/best.npz
```

Checkpoints are namespaced by model: `dist/hands_recognition/checkpoints/level0/stage1/`.
The gate metric is `mpjpe` (set the bar with `gate.max_mpjpe` in the level config).

## How it works

- **Landmarks** (`LANDMARK_NAMES`, 21): `wrist`, then `{thumb,index,middle,ring,pinky}` ×
  `{cmc/mcp, pip, dip, tip}` (the thumb uses cmc/mcp/ip/tip).
- **Skeleton** (`HAND_CONNECTIONS`, 21 bones): 6 palm edges (wrist→finger bases + across the
  knuckles) plus 3 phalanges per finger. Recognizing the 21 points reconstructs the whole
  articulated hand, which is what the overlay draws.
- **Model**: a small MLP (frame → 21×2), trained with mean-squared error on the synthetic
  constellation; `mean_keypoint_error` is the gate/eval metric.

See the framework docs ([../../../docs/README.md](../../../docs/README.md)) for how models,
stages and the `ModelSpec` seam fit together.
