# hands_recognition — guide

A compact, **non-text, non-transformer** model that proves the RDMCA framework is
task-agnostic: it regresses the **21 standard hand landmarks** (wrist + four joints per
finger) from a downscaled grayscale frame, and — via `HAND_CONNECTIONS` — reconstructs
the **articulated hand skeleton** (every phalanx) for a live camera overlay.

It trains and evaluates through the *same* `ModelSpec` seam as `cognition`
([../pose.py](../pose.py)): only the metric changes — **lower `mpjpe`** (mean keypoint
error) is better.

**Two modes, same model:**
- **Synthetic demo (default)** — a tiny MLP on a Gaussian blob + a fixed landmark
  constellation. Nothing is downloaded; it proves the pipeline but does **not** track a real
  hand (it only learned the synthetic distribution).
- **Real hands (opt-in)** — a small **CNN** trained on the **FreiHAND** dataset to detect a
  real hand from the webcam. Enabled by a config + the dataset on disk (see
  [Detect real hands](#detect-real-hands-freihand)). The camera auto-selects whichever
  checkpoint you trained and rebuilds the matching architecture.

## Layout

```
models/hands_recognition/
  pose.py             HandPoseNet (synthetic MLP) + HandPoseCNN (real) + build_spec (ModelSpec)
  data_freihand.py    FreiHandLoader — real RGB hands → projected 21×2 keypoints
  configs/hands2d.yaml real-hand training config (CNN + FreiHAND)
  stages/stage01_keypoints/   the single curriculum stage (keypoint regression)
  uses/camera/run_camera.py   live-camera use case (skeleton overlay + FPS HUD)
  data/freihand/      the downloaded FreiHAND dataset (gitignored; only for real mode)
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

## Detect real hands (FreiHAND)

The synthetic model above can't track a real hand. To detect **your** hand, train the CNN
on real data — the **FreiHAND** dataset (real RGB hands + 21 keypoints):

1. **Download** FreiHAND (FreiHAND_pub_v2) and **unzip into** `models/hands_recognition/data/freihand/`
   — the folder must contain `training_xyz.json`, `training_K.json`, and `training/rgb/*.jpg`.
   (Dataset page: https://lmb.informatik.uni-freiburg.de/projects/freihand/ — gitignored.)
2. **Train** with the real-hand config (it selects the CNN + the real loader). Do **not**
   pass `--level` (it would override `--config`):

   ```bash
   rdmca train --config models/hands_recognition/configs/hands2d.yaml
   ```

   Checkpoints land in `dist/hands_recognition/checkpoints/level1/stage1/`. The 2D keypoints
   come from projecting the 3D labels with the camera intrinsics (`uv = K · xyz`, normalized).
3. **Run the camera** — no flags needed: it auto-discovers the newest checkpoint, reads its
   `audit.json` to rebuild the exact CNN (img_size / channels), and preprocesses each frame
   to match training:

   ```bash
   rdmca uses camera
   ```

**v1 limitation:** this is a single-hand **regressor** — hold ONE hand so it roughly **fills
the frame** (there is no detect-then-crop stage yet). It is **2D** (the overlay is a 2D
skeleton); depth/3D would be a future `21×3` head. Tune size/speed in
[../configs/hands2d.yaml](../configs/hands2d.yaml) (`model.img_size`, `model.in_channels`,
`training.*`, `gate.max_mpjpe`).

## How it works

- **Landmarks** (`LANDMARK_NAMES`, 21): `wrist`, then `{thumb,index,middle,ring,pinky}` ×
  `{cmc/mcp, pip, dip, tip}` (the thumb uses cmc/mcp/ip/tip).
- **Skeleton** (`HAND_CONNECTIONS`, 21 bones): 6 palm edges (wrist→finger bases + across the
  knuckles) plus 3 phalanges per finger. Recognizing the 21 points reconstructs the whole
  articulated hand, which is what the overlay draws.
- **Model**: synthetic mode = a small MLP (frame → 21×2); real mode = `HandPoseCNN` (RGB
  image → 21×2, strided convs + global pool + MLP head). Both train with mean-squared error;
  `mean_keypoint_error` is the gate/eval metric. The camera rebuilds whichever one the
  checkpoint was trained as (from its `audit.json` via the framework's `trained_arch`).

See the framework docs ([../../../docs/README.md](../../../docs/README.md)) for how models,
stages and the `ModelSpec` seam fit together.
