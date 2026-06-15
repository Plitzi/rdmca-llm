# hands_recognition — guide

A compact, **non-text, non-transformer** model that proves the RDMCA framework is
task-agnostic: it recovers the **21 standard hand landmarks** (wrist + four joints per
finger) and — via `HAND_CONNECTIONS` — reconstructs the **articulated hand skeleton**
(every phalanx) for a live camera overlay.

It trains and evaluates through the *same* `ModelSpec` seam as `cognition`
([../pose.py](../pose.py)): only the metric changes — **lower `mpjpe`** (mean per-joint
position error) is better.

**Two modes, same model:**
- **Synthetic demo (default)** — a tiny MLP on a Gaussian blob + a fixed landmark
  constellation. Nothing is downloaded; it proves the pipeline but does **not** track a real
  hand (it only learned the synthetic distribution). Predicts 21×2 of a hand that fills the
  frame.
- **Real hands (opt-in)** — a **heatmap FCN** trained on the **FreiHAND** dataset that
  **localizes a real hand anywhere in the frame** and recovers **3D** keypoints (x, y +
  root-relative depth). Enabled by a config + the dataset on disk (see
  [Detect real hands](#detect-real-hands-freihand)). The camera auto-selects whichever
  checkpoint you trained and rebuilds the matching architecture.

## Layout

```
models/hands_recognition/
  pose.py             HandPoseNet (synthetic MLP) + HandHeatmapNet (real FCN) + soft_argmax + build_spec
  data_freihand.py    FreiHandLoader (heatmaps + 3D + localization) + download_freihand
  __init__.py         build_spec + prepare_stage hook (rdmca prepare downloads FreiHAND)
  configs/hands2d.yaml real-hand training config (heatmap FCN + FreiHAND; tier: vision-edge)
  configs/levels/     this model's per-model level configs (the level constructor lives in src/levels.py)
  stages/stage01_keypoints/   the single curriculum stage (keypoint heatmaps)
  uses/camera/run_camera.py   live-camera use case (skeleton + depth overlay + FPS HUD)
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
rdmca uses camera --checkpoint dist/hands_recognition/checkpoints/level1/stage1/best.npz
```

The window overlays the **hand skeleton** (a line per bone/phalanx, a dot per joint) and a
**HUD showing the measured FPS** against the selected target. With a real (heatmap) model the
joints are **coloured + sized by depth** (near = red/large, far = blue/small). `--fps {30,60}`
both asks the device for that rate and paces the loop to it. Press `q` or `ESC` to quit. With
random weights the points won't track a real hand — train the model first.

## Train the synthetic demo (no data, no tokenizer)

The default model needs **no data prep and no tokenizer** — its `ModelSpec` loader generates
synthetic frames:

```bash
rdmca info  --model hands_recognition            # confirm the stage is discovered
rdmca train --model hands_recognition --level 0  # train stage 1 on synthetic data
rdmca uses camera --checkpoint dist/hands_recognition/checkpoints/level0/stage1/best.npz
```

Checkpoints are namespaced by model: `dist/hands_recognition/checkpoints/level0/stage1/`.
The gate metric is `mpjpe` (set the bar with `gate.max_mpjpe` in the level config).

## Detect real hands (FreiHAND)

The synthetic model can't track a real hand. To detect **your** hand — anywhere in the frame,
in 3D — train the **heatmap FCN** on real data. The download is part of the **prepare**
pipeline (just like cognition prepares its corpus), via this model's `prepare_stage` hook:

```bash
# 1. Download + extract FreiHAND into models/hands_recognition/data/freihand/ (~4 GB,
#    idempotent + resumable — re-run to resume; skips if already prepared):
rdmca prepare --config models/hands_recognition/configs/hands2d.yaml

# 2. Train the heatmap FCN on real, localized, 3D hands. Do NOT pass --level (it would
#    override --config):
rdmca train --config models/hands_recognition/configs/hands2d.yaml

# 3. Run the camera — no flags: it auto-discovers the newest checkpoint, reads its audit.json
#    to rebuild the exact FCN (img_size / channels / heatmap_size), and localizes the hand:
rdmca uses camera
```

Checkpoints land in `dist/hands_recognition/checkpoints/level1/stage1/`. Targets come from
the 3D labels: 2D via the camera intrinsics (`uv = K · xyz`, normalized), depth as each
keypoint's camera-z minus the wrist's, divided by the wrist→middle-MCP bone length (so it is
scale-invariant). **Location augmentation** pastes the hand at a random position/scale on a
random background, so the FCN learns to find it anywhere — not just filling the frame.

Tune in [../configs/hands2d.yaml](../configs/hands2d.yaml): `model.img_size`,
`model.heatmap_size` (= img_size/4), `model.in_channels`, `model.dims` (3 = with depth, 2 =
planar), `model.depth_weight`, `dataset.localize`, `training.*`, `gate.max_mpjpe`. The shared
training cadence + resource block come from `tier: vision-edge` (see [the level
constructor](../../../src/levels.py)).

## How it works

- **Landmarks** (`LANDMARK_NAMES`, 21): `wrist`, then `{thumb,index,middle,ring,pinky}` ×
  `{cmc/mcp, pip, dip, tip}` (the thumb uses cmc/mcp/ip/tip).
- **Skeleton** (`HAND_CONNECTIONS`, 21 bones): 6 palm edges (wrist→finger bases + across the
  knuckles) plus 3 phalanges per finger. Recognizing the 21 points reconstructs the whole
  articulated hand, which is what the overlay draws.
- **Models**: synthetic = a small MLP (frame → 21×2, MSE). Real = `HandHeatmapNet`, an
  encoder-decoder that emits 21 spatial **heatmaps** (trained with MSE against Gaussian
  targets) plus a **depth** branch (21 root-relative z). At inference **soft-argmax** turns
  each heatmap into an (x,y) anywhere in the frame; the depth branch gives z → 21×3. The
  metric is `mpjpe` (3D when `model.dims == 3`). The camera rebuilds whichever arch the
  checkpoint was trained as (from its `audit.json` via the framework's `trained_arch`).

See the framework docs ([../../../docs/README.md](../../../docs/README.md)) for how models,
stages and the `ModelSpec` seam fit together.
