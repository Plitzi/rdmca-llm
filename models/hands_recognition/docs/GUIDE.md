# hands_recognition — guide

A compact, **non-text, non-transformer** model that proves the RDMCA framework is
task-agnostic. Built for **VR**: a webcam captures the scene, and the model finds **both your
hands anywhere in the frame**, recovers their 21 landmarks in **3D**, tells **left from right**,
reads each **finger's** state, and recognizes **gestures** (e.g. thumbs-up).

It trains and evaluates through the *same* `ModelSpec` seam as `cognition`
([../pose.py](../pose.py)); lower scores are better (mpjpe for the detector, 1−accuracy for the
behavioral heads).

**Two architectures, one ModelSpec:**
- **Synthetic demo (fallback)** — a tiny MLP on a Gaussian blob + a fixed constellation. No
  download; the headless selftest + CI path. Single hand, 2D, fills the frame.
- **Real, multi-hand (the model)** — a **multi-hand heatmap FCN**: per slot (up to `n_hands`,
  default 2) it emits 21 spatial heatmaps + a depth branch + a presence logit. Soft-argmax
  localizes each hand anywhere in the frame. Trained on **FreiHAND** with location +
  multi-hand augmentation.

**Levels = model SIZE; the 3 stages = the curriculum** (the RDMCA convention). Every level runs
the same curriculum; `level 0` is a small/fast FCN (64 px), `level 1` the standard one (128 px).
The three stages share one backbone (the framework's frozen-core + behavioral-head pattern):

| Stage | Kind | Learns | Data | Gate |
|------|------|--------|------|------|
| 1 `keypoints` | cognitive (frozen base) | up to 2 hands' 21 3D keypoints + presence | FreiHAND | `mpjpe` |
| 2 `handstate` | behavioral head | per hand: left/right + each finger extended/curled | FreiHAND (free labels) | `handstate_err` |
| 3 `gestures` | behavioral head | per hand: gesture class (thumbs-up, …) | gesture dataset (HaGRID subset) | `gesture_err` |

After stage 1 the backbone freezes; stages 2 and 3 train only their small head on top, so each
later checkpoint keeps every earlier capability.

## Layout

```
models/hands_recognition/
  pose.py             HandHeatmapNet (multi-hand) + HandStateHead + GestureHead + soft_argmax + build_spec
  data_freihand.py    FreiHandLoader (multi-hand heatmaps + 3D + localization + labels) + download_freihand
  data_gestures.py    GestureLoader + download_gestures (the gesture dataset)
  __init__.py         build_spec + prepare_stage hook (downloads FreiHAND or gestures, by stage)
  configs/levels/     per-model ladder: _base.yaml (3-stage curriculum) + level0/level1 (SIZE only)
  stages/stage01_keypoints/ · stage02_handstate/ · stage03_gestures/   the curriculum stages
  uses/camera/run_camera.py   live camera (≤2 skeletons + depth, L/R + fingers, gesture; FPS HUD)
  data/{freihand,gestures}/   downloaded datasets (gitignored)
  docs/GUIDE.md       this file
```

The level constructor (boilerplate shared across models: training cadence + resources) lives in
[../../../src/levels.py](../../../src/levels.py); each level opts in with `tier: vision-edge`.

## Use it now — the camera

```bash
rdmca uses camera --selftest                  # headless, no webcam/opencv — proves the pipeline
rdmca uses camera                             # webcam (needs opencv); 30 FPS, auto 720p
rdmca uses camera --fps 60 --resolution 1920x1080   # 60 FPS on a 1080p camera
```

The window draws, for every detected hand: the **skeleton** (a line per phalanx, joints
**coloured + sized by depth** — near = red/large, far = blue/small) and, when the loaded
checkpoint trained them, an **L/R** label + extended-finger count (stage 2) and the **gesture**
name (stage 3). A HUD shows measured FPS and capture resolution. Press `q`/`ESC` to quit.

**Resolution & 60 FPS.** `--resolution` accepts `WxH` (e.g. `1920x1080`) or `auto` (=1280×720);
the camera works with 720p/1080p and more powerful cameras. The model always resizes the frame
to its `img_size` internally, so **inference cost is independent of capture resolution** — that
decoupling is what keeps a powerful camera real-time. The bottleneck is capture/sync, not the
(tiny) FCN; `--level 0` is the fast tier, `--level 1` the standard one.

## Train it (you run the training)

Standard form: `--model hands_recognition` (+ `--level`), no `--config`. `prepare` downloads the
right dataset for each stage via the model's `prepare_stage` hook (the same data step cognition
uses for its corpus):

```bash
rdmca info --model hands_recognition          # confirm the 3 stages + levels are discovered

rdmca prepare --model hands_recognition --level 1   # downloads FreiHAND (stages 1-2)
rdmca train   --model hands_recognition --level 1   # trains stage 1 → 2 → 3 in order
rdmca uses camera                                   # auto-discovers the newest checkpoint
```

Stage 3 downloads the gesture dataset (a HaGRID subset) in its own prepare step. Checkpoints are
namespaced by model + level + stage: `dist/hands_recognition/checkpoints/level{L}/stage{N}/`. The
camera reads the checkpoint's `audit.json` (`trained_arch`) to rebuild the EXACT geometry
(img_size / channels / heatmap_size / n_hands) and, from the `stageN` path, which behavioral
heads to attach.

## How it works

- **Multi-hand detector (stage 1).** Slots are POSITIONAL — training composites up to `n_hands`
  real hands (some mirrored → left hands) onto random backgrounds and assigns the left-most to
  slot 0; a presence logit gates each slot. Depth is root-relative (camera-z minus the wrist's,
  divided by the wrist→middle-MCP bone length → scale-invariant). Soft-argmax decodes each slot.
- **Handedness + finger state (stage 2).** A small MLP head on the frozen backbone's predicted
  21×3 keypoints. Labels are free: handedness from the loader's mirror flag, finger
  extended/curled from the 3D keypoint geometry.
- **Gestures (stage 3).** A classifier head on the dominant hand's keypoints, trained on the
  gesture dataset. Add a gesture by extending `data_gestures.GESTURES` + its dataset folder.

**Honest scope.** Quality ("tracks MY hands, both, with the right gestures") depends on the
training you run; the architecture (multi-hand heatmaps + location augmentation + behavioral
heads) is the standard path. Up to 2 hands; depth is 2.5D (root-relative), not absolute metric.
The gesture dataset URL is pinned in `data_gestures._GESTURES_URL` (a HaGRID subset) — set it
before `prepare`-ing stage 3.

See the framework docs ([../../../docs/README.md](../../../docs/README.md)) for how models,
stages and the `ModelSpec` seam fit together.
