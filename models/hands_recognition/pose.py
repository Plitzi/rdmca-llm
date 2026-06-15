"""hands_recognition — a small hand-pose regressor + its ModelSpec.

A deliberately compact, self-contained vision model that proves the framework is
task-agnostic: NOT a transformer and NOT text. It regresses the 21 standard hand
landmarks (wrist + four joints per finger, x/y in [0, 1]) from a downscaled grayscale
frame; together with `HAND_CONNECTIONS` (the bones/phalanges) those landmarks form an
articulated hand SKELETON the camera overlay draws. The data is SYNTHETIC (a bright blob
at a random spot → a fixed, anatomically-plausible landmark constellation around it), so
the whole pipeline runs with no dataset download — a real, learnable toy task.

Built on the shared backend (`src.backend`), so it trains/infers on MLX or torch like
the rest of the framework. The camera use case (uses/camera) consumes this net live and
overlays the skeleton (every phalanx) on the webcam feed.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import src.backend as backend

B = backend.current()
nn = B.nn
ops = B.ops

IMG_SIZE = 32  # frames are downscaled to IMG_SIZE×IMG_SIZE grayscale
N_KEYPOINTS = 21  # standard hand-landmark count (MediaPipe topology)
_IN = IMG_SIZE * IMG_SIZE
_OUT = N_KEYPOINTS * 2

# Landmark index map (the standard 21-point hand model): a wrist plus four joints per
# finger — the metacarpophalangeal (MCP), proximal (PIP) and distal (DIP) joints and the
# fingertip (TIP). Naming each landmark lets the overlay label fingers/phalanges.
LANDMARK_NAMES = (
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)  # fmt: skip
assert len(LANDMARK_NAMES) == N_KEYPOINTS

# The hand SKELETON: each pair is a bone (a phalanx along a finger, or a palm edge). The
# camera overlay draws a line per connection, so recognizing the 21 landmarks reconstructs
# the articulated hand (every phalanx), not just a scatter of points.
HAND_CONNECTIONS = (
    # palm (wrist → each finger base, and across the knuckles)
    (0, 1), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17),
    (1, 2), (2, 3), (3, 4),            # thumb phalanges
    (5, 6), (6, 7), (7, 8),            # index phalanges
    (9, 10), (10, 11), (11, 12),       # middle phalanges
    (13, 14), (14, 15), (15, 16),      # ring phalanges
    (17, 18), (18, 19), (19, 20),      # pinky phalanges
)  # fmt: skip

# A fixed, anatomically-plausible hand constellation (offsets from the palm centre, in
# image coords: +x right, +y DOWN). The synthetic target places this around the blob so
# the net learns a consistent, hand-shaped layout it can articulate.
_HAND_LANDMARKS = np.array(
    [
        (0.00, 0.90),  # 0  wrist (below the palm)
        (-0.25, 0.65),
        (-0.45, 0.45),
        (-0.60, 0.30),
        (-0.72, 0.18),  # thumb (fans left)
        (-0.22, 0.35),
        (-0.26, 0.05),
        (-0.28, -0.15),
        (-0.30, -0.32),  # index
        (0.00, 0.32),
        (0.00, -0.02),
        (0.00, -0.24),
        (0.00, -0.42),  # middle (longest)
        (0.20, 0.35),
        (0.22, 0.02),
        (0.24, -0.18),
        (0.26, -0.34),  # ring
        (0.38, 0.40),
        (0.42, 0.18),
        (0.45, 0.02),
        (0.48, -0.12),  # pinky
    ],
    dtype=np.float32,
)
assert _HAND_LANDMARKS.shape == (N_KEYPOINTS, 2)


class HandPoseNet(nn.Module):
    """MLP: flattened grayscale frame → 21×2 keypoints. Small enough to train on CPU."""

    def __init__(self, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(_IN, hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.fc3 = nn.Linear(hidden // 2, _OUT)
        # A tiny config so the trainer/dashboard can introspect it like any model.
        self.cfg = SimpleNamespace(n_layers=3, d_model=hidden, context_len=_IN)

    def __call__(self, x):  # x: [N, _IN] in [0,1]
        h = ops.relu(self.fc1(x))
        h = ops.relu(self.fc2(h))
        return self.fc3(h)  # [N, _OUT]

    def count_params(self, include_sectors: bool = True) -> int:
        h = self.cfg.d_model
        return (_IN * h + h) + (h * (h // 2) + h // 2) + ((h // 2) * _OUT + _OUT)


def build_pose_net(hidden: int = 256) -> HandPoseNet:
    """Construct the net with weights allocated (a forward pass materializes params)."""
    net = HandPoseNet(hidden)
    _ = net(ops.array(np.zeros((1, _IN), dtype=np.float32)))
    B.engine.eval(net.parameters())
    return net


def synth_batch(n: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """A batch of (frames, keypoints): a Gaussian blob at a random centre, with the hand
    constellation laid around it. frames [n, _IN] float32, keypoints [n, _OUT] in [0,1]."""
    rng = np.random.default_rng(seed)
    frames = np.zeros((n, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    keypts = np.zeros((n, N_KEYPOINTS, 2), dtype=np.float32)
    ys, xs = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    for i in range(n):
        cx, cy = rng.uniform(0.25, 0.75, size=2)  # blob centre in [0,1]
        px, py = cx * IMG_SIZE, cy * IMG_SIZE
        frames[i] = np.exp(-(((xs - px) ** 2 + (ys - py) ** 2) / (2 * 4.0**2)))
        pts = np.stack([cx, cy]) + 0.18 * _HAND_LANDMARKS  # constellation around the centre
        keypts[i] = np.clip(pts, 0.0, 1.0)
    return frames.reshape(n, _IN), keypts.reshape(n, _OUT)


def _mse(pred, target):
    diff = pred - target
    return ops.mean(diff * diff)


def mean_keypoint_error(pred, target) -> float:
    """Mean per-keypoint Euclidean error (the gate metric, lower = better)."""
    d = pred - target
    return float(B.engine.item(ops.mean(ops.sqrt(ops.mean(d * d, axis=-1) + 1e-9))))


class _SynthLoader:
    """Minimal training loader the trainer can drive: yields (frames, keypoints) numpy
    batches. Vision data is generated on the fly, so there is no on-disk corpus, skip
    index or replay (those text-stream concepts are no-ops here)."""

    def __init__(self, batch_size: int, seed: int = 0):
        self.batch_size = batch_size
        self._rng_seed = seed
        self._step = 0
        self.epoch_tokens = 0  # unbounded synthetic stream → no corpus cap
        self.passes = 0
        self.last_was_replay = False
        self.replay_fraction = 0.0

    def next_batch(self):
        self._step += 1
        return synth_batch(self.batch_size, seed=self._rng_seed + self._step)

    def skip(self, n: int) -> int:
        self._step += n
        return n

    def save_skip_index(self, path) -> None:  # no corpus to index
        pass

    def load_skip_index(self, path) -> bool:
        return False


def build_spec(cfg: dict):
    """The hands_recognition ModelSpec — how the framework builds/trains/evaluates it.
    Lower `mpjpe` (mean keypoint error) is better, matching the trainer's ratchet."""
    from src.plugins import ModelSpec

    def build_model(stage: int, cfg: dict, root):
        seed = int((cfg.get("training", {}) or {}).get("seed", 0))
        B.engine.set_seed(seed)
        net = build_pose_net(int(cfg.get("model", {}).get("d_model", 256)))
        precision = (cfg.get("training", {}) or {}).get("precision", "fp32")
        return net, net.cfg, None, precision, seed

    def build_loader(stage: int, cfg: dict):
        bs = int((cfg.get("training", {}) or {}).get("batch_size", 32))
        return _SynthLoader(bs, seed=int((cfg.get("training", {}) or {}).get("seed", 0)))

    def objective(model, batch):
        frames, keypts = batch  # (images, targets)
        return _mse(model(frames), keypts)

    def evaluate(model, stage, val_batches=None, cfg=None, log=print, step=None):
        B.engine.set_eval(model)
        errs = [mean_keypoint_error(model(f), k) for f, k in (val_batches or [])]
        B.engine.set_train(model)
        score = float(np.mean(errs)) if errs else float("inf")
        threshold = float(((cfg or {}).get("gate", {}) or {}).get("max_mpjpe", 0.05))
        passed = score <= threshold
        tag = f"step={step:,} | " if step is not None else ""
        log(f"[gate] {tag}mpjpe={score:.4f} <= {threshold:.4f} -> {'PASS' if passed else 'fail'}")
        return score, passed

    return ModelSpec(
        name="hand-pose",
        build_model=build_model,
        build_loader=build_loader,
        objective=objective,
        evaluate=evaluate,
        gate_metric="mpjpe",
    )
