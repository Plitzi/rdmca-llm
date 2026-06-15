"""hands_recognition — hand-pose regressors + their ModelSpec.

A compact vision model that proves the framework is task-agnostic: NOT a transformer and
NOT text. It regresses the 21 standard hand landmarks (wrist + four joints per finger);
with `HAND_CONNECTIONS` (the bones/phalanges) those form an articulated hand SKELETON the
camera overlay draws. Two interchangeable architectures, same ModelSpec:

  • HandPoseNet — a tiny MLP on a downscaled grayscale frame, trained on SYNTHETIC data
    (a blob + a fixed landmark constellation). No download; proves the pipeline but does
    not track a REAL hand. Predicts 21×2 (x,y in [0,1]) of a hand that fills the frame.
  • HandHeatmapNet — a fully-convolutional encoder-decoder that emits 21 spatial HEATMAPS
    plus a per-keypoint DEPTH (z relative to the wrist) → 21×3. Soft-argmax turns each
    heatmap into an (x,y) anywhere in the frame, so it LOCALIZES a real hand wherever it
    is (not just filling the frame). Trained on the real FreiHAND dataset (see
    data_freihand.py) with location augmentation. Opt-in via the model's level configs
    (configs/levels/levelN.yaml, `model.arch: heatmap`).

`build_spec` picks the arch + loader from the config (`model.arch`, `dataset.root`). Built
on the shared backend (`src.backend`) → trains/infers on MLX or torch. The camera use case
(uses/camera) rebuilds whichever arch the checkpoint was trained as and overlays the skeleton.
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


class HandHeatmapNet(nn.Module):
    """Fully-convolutional hand pose for REAL, LOCALIZED hands: an RGB (or gray) image
    [N, C, H, W] in [0,1] → 21 spatial heatmaps [N, 21, hs, hs] + a per-keypoint depth
    [N, 21] (z relative to the wrist). Unlike a flat regressor it keeps spatial structure
    end-to-end, so soft-argmax over each heatmap recovers an (x,y) ANYWHERE in the frame —
    it localizes the hand wherever it is, instead of assuming it fills the frame.

    Encoder: four stride-2 conv blocks (H,W shrink 16× → bottleneck img_size/16). Decoder:
    two ConvTranspose2d up-samples (→ img_size/4 = heatmap_size) then a 1×1 conv to 21
    channels. Depth branch: global-average-pooled bottleneck → MLP → 21. Requires
    `heatmap_size == img_size // 4` (the two up-samples from the /16 bottleneck)."""

    def __init__(
        self, img_size: int = 128, in_channels: int = 3, width: int = 128, heatmap_size: int = 32
    ):
        super().__init__()
        if heatmap_size != img_size // 4:
            raise ValueError(
                f"heatmap_size must be img_size//4 ({img_size // 4}), got {heatmap_size} "
                f"(the decoder up-samples the /16 bottleneck twice)."
            )
        self.c1 = nn.Conv2d(in_channels, width // 8, 3, stride=2, padding=1)  # H/2
        self.c2 = nn.Conv2d(width // 8, width // 4, 3, stride=2, padding=1)  # H/4
        self.c3 = nn.Conv2d(width // 4, width // 2, 3, stride=2, padding=1)  # H/8
        self.c4 = nn.Conv2d(width // 2, width, 3, stride=2, padding=1)  # H/16 (bottleneck)
        self.d1 = nn.ConvTranspose2d(width, width // 2, 4, stride=2, padding=1)  # H/8
        self.d2 = nn.ConvTranspose2d(width // 2, width // 4, 4, stride=2, padding=1)  # H/4
        self.heatmap = nn.Conv2d(width // 4, N_KEYPOINTS, 1)  # → 21 heatmaps at H/4
        self.depth1 = nn.Linear(width, width)
        self.depth2 = nn.Linear(width, N_KEYPOINTS)  # 21 root-relative depths
        # Record geometry so the audit captures it and the camera rebuilds the EXACT net.
        self.cfg = SimpleNamespace(
            arch="heatmap",
            img_size=img_size,
            in_channels=in_channels,
            d_model=width,
            heatmap_size=heatmap_size,
            dims=3,
            n_layers=6,
            context_len=img_size,
        )

    def __call__(self, x):  # x: [N, C, H, W] in [0,1] → (heatmaps [N,21,hs,hs], z [N,21])
        h = ops.relu(self.c1(x))
        h = ops.relu(self.c2(h))
        h = ops.relu(self.c3(h))
        bottleneck = ops.relu(self.c4(h))  # [N, width, H/16, W/16]
        u = ops.relu(self.d1(bottleneck))
        u = ops.relu(self.d2(u))
        heatmaps = self.heatmap(u)  # [N, 21, hs, hs] (raw — softmax happens in soft-argmax)
        pooled = ops.mean(bottleneck, axis=(2, 3))  # global average pool → [N, width]
        z = self.depth2(ops.relu(self.depth1(pooled)))  # [N, 21] root-relative depth
        return heatmaps, z

    def count_params(self, include_sectors: bool = True) -> int:
        w, cin, k = self.cfg.d_model, self.cfg.in_channels, N_KEYPOINTS

        def conv(i, o, ksz):
            return i * o * ksz * ksz + o

        return (
            conv(cin, w // 8, 3)
            + conv(w // 8, w // 4, 3)
            + conv(w // 4, w // 2, 3)
            + conv(w // 2, w, 3)
            + conv(w, w // 2, 4)  # ConvTranspose2d (same weight count)
            + conv(w // 2, w // 4, 4)
            + conv(w // 4, k, 1)
            + (w * w + w)
            + (w * k + k)
        )


def soft_argmax(heatmaps_np: np.ndarray) -> np.ndarray:
    """[N,21,hs,hs] raw heatmaps → [N,21,2] (x,y) in [0,1] via 2D spatial soft-argmax.
    Inference/eval only (no gradient), so it runs in numpy: a spatial softmax over each
    heatmap then the expected grid coordinate. Localizes the peak anywhere in the frame."""
    n, k, hh, ww = heatmaps_np.shape
    flat = heatmaps_np.reshape(n, k, hh * ww).astype(np.float64)
    flat = flat - flat.max(axis=-1, keepdims=True)  # stabilize the exp
    p = np.exp(flat)
    p /= p.sum(axis=-1, keepdims=True)
    p = p.reshape(n, k, hh, ww)
    xs = np.arange(ww) / max(ww - 1, 1)
    ys = np.arange(hh) / max(hh - 1, 1)
    ex = (p.sum(axis=2) * xs).sum(axis=-1)  # marginalize rows → E[x]
    ey = (p.sum(axis=3) * ys).sum(axis=-1)  # marginalize cols → E[y]
    return np.stack([ex, ey], axis=-1).astype(np.float32)  # [N,21,2]


def build_heatmap_net(model_cfg: dict) -> HandHeatmapNet:
    """Construct the heatmap FCN from a config's `model` block (img_size / in_channels /
    conv_width / heatmap_size), with weights allocated by a dummy forward pass."""
    img_size = int(model_cfg.get("img_size", 128))
    in_channels = int(model_cfg.get("in_channels", 3))
    width = int(model_cfg.get("conv_width", model_cfg.get("d_model", 128)))
    heatmap_size = int(model_cfg.get("heatmap_size", img_size // 4))
    net = HandHeatmapNet(img_size, in_channels, width, heatmap_size)
    _ = net(ops.array(np.zeros((1, in_channels, img_size, img_size), dtype=np.float32)))
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


def _mpjpe(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean per-joint position error: mean over keypoints (and samples) of the Euclidean
    distance between predicted and target coordinates. [N,21,D] → scalar (lower = better)."""
    d = pred - target
    return float(np.mean(np.sqrt((d**2).sum(axis=-1) + 1e-9)))


def build_spec(cfg: dict):
    """The hands_recognition ModelSpec — how the framework builds/trains/evaluates it.
    Lower `mpjpe` (mean keypoint error) is better, matching the trainer's ratchet.

    Two arches behind one spec: `model.arch == "heatmap"` → the FCN trained on real
    FreiHAND (localized, 3D); anything else → the synthetic MLP (no download, CI demo)."""
    from src.plugins import ModelSpec

    mcfg = cfg.get("model", {}) or {}
    is_heatmap = mcfg.get("arch") == "heatmap"
    dims = int(mcfg.get("dims", 3))  # 3 = include root-relative depth in the metric
    depth_weight = float(mcfg.get("depth_weight", 0.1))  # balances depth MSE vs heatmap MSE

    def _make_real_loader(cfg: dict, split: str):
        from models.hands_recognition.data_freihand import FreiHandLoader

        mcfg = cfg.get("model", {}) or {}
        dcfg = cfg.get("dataset", {}) or {}
        tcfg = cfg.get("training", {}) or {}
        img_size = int(mcfg.get("img_size", 128))
        return FreiHandLoader(
            root=dcfg["root"],
            batch_size=int(tcfg.get("batch_size", 32)),
            img_size=img_size,
            in_channels=int(mcfg.get("in_channels", 3)),
            split=split,
            seed=int(tcfg.get("seed", 0)),
            augment=bool(dcfg.get("augment", False)) and split == "train",
            heatmap_size=int(mcfg.get("heatmap_size", img_size // 4)),
            localize=bool(dcfg.get("localize", True)),
        )

    def build_model(stage: int, cfg: dict, root):
        mcfg = cfg.get("model", {}) or {}
        seed = int((cfg.get("training", {}) or {}).get("seed", 0))
        B.engine.set_seed(seed)
        precision = (cfg.get("training", {}) or {}).get("precision", "fp32")
        # arch="heatmap" → real-hand FCN; otherwise the synthetic MLP (default, no download).
        net = (
            build_heatmap_net(mcfg) if is_heatmap else build_pose_net(int(mcfg.get("d_model", 256)))
        )
        return net, net.cfg, None, precision, seed

    def build_loader(stage: int, cfg: dict):
        # Heatmap arch trains on the real FreiHAND loader; else the synthetic stream.
        if is_heatmap:
            return _make_real_loader(cfg, "train")
        bs = int((cfg.get("training", {}) or {}).get("batch_size", 32))
        return _SynthLoader(bs, seed=int((cfg.get("training", {}) or {}).get("seed", 0)))

    def objective(model, batch):
        if is_heatmap:
            imgs, heatmaps_t, z_t, _kpts = batch
            pred_hm, pred_z = model(ops.array(imgs))
            loss = _mse(pred_hm, ops.array(heatmaps_t))
            return loss + depth_weight * _mse(pred_z, ops.array(z_t))
        frames, keypts = batch
        return _mse(model(frames), keypts)

    def _eval_heatmap(model, cfg):
        # Honest gate: a HELD-OUT, localized val split (re-read each eval; infrequent + cheap).
        vloader = _make_real_loader(cfg, "val")
        errs = []
        for _ in range(min(8, max(1, vloader.num_batches()))):
            imgs, _hm, z_t, kpts = vloader.next_batch()
            pred_hm, pred_z = model(ops.array(imgs))
            coords = soft_argmax(np.asarray(ops.to_numpy(pred_hm)))  # [N,21,2] in [0,1]
            if dims == 3:
                z_pred = np.asarray(ops.to_numpy(pred_z))[..., None]  # [N,21,1]
                pred = np.concatenate([coords, z_pred], axis=-1)
                tgt = np.concatenate([kpts, z_t[..., None]], axis=-1)
            else:
                pred, tgt = coords, kpts
            errs.append(_mpjpe(pred, tgt))
        return float(np.mean(errs)) if errs else float("inf")

    def evaluate(model, stage, val_batches=None, cfg=None, log=print, step=None):
        B.engine.set_eval(model)
        cfg = cfg or {}
        if is_heatmap:
            score = _eval_heatmap(model, cfg)
        else:
            # The trainer hands raw numpy val batches (synthetic path) — wrap for the metric.
            errs = [
                mean_keypoint_error(model(ops.array(f)), ops.array(k))
                for f, k in (val_batches or [])
            ]
            score = float(np.mean(errs)) if errs else float("inf")
        B.engine.set_train(model)
        threshold = float((cfg.get("gate", {}) or {}).get("max_mpjpe", 0.05))
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
