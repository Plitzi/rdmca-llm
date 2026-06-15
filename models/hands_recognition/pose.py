"""hands_recognition — multi-hand pose + behavioral heads + their ModelSpec.

A compact vision model that proves the framework is task-agnostic: NOT a transformer and
NOT text. It recovers the 21 standard hand landmarks (wrist + four joints per finger); with
`HAND_CONNECTIONS` (the bones/phalanges) those form an articulated hand SKELETON the camera
overlay draws. Two architectures behind one ModelSpec:

  • HandPoseNet — a tiny MLP on a downscaled grayscale frame, trained on SYNTHETIC data
    (a blob + a fixed landmark constellation). No download; the no-dataset CI/demo fallback.
    Predicts 21×2 of a single hand that fills the frame.
  • HandHeatmapNet — a fully-convolutional MULTI-HAND encoder-decoder: per slot (up to
    `n_hands`) it emits 21 spatial HEATMAPS + a per-keypoint DEPTH, plus a per-slot PRESENCE
    logit. Soft-argmax localizes each hand ANYWHERE in the frame (the VR foundation: both
    hands at once). Trained on real FreiHAND (see data_freihand.py) with location + multi-hand
    augmentation. The real path, via the model's level configs (`model.arch: heatmap`).

Three stages share the heatmap backbone (the framework's frozen-core + behavioral pattern):
stage 1 trains the multi-hand detector (frozen base); stage 2 adds `HandStateHead`
(handedness + finger extended/curled) and stage 3 `GestureHead` (e.g. thumbs-up) as behavioral
heads on the FROZEN backbone. `build_model` attaches/freezes per stage; `objective` dispatches
on `model._active_stage`; `evaluate` on the stage. Built on the shared backend (`src.backend`)
→ MLX or torch. The camera (uses/camera) rebuilds whichever arch + heads the checkpoint trained.
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


N_HANDS = 2  # hand SLOTS the detector predicts at once (VR: both hands). Slots are POSITIONAL
# (left-most hand → slot 0); handedness (which is the left/right hand) is the stage-2 head.


class HandHeatmapNet(nn.Module):
    """Fully-convolutional MULTI-HAND pose for REAL, LOCALIZED hands: an RGB (or gray) image
    [N, C, H, W] in [0,1] → `n_hands` slots, each with 21 spatial heatmaps + a per-keypoint
    depth (z relative to that hand's wrist), plus a per-slot PRESENCE logit (is a hand in this
    slot?). Soft-argmax over each heatmap recovers an (x,y) ANYWHERE in the frame, so it
    localizes up to `n_hands` hands wherever they are — the foundation for VR (both hands).

    Outputs: heatmaps [N, n_hands·21, hs, hs], depth [N, n_hands·21], presence [N, n_hands].
    Slots are POSITIONAL (training assigns the left-most hand to slot 0); the left/right
    HANDEDNESS of each slot is a separate behavioral head (stage 2).

    Encoder: four stride-2 conv blocks (H,W shrink 16× → bottleneck img_size/16). Decoder:
    two ConvTranspose2d up-samples (→ img_size/4 = heatmap_size) then a 1×1 conv to n_hands·21
    channels. Depth + presence branches: global-average-pooled bottleneck → MLP. Requires
    `heatmap_size == img_size // 4` (the two up-samples from the /16 bottleneck)."""

    def __init__(
        self,
        img_size: int = 128,
        in_channels: int = 3,
        width: int = 128,
        heatmap_size: int = 32,
        n_hands: int = N_HANDS,
    ):
        super().__init__()
        if heatmap_size != img_size // 4:
            raise ValueError(
                f"heatmap_size must be img_size//4 ({img_size // 4}), got {heatmap_size} "
                f"(the decoder up-samples the /16 bottleneck twice)."
            )
        out_kp = n_hands * N_KEYPOINTS  # heatmap + depth channels span every slot's keypoints
        self.c1 = nn.Conv2d(in_channels, width // 8, 3, stride=2, padding=1)  # H/2
        self.c2 = nn.Conv2d(width // 8, width // 4, 3, stride=2, padding=1)  # H/4
        self.c3 = nn.Conv2d(width // 4, width // 2, 3, stride=2, padding=1)  # H/8
        self.c4 = nn.Conv2d(width // 2, width, 3, stride=2, padding=1)  # H/16 (bottleneck)
        self.d1 = nn.ConvTranspose2d(width, width // 2, 4, stride=2, padding=1)  # H/8
        self.d2 = nn.ConvTranspose2d(width // 2, width // 4, 4, stride=2, padding=1)  # H/4
        self.heatmap = nn.Conv2d(width // 4, out_kp, 1)  # → n_hands·21 heatmaps at H/4
        self.depth1 = nn.Linear(width, width)
        self.depth2 = nn.Linear(width, out_kp)  # n_hands·21 root-relative depths
        self.presence = nn.Linear(width, n_hands)  # per-slot "a hand is here" logit
        # Record geometry so the audit captures it and the camera rebuilds the EXACT net.
        self.cfg = SimpleNamespace(
            arch="heatmap",
            img_size=img_size,
            in_channels=in_channels,
            d_model=width,
            heatmap_size=heatmap_size,
            dims=3,
            n_hands=n_hands,
            n_layers=6,
            context_len=img_size,
        )

    def __call__(
        self, x
    ):  # x: [N,C,H,W] → (heatmaps [N,n_hands·21,hs,hs], z, presence [N,n_hands])
        h = ops.relu(self.c1(x))
        h = ops.relu(self.c2(h))
        h = ops.relu(self.c3(h))
        bottleneck = ops.relu(self.c4(h))  # [N, width, H/16, W/16]
        u = ops.relu(self.d1(bottleneck))
        u = ops.relu(self.d2(u))
        heatmaps = self.heatmap(u)  # [N, n_hands·21, hs, hs] (raw — softmax in soft-argmax)
        pooled = ops.mean(bottleneck, axis=(2, 3))  # global average pool → [N, width]
        z = self.depth2(ops.relu(self.depth1(pooled)))  # [N, n_hands·21] root-relative depth
        presence = self.presence(pooled)  # [N, n_hands] presence logits
        return heatmaps, z, presence

    def count_params(self, include_sectors: bool = True) -> int:
        w, cin, n = self.cfg.d_model, self.cfg.in_channels, self.cfg.n_hands
        kp = n * N_KEYPOINTS

        def conv(i, o, ksz):
            return i * o * ksz * ksz + o

        return (
            conv(cin, w // 8, 3)
            + conv(w // 8, w // 4, 3)
            + conv(w // 4, w // 2, 3)
            + conv(w // 2, w, 3)
            + conv(w, w // 2, 4)  # ConvTranspose2d (same weight count)
            + conv(w // 2, w // 4, 4)
            + conv(w // 4, kp, 1)
            + (w * w + w)
            + (w * kp + kp)
            + (w * n + n)  # presence head
        )


def soft_argmax(heatmaps_np: np.ndarray) -> np.ndarray:
    """[N,K,hs,hs] raw heatmaps → [N,K,2] (x,y) in [0,1] via 2D spatial soft-argmax (K spans
    every slot's keypoints, n_hands·21). Inference/eval only (no gradient), so it runs in
    numpy: a spatial softmax over each heatmap then the expected grid coordinate. Localizes
    each peak anywhere in the frame."""
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
    return np.stack([ex, ey], axis=-1).astype(np.float32)  # [N,K,2]


def predict_hands(net, imgs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the multi-hand backbone and decode every slot in numpy (inference/eval only):
    returns (kpts [N, n_hands, 21, 2] in [0,1], z [N, n_hands, 21], presence [N, n_hands] in
    [0,1]). Soft-argmax localizes each slot's keypoints; the depth/presence branches give z and
    the sigmoid presence. Frozen-backbone safe (no gradient — numpy)."""
    n_hands = net.cfg.n_hands
    heatmaps, z, presence = net(ops.array(np.asarray(imgs, dtype=np.float32)))
    hm = np.asarray(ops.to_numpy(heatmaps))  # [N, n_hands·21, hs, hs]
    coords = soft_argmax(hm)  # [N, n_hands·21, 2]
    n = coords.shape[0]
    kpts = coords.reshape(n, n_hands, N_KEYPOINTS, 2)
    zz = np.asarray(ops.to_numpy(z)).reshape(n, n_hands, N_KEYPOINTS)
    pres = 1.0 / (1.0 + np.exp(-np.asarray(ops.to_numpy(presence))))  # sigmoid → [N, n_hands]
    return kpts.astype(np.float32), zz.astype(np.float32), pres.astype(np.float32)


def build_heatmap_net(model_cfg: dict) -> HandHeatmapNet:
    """Construct the multi-hand heatmap FCN from a config's `model` block (img_size /
    in_channels / conv_width / heatmap_size / n_hands), weights allocated by a dummy forward."""
    img_size = int(model_cfg.get("img_size", 128))
    in_channels = int(model_cfg.get("in_channels", 3))
    width = int(model_cfg.get("conv_width", model_cfg.get("d_model", 128)))
    heatmap_size = int(model_cfg.get("heatmap_size", img_size // 4))
    n_hands = int(model_cfg.get("n_hands", N_HANDS))
    net = HandHeatmapNet(img_size, in_channels, width, heatmap_size, n_hands)
    _ = net(ops.array(np.zeros((1, in_channels, img_size, img_size), dtype=np.float32)))
    B.engine.eval(net.parameters())
    return net


_STATE_IN = N_KEYPOINTS * 3  # a slot's 21 keypoints × (x, y, z) feed the behavioral heads


class HandStateHead(nn.Module):
    """Behavioral head (stage 2): from ONE hand's 21×3 keypoints → handedness (2 logits:
    right/left) + per-finger extended/curled (5 logits). A small MLP on the FROZEN backbone's
    predicted keypoints — so it reads the hand's articulation, not the raw pixels."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(_STATE_IN, hidden)
        self.handed = nn.Linear(hidden, 2)  # right (0) / left (1)
        self.finger = nn.Linear(hidden, 5)  # per-finger extended/curled logit

    def __call__(self, x):  # x: [M, 63] → ([M,2] handedness logits, [M,5] finger logits)
        h = ops.relu(self.fc1(x))
        return self.handed(h), self.finger(h)


def build_state_head(hidden: int = 64) -> HandStateHead:
    head = HandStateHead(hidden)
    _ = head(ops.array(np.zeros((1, _STATE_IN), dtype=np.float32)))  # materialize params
    B.engine.eval(head.parameters())
    return head


class GestureHead(nn.Module):
    """Behavioral head (stage 3): from ONE hand's 21×3 keypoints → gesture-class logits
    (e.g. thumbs-up). A small MLP on the FROZEN backbone's predicted keypoints; the gesture
    vocabulary size is set per config/dataset."""

    def __init__(self, n_gestures: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(_STATE_IN, hidden)
        self.out = nn.Linear(hidden, n_gestures)

    def __call__(self, x):  # x: [M, 63] → [M, n_gestures] logits
        return self.out(ops.relu(self.fc1(x)))


def build_gesture_head(n_gestures: int, hidden: int = 64) -> GestureHead:
    head = GestureHead(n_gestures, hidden)
    _ = head(ops.array(np.zeros((1, _STATE_IN), dtype=np.float32)))  # materialize params
    B.engine.eval(head.parameters())
    return head


def hand_features(net, imgs) -> tuple[np.ndarray, np.ndarray]:
    """Frozen-backbone slot features for the behavioral heads: returns (feats [N·n_hands, 63]
    of each slot's predicted 21×3 keypoints, presence [N·n_hands] in [0,1]). Numpy (no grad —
    the backbone is frozen); the head trains on these as constant inputs."""
    kpts, z, presence = predict_hands(net, imgs)  # [N,nh,21,2], [N,nh,21], [N,nh]
    n, nh = kpts.shape[:2]
    feats = np.concatenate([kpts, z[..., None]], axis=-1).reshape(n * nh, N_KEYPOINTS * 3)
    return feats.astype(np.float32), presence.reshape(n * nh)


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
    Lower scores are better (mpjpe for stage 1; 1−accuracy for the behavioral stages),
    matching the trainer's ratchet.

    Two arches behind one spec: `model.arch == "heatmap"` → the MULTI-HAND FCN trained on real
    FreiHAND (localized, 3D); anything else → the synthetic MLP (no download, CI demo). For the
    heatmap arch the stages share ONE backbone: stage 1 trains the multi-hand detector (frozen
    base); stages 2/3 attach a behavioral head on the frozen backbone. `objective` dispatches
    on `model._active_stage` (set by build_model); `evaluate` gets the stage directly."""
    from src.plugins import ModelSpec

    mcfg = cfg.get("model", {}) or {}
    is_heatmap = mcfg.get("arch") == "heatmap"
    dims = int(mcfg.get("dims", 3))  # 3 = include root-relative depth in the metric
    depth_weight = float(mcfg.get("depth_weight", 0.1))  # balances depth MSE vs heatmap MSE
    presence_weight = float(mcfg.get("presence_weight", 0.5))  # per-slot presence BCE weight

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
            n_hands=int(mcfg.get("n_hands", N_HANDS)),
        )

    def _n_gestures():
        from models.hands_recognition.data_gestures import n_gestures

        return int(mcfg.get("n_gestures", n_gestures()))

    def _load_prev_stage(net, root, stage):
        """Load the PRIOR stage's weights into `net` (strict=False, so heads not yet present
        keep their fresh init). Stage 2 loads the frozen stage-1 core; stage ≥ 3 loads the
        previous stage's checkpoint (carrying the backbone + earlier heads forward)."""
        if root is None:
            return
        from pathlib import Path

        from src.training.checkpoint import resolve_stage_checkpoint

        if stage == 2:
            frozen = Path(root) / "foundational" / "theta_f_frozen.npz"
            if frozen.exists():
                B.engine.load_weights(net, str(frozen))
                return
            path, _l, _m = resolve_stage_checkpoint(Path(root) / "stage1")
        else:
            path, _l, _m = resolve_stage_checkpoint(Path(root) / f"stage{stage - 1}")
        if path:
            B.engine.load_weights(net, str(path))

    def build_model(stage: int, cfg: dict, root):
        mcfg = cfg.get("model", {}) or {}
        seed = int((cfg.get("training", {}) or {}).get("seed", 0))
        B.engine.set_seed(seed)
        precision = (cfg.get("training", {}) or {}).get("precision", "fp32")
        if not is_heatmap:  # synthetic MLP (no download): single stage, full model trainable
            net = build_pose_net(int(mcfg.get("d_model", 256)))
            net._active_stage = stage
            return net, net.cfg, None, precision, seed
        # Multi-hand FCN. Stage 1 trains the whole detector; stages 2/3 attach a behavioral
        # head on the FROZEN backbone (the same frozen-core + sector pattern). Earlier heads are
        # carried forward (attached + loaded) so a later checkpoint keeps every capability.
        net = build_heatmap_net(mcfg)
        if stage >= 2:
            net.state_head = build_state_head(int(mcfg.get("state_hidden", 64)))
        if stage >= 3:
            net.gesture_head = build_gesture_head(
                _n_gestures(), int(mcfg.get("gesture_hidden", 64))
            )
        if stage >= 2:
            _load_prev_stage(net, root, stage)
            # Freeze everything; train ONLY the active stage's head.
            active_head = net.gesture_head if stage >= 3 else net.state_head
            B.engine.set_trainable(net, [active_head])
        net._active_stage = stage  # dispatch: 1 = detector, 2 = hand state, 3 = gesture
        return net, net.cfg, None, precision, seed

    def _make_gesture_loader(cfg: dict, split: str):
        from models.hands_recognition.data_gestures import GestureLoader

        mcfg = cfg.get("model", {}) or {}
        dcfg = cfg.get("dataset", {}) or {}
        tcfg = cfg.get("training", {}) or {}
        return GestureLoader(
            root=dcfg.get("gesture_root", "models/hands_recognition/data/gestures"),
            batch_size=int(tcfg.get("batch_size", 32)),
            img_size=int(mcfg.get("img_size", 128)),
            in_channels=int(mcfg.get("in_channels", 3)),
            split=split,
            seed=int(tcfg.get("seed", 0)),
        )

    def build_loader(stage: int, cfg: dict):
        if not is_heatmap:  # synthetic stream (no download)
            bs = int((cfg.get("training", {}) or {}).get("batch_size", 32))
            return _SynthLoader(bs, seed=int((cfg.get("training", {}) or {}).get("seed", 0)))
        if stage >= 3:  # gesture classification trains on the labelled gesture dataset
            return _make_gesture_loader(cfg, "train")
        return _make_real_loader(cfg, "train")  # stages 1-2 train on FreiHAND

    def _objective_detector(model, batch):
        """Stage 1: multi-hand detector — heatmap MSE + depth MSE + per-slot presence BCE."""
        imgs, heatmaps_t, z_t, presence_t, *_ = batch
        pred_hm, pred_z, pred_pres = model(ops.array(imgs))
        loss = _mse(pred_hm, ops.array(heatmaps_t)) + depth_weight * _mse(pred_z, ops.array(z_t))
        return loss + presence_weight * ops.bce_with_logits(pred_pres, ops.array(presence_t))

    def _objective_handstate(model, batch):
        """Stage 2: handedness + finger-state head on the FROZEN backbone's predicted
        keypoints. Supervised on the GROUND-TRUTH present slots (CE handedness + BCE finger)."""
        imgs, _hm, _z, presence_t, _kpts, hand_t, finger_t = batch
        feats, _pres = hand_features(model, imgs)  # [N·nh, 63] predicted-keypoint features
        mask = presence_t.reshape(-1).astype(bool)
        if not mask.any():
            return ops.array(0.0)  # no labeled hand in this batch (rare) — nothing to learn
        hl, fl = model.state_head(ops.array(feats[mask]))
        hand_loss = ops.cross_entropy(hl, ops.array(hand_t.reshape(-1)[mask].astype(np.int64)))
        fing_loss = ops.bce_with_logits(fl, ops.array(finger_t.reshape(-1, 5)[mask]))
        return hand_loss + fing_loss

    def _dominant_feats(model, imgs):
        """Per-sample features of the MOST-present hand slot (for the gesture head): runs the
        frozen backbone, picks each sample's highest-presence slot → [N, 63] numpy."""
        feats, pres = hand_features(model, imgs)  # [N·nh, 63], [N·nh]
        n = imgs.shape[0]
        nh = pres.shape[0] // n
        dom = pres.reshape(n, nh).argmax(axis=1)  # [N] dominant slot per sample
        return feats.reshape(n, nh, -1)[np.arange(n), dom]  # [N, 63]

    def _objective_gesture(model, batch):
        """Stage 3: gesture classification on the FROZEN backbone's dominant-hand features."""
        imgs, labels = batch
        logits = model.gesture_head(ops.array(_dominant_feats(model, imgs)))
        return ops.cross_entropy(logits, ops.array(np.asarray(labels, dtype=np.int64)))

    def objective(model, batch):
        if not is_heatmap:
            frames, keypts = batch
            return _mse(model(frames), keypts)
        stage = getattr(model, "_active_stage", 1)
        if stage >= 3:
            return _objective_gesture(model, batch)
        if stage == 2:
            return _objective_handstate(model, batch)
        return _objective_detector(model, batch)

    def _eval_detector(model, cfg):
        # Honest gate: a HELD-OUT, localized val split (re-read each eval; infrequent + cheap).
        # mpjpe over the PRESENT slots only (ground-truth presence selects which to score).
        vloader = _make_real_loader(cfg, "val")
        n_hands = int(mcfg.get("n_hands", N_HANDS))
        errs = []
        for _ in range(min(8, max(1, vloader.num_batches()))):
            imgs, _hm, z_t, presence_t, kpts_t, *_ = vloader.next_batch()
            pk, pz, _pp = predict_hands(model, imgs)  # [N,nh,21,2], [N,nh,21], [N,nh]
            n = imgs.shape[0]
            if dims == 3:
                pred = np.concatenate([pk, pz[..., None]], axis=-1)  # [N,nh,21,3]
                z_slots = z_t.reshape(n, n_hands, N_KEYPOINTS)[..., None]
                tgt = np.concatenate([kpts_t, z_slots], axis=-1)
            else:
                pred, tgt = pk, kpts_t
            mask = presence_t.astype(bool)  # [N,nh]
            if mask.any():
                errs.append(_mpjpe(pred[mask], tgt[mask]))  # [M,21,D] over present slots
        return float(np.mean(errs)) if errs else float("inf")

    def _eval_handstate(model, cfg):
        # Accuracy of handedness + finger-state over the held-out, localized val slots.
        vloader = _make_real_loader(cfg, "val")
        hand_ok = hand_tot = fing_ok = fing_tot = 0
        for _ in range(min(8, max(1, vloader.num_batches()))):
            imgs, _hm, _z, presence_t, _kpts, hand_t, finger_t = vloader.next_batch()
            feats, _pres = hand_features(model, imgs)
            mask = presence_t.reshape(-1).astype(bool)
            if not mask.any():
                continue
            hl, fl = model.state_head(ops.array(feats[mask]))
            hpred = np.asarray(ops.to_numpy(hl)).argmax(axis=-1)
            fpred = (np.asarray(ops.to_numpy(fl)) > 0).astype(np.float32)
            hand_ok += int((hpred == hand_t.reshape(-1)[mask]).sum())
            hand_tot += int(mask.sum())
            fing_ok += int((fpred == finger_t.reshape(-1, 5)[mask]).sum())
            fing_tot += int(mask.sum()) * 5
        acc = 0.5 * (hand_ok / max(hand_tot, 1) + fing_ok / max(fing_tot, 1))
        return 1.0 - acc  # lower is better (matches the ratchet)

    def _eval_gesture(model, cfg):
        # Gesture classification accuracy over the held-out gesture val split → 1 − accuracy.
        vloader = _make_gesture_loader(cfg, "val")
        correct = total = 0
        for _ in range(min(8, max(1, vloader.num_batches()))):
            imgs, labels = vloader.next_batch()
            logits = model.gesture_head(ops.array(_dominant_feats(model, imgs)))
            pred = np.asarray(ops.to_numpy(logits)).argmax(axis=-1)
            correct += int((pred == labels).sum())
            total += len(labels)
        return 1.0 - correct / max(total, 1)

    def evaluate(model, stage, val_batches=None, cfg=None, log=print, step=None):
        B.engine.set_eval(model)
        cfg = cfg or {}
        gate = cfg.get("gate", {}) or {}
        if not is_heatmap:
            errs = [
                mean_keypoint_error(model(ops.array(f)), ops.array(k))
                for f, k in (val_batches or [])
            ]
            score = float(np.mean(errs)) if errs else float("inf")
            metric, threshold = "mpjpe", float(gate.get("max_mpjpe", 0.05))
        elif stage >= 3:
            score = _eval_gesture(model, cfg)
            metric, threshold = "gesture_err", float(gate.get("max_gesture_err", 0.3))
        elif stage == 2:
            score = _eval_handstate(model, cfg)
            metric, threshold = "handstate_err", float(gate.get("max_handstate_err", 0.3))
        else:
            score = _eval_detector(model, cfg)
            metric, threshold = "mpjpe", float(gate.get("max_mpjpe", 0.1))
        B.engine.set_train(model)
        passed = score <= threshold
        tag = f"step={step:,} | " if step is not None else ""
        log(
            f"[gate] {tag}{metric}={score:.4f} <= {threshold:.4f} -> {'PASS' if passed else 'fail'}"
        )
        return score, passed

    return ModelSpec(
        name="hand-pose",
        build_model=build_model,
        build_loader=build_loader,
        objective=objective,
        evaluate=evaluate,
        gate_metric="mpjpe",
    )
