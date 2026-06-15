"""FreiHAND real-hand data loader for hands_recognition.

FreiHAND ships real RGB hand images (one centered hand) with 3D keypoints + camera
intrinsics. We project the 3D points to 2D with `uv = K · xyz` and normalize to [0, 1] of
the image, giving the same 21×2 target the synthetic generator produces — so the trainer,
objective and metric are unchanged; only the data is real. Images are loaded and resized on
the fly (the dataset is large), so memory stays flat.

Layout expected under `root/` (the standard FreiHAND_pub_v2 unzip):
    training_xyz.json   list of M× [21,3] 3D keypoints (camera frame)
    training_K.json     list of M× [3,3] camera intrinsics
    training/rgb/*.jpg   the images (M or 4·M — green-screen augmentations repeat xyz)

This loader mirrors the synthetic `_SynthLoader` interface so the ModelSpec can swap it in
transparently (next_batch + telemetry + no-op skip/index).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from models.hands_recognition.pose import N_KEYPOINTS

_SPLIT_SEED = 12345  # FIXED so independently-built train/val loaders agree on the partition

# Public FreiHAND release (real RGB hands + 3D keypoints + camera intrinsics).
_FREIHAND_URL = "https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip"
# Annotation file whose presence means "already prepared" — the download is idempotent.
_READY_MARKER = "training_xyz.json"


def is_prepared(root: str | Path) -> bool:
    """True when FreiHAND is already extracted under `root` (the marker annotation +
    its images exist), so prepare/download can skip the multi-GB fetch."""
    root = Path(root)
    return (root / _READY_MARKER).exists() and (root / "training" / "rgb").is_dir()


def download_freihand(root: str | Path, *, url: str = _FREIHAND_URL) -> Path:
    """Download + extract the FreiHAND dataset into `root` (idempotent, resumable).

    Mirrors how cognition prepares its corpus: this is the data step `rdmca prepare`
    runs for hands_recognition (via the model's `prepare_stage` hook), NOT an ad-hoc
    curl. The ~4 GB zip is streamed to `root/FreiHAND_pub_v2.zip.part` with HTTP-Range
    resume; on a clean finish it's renamed and extracted, then deleted. Re-running once
    `training_xyz.json` exists is a no-op.
    """
    import zipfile

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if is_prepared(root):
        print(f"  FreiHAND already prepared in {root} — skipping download.")
        return root

    zip_path = root / "FreiHAND_pub_v2.zip"
    part_path = root / "FreiHAND_pub_v2.zip.part"
    if not zip_path.exists():
        _download_resumable(url, part_path)
        part_path.rename(zip_path)

    print(f"  Extracting {zip_path.name} → {root} …")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    # Some mirrors wrap everything in a top FreiHAND_pub_v2/ dir; flatten so the loader's
    # `root/training_xyz.json` layout holds either way.
    if not (root / _READY_MARKER).exists():
        nested = next((d for d in root.iterdir() if (d / _READY_MARKER).exists()), None)
        if nested is not None:
            for item in nested.iterdir():
                item.rename(root / item.name)
            nested.rmdir()
    if not is_prepared(root):
        raise RuntimeError(
            f"FreiHAND extraction did not yield {_READY_MARKER} + training/rgb under {root}."
        )
    zip_path.unlink()
    print(f"  FreiHAND ready in {root}.")
    return root


def _download_resumable(url: str, part_path: Path) -> None:
    """Stream `url` to `part_path`, resuming from however many bytes are already there
    (HTTP Range). Prints coarse progress; raises on an incomplete transfer."""
    import urllib.request

    have = part_path.stat().st_size if part_path.exists() else 0
    req = urllib.request.Request(url)
    if have:
        req.add_header("Range", f"bytes={have}-")
        print(f"  Resuming download at {have / 1e6:.0f} MB …")
    else:
        print(f"  Downloading FreiHAND (~4 GB) from {url} …")

    with urllib.request.urlopen(req) as resp:
        # total = bytes remaining (Range) + what we already have.
        remaining = int(resp.headers.get("Content-Length", 0))
        total = remaining + have
        mode = "ab" if have and resp.status == 206 else "wb"
        if mode == "wb":
            have = 0  # server ignored Range → restart cleanly
        done = have
        with open(part_path, mode) as out:
            while chunk := resp.read(1 << 20):  # 1 MB
                out.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e6:7.0f} / {total / 1e6:.0f} MB", end="", flush=True)
        print()
    if total and part_path.stat().st_size < total:
        raise RuntimeError("FreiHAND download incomplete — re-run to resume.")


# Per-finger (pip, tip) landmark indices — used to derive the extended/curled state of each
# finger from the 3D keypoints (a finger is "extended" when its tip is farther from the wrist
# than its middle joint). Order: thumb, index, middle, ring, pinky (matches the 5 head logits).
_FINGER_PIP_TIP = ((3, 4), (6, 8), (10, 12), (14, 16), (18, 20))


def finger_states(xyz: np.ndarray) -> np.ndarray:
    """[21,3] hand keypoints → [5] in {0,1}: per finger, 1 if EXTENDED (tip farther from the
    wrist than the mid joint), else 0 (curled). Wrist = landmark 0. Mirror-invariant geometry,
    so it's the free finger-state label for the stage-2 head."""
    wrist = xyz[0]
    out = np.zeros(5, dtype=np.float32)
    for i, (pip, tip) in enumerate(_FINGER_PIP_TIP):
        d_tip = np.linalg.norm(xyz[tip] - wrist)
        d_pip = np.linalg.norm(xyz[pip] - wrist)
        out[i] = 1.0 if d_tip > d_pip else 0.0
    return out


class FreiHandLoader:
    """Yields MULTI-HAND heatmap-supervision batches from a deterministic train/val split of
    FreiHAND for HandHeatmapNet. Each sample composites up to `n_hands` real hands onto a random
    background at random positions/scales (some possibly MIRRORED → left hands), then assigns
    them to POSITIONAL slots (left-most hand → slot 0). Returns:

        (images     [N, C, H, W]          in [0,1],
         heatmaps   [N, n_hands·21, hs, hs] Gaussian targets per slot (zeros for absent slots),
         z          [N, n_hands·21]         root-relative, scale-normalized depth per slot,
         presence   [N, n_hands]            1 if a hand occupies the slot, else 0,
         kpts       [N, n_hands, 21, 2]     (x,y) in [0,1] — ground truth for the mpjpe metric,
         handedness [N, n_hands]            0=right, 1=left (from the mirror flag),
         finger     [N, n_hands, 5]         per-finger extended/curled — labels for stage 2)

    Localization (always on for multi-hand) is what lets the FCN find hands ANYWHERE in the
    frame — the foundation for VR. Depth z is each keypoint's camera-z minus its wrist's,
    divided by the wrist→middle-MCP bone length (scale-invariant). Mirrors the synthetic
    loader's telemetry + no-op skip/index surface."""

    def __init__(
        self,
        root: str,
        batch_size: int,
        img_size: int = 128,
        in_channels: int = 3,
        split: str = "train",
        seed: int = 0,
        augment: bool = False,
        val_fraction: float = 0.05,
        heatmap_size: int = 32,
        localize: bool = True,
        sigma: float = 1.5,
        localize_scale: tuple[float, float] = (0.35, 0.6),
        n_hands: int = 2,
    ):
        self.root = Path(root)
        self.batch_size = batch_size
        self.img_size = img_size
        self.in_channels = in_channels
        self.augment = augment
        self.heatmap_size = heatmap_size
        self.localize = localize
        self.sigma = sigma
        self.localize_scale = localize_scale
        self.n_hands = n_hands
        self._rng = np.random.default_rng(seed)
        # Telemetry the trainer reads (no replay/corpus concepts for on-the-fly vision data).
        self.passes = 0
        self.last_was_replay = False
        self.replay_fraction = 0.0

        xyz_path, k_path = self.root / "training_xyz.json", self.root / "training_K.json"
        if not xyz_path.exists() or not k_path.exists():
            raise FileNotFoundError(
                f"FreiHAND annotations not found in {self.root} "
                f"(need training_xyz.json + training_K.json). See the GUIDE for the download."
            )
        self._xyz = np.asarray(json.loads(xyz_path.read_text()), dtype=np.float32)  # [M,21,3]
        self._k = np.asarray(json.loads(k_path.read_text()), dtype=np.float32)  # [M,3,3]
        n_anno = len(self._xyz)
        # Root-relative, scale-normalized depth: (z - z_wrist) / ||wrist→middle_mcp||. Wrist=0,
        # middle-MCP=9. The bone-length divisor makes z invariant to hand size / distance.
        scale = np.linalg.norm(self._xyz[:, 9, :] - self._xyz[:, 0, :], axis=1)  # [M]
        self._z = ((self._xyz[:, :, 2] - self._xyz[:, 0:1, 2]) / (scale[:, None] + 1e-6)).astype(
            np.float32
        )  # [M,21]

        rgb_dir = self.root / "training" / "rgb"
        files = sorted(rgb_dir.glob("*.jpg")) or sorted(rgb_dir.glob("*.png"))
        if not files:
            raise FileNotFoundError(f"No images under {rgb_dir} (expected training/rgb/*.jpg).")
        self._files = files
        # Image p shares annotation p % n_anno (the 4 green-screen variants repeat the pose).
        self._anno_of = [p % n_anno for p in range(len(files))]

        # Deterministic, reproducible disjoint train/val split over image indices.
        idx = np.arange(len(files))
        np.random.default_rng(_SPLIT_SEED).shuffle(idx)
        n_val = max(1, int(len(idx) * val_fraction))
        self._indices = idx[:n_val] if split == "val" else idx[n_val:]
        self.epoch_tokens = len(self._indices)  # "tokens" = samples → corpus-pass cap works
        self._order = self._indices.copy()
        self._rng.shuffle(self._order)
        self._cursor = 0

    def num_batches(self) -> int:
        return max(1, len(self._indices) // self.batch_size)

    def _next_index(self) -> int:
        """The next image index in the shuffled epoch order (reshuffles at the boundary). A
        sample consumes one index per composited hand, so this centralizes the wraparound."""
        if self._cursor >= len(self._order):
            self._cursor = 0
            self.passes += 1
            self._rng.shuffle(self._order)
        idx = int(self._order[self._cursor])
        self._cursor += 1
        return idx

    def _gaussian_heatmaps(self, kpts: np.ndarray) -> np.ndarray:
        """kpts [21,2] in [0,1] → [21, hs, hs] target heatmaps, a Gaussian bump per keypoint
        (peak at the keypoint, std `sigma` on the heatmap grid). Off-canvas keypoints simply
        produce a near-empty map."""
        hs = self.heatmap_size
        grid = np.arange(hs, dtype=np.float32)
        yy, xx = np.meshgrid(grid, grid, indexing="ij")  # [hs,hs]
        gx = (kpts[:, 0] * (hs - 1))[:, None, None]  # [21,1,1]
        gy = (kpts[:, 1] * (hs - 1))[:, None, None]
        d2 = (xx[None] - gx) ** 2 + (yy[None] - gy) ** 2  # [21,hs,hs]
        return np.exp(-d2 / (2.0 * self.sigma**2)).astype(np.float32)

    def _place_one_hand(self, arr: np.ndarray):
        """Composite ONE real hand onto the canvas `arr` (in place) at a random position/scale,
        possibly MIRRORED (→ left hand). Returns (kpts [21,2] in canvas [0,1], z [21],
        handedness 0/1, finger_state [5], mean_x) — mean_x sets the positional slot order."""
        from PIL import Image

        size = self.img_size
        image_idx = self._next_index()
        anno = self._anno_of[image_idx]
        img = Image.open(self._files[image_idx]).convert("RGB")
        w0, h0 = img.size
        uv = (self._k[anno] @ self._xyz[anno].T).T  # [21,3]
        kpts = ((uv[:, :2] / uv[:, 2:3]) / np.array([w0, h0], np.float32)).astype(np.float32)
        z = self._z[anno].copy()
        finger = finger_states(self._xyz[anno])  # geometric, mirror-invariant
        mirror = self.localize and self._rng.random() < 0.5  # FreiHAND is right; mirror → left

        if self.localize:
            lo, hi = self.localize_scale
            stamp = max(8, int(size * self._rng.uniform(lo, hi)))
            ox = int(self._rng.integers(0, size - stamp + 1))
            oy = int(self._rng.integers(0, size - stamp + 1))
        else:
            stamp, ox, oy = size, 0, 0  # fills the frame
        stamp_arr = np.asarray(img.resize((stamp, stamp)), dtype=np.float32) / 255.0
        if mirror:
            stamp_arr = stamp_arr[:, ::-1, :].copy()
            kpts[:, 0] = 1.0 - kpts[:, 0]
        arr[oy : oy + stamp, ox : ox + stamp] = stamp_arr
        kpts = (np.array([ox, oy], np.float32) + kpts * stamp) / size  # → canvas [0,1]
        return kpts, z, (1 if mirror else 0), finger, float(kpts[:, 0].mean())

    def _n_present(self) -> int:
        """How many hands to composite this sample. Favors 1–n_hands, with a small chance of 0
        (an empty scene → teaches presence=0). Without localization, always a single hand."""
        if not self.localize:
            return 1
        if self.n_hands == 2:
            return int(self._rng.choice([0, 1, 2], p=[0.05, 0.35, 0.60]))
        return int(self._rng.integers(1, self.n_hands + 1))

    def _compose_sample(self):
        """Build one multi-hand training sample. Returns the per-slot targets (see class docs):
        (chw, heatmaps [n_hands·21,hs,hs], z [n_hands·21], presence [n_hands], kpts
        [n_hands,21,2], handedness [n_hands], finger [n_hands,5])."""
        size, hs, n_hands = self.img_size, self.heatmap_size, self.n_hands
        base = self._rng.uniform(0.0, 1.0, size=3).astype(np.float32)  # random background
        arr = np.clip(base + 0.1 * self._rng.standard_normal((size, size, 3)), 0.0, 1.0)
        hands = [self._place_one_hand(arr) for _ in range(self._n_present())]
        hands.sort(key=lambda h: h[4])  # POSITIONAL slots: left-most hand → slot 0

        if self.augment:
            arr = np.clip(arr * self._rng.uniform(0.7, 1.3), 0.0, 1.0)  # brightness jitter
        if self.in_channels == 1:
            arr = arr.mean(axis=2, keepdims=True)
        chw = np.transpose(arr, (2, 0, 1)).astype(np.float32)

        heatmaps = np.zeros((n_hands * N_KEYPOINTS, hs, hs), dtype=np.float32)
        z = np.zeros(n_hands * N_KEYPOINTS, dtype=np.float32)
        presence = np.zeros(n_hands, dtype=np.float32)
        kpts = np.zeros((n_hands, N_KEYPOINTS, 2), dtype=np.float32)
        handedness = np.zeros(n_hands, dtype=np.int64)
        finger = np.zeros((n_hands, 5), dtype=np.float32)
        for slot, (k, zz, hand, fing, _mx) in enumerate(hands[:n_hands]):
            s = slot * N_KEYPOINTS
            heatmaps[s : s + N_KEYPOINTS] = self._gaussian_heatmaps(k)
            z[s : s + N_KEYPOINTS] = zz
            presence[slot] = 1.0
            kpts[slot] = k
            handedness[slot] = hand
            finger[slot] = fing
        return chw, heatmaps, z, presence, kpts, handedness, finger

    def next_batch(self):
        n, c, size, hs, nh = (
            self.batch_size,
            self.in_channels,
            self.img_size,
            self.heatmap_size,
            self.n_hands,
        )
        imgs = np.zeros((n, c, size, size), dtype=np.float32)
        heatmaps = np.zeros((n, nh * N_KEYPOINTS, hs, hs), dtype=np.float32)
        z = np.zeros((n, nh * N_KEYPOINTS), dtype=np.float32)
        presence = np.zeros((n, nh), dtype=np.float32)
        kpts = np.zeros((n, nh, N_KEYPOINTS, 2), dtype=np.float32)
        handedness = np.zeros((n, nh), dtype=np.int64)
        finger = np.zeros((n, nh, 5), dtype=np.float32)
        for b in range(n):
            imgs[b], heatmaps[b], z[b], presence[b], kpts[b], handedness[b], finger[b] = (
                self._compose_sample()
            )
        return imgs, heatmaps, z, presence, kpts, handedness, finger

    # Vision data is generated/streamed on the fly: the text-stream skip/index is a no-op.
    def skip(self, n: int) -> int:
        self._cursor += n
        return n

    def save_skip_index(self, path) -> None:
        pass

    def load_skip_index(self, path) -> bool:
        return False
