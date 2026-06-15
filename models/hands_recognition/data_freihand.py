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

_OUT = N_KEYPOINTS * 2
_SPLIT_SEED = 12345  # FIXED so independently-built train/val loaders agree on the partition


class FreiHandLoader:
    """Yields (images [N, C, H, W] in [0,1], keypoints [N, 42] in [0,1]) batches from a
    deterministic train/val split of FreiHAND. Same surface as the synthetic loader."""

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
    ):
        self.root = Path(root)
        self.batch_size = batch_size
        self.img_size = img_size
        self.in_channels = in_channels
        self.augment = augment
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

    def _load_one(self, image_idx: int) -> tuple[np.ndarray, np.ndarray]:
        from PIL import Image

        img = Image.open(self._files[image_idx]).convert("RGB")
        w0, h0 = img.size  # original resolution (PIL gives width, height)
        anno = self._anno_of[image_idx]
        uv = (self._k[anno] @ self._xyz[anno].T).T  # [21,3]
        uv = uv[:, :2] / uv[:, 2:3]  # pixel coords in original resolution
        kpts = uv / np.array([w0, h0], dtype=np.float32)  # → [0,1], resolution-independent

        arr = np.asarray(img.resize((self.img_size, self.img_size)), dtype=np.float32) / 255.0
        if self.in_channels == 1:
            arr = arr.mean(axis=2, keepdims=True)
        if self.augment:
            if self._rng.random() < 0.5:  # horizontal flip (mirror x of every keypoint)
                arr = arr[:, ::-1, :].copy()
                kpts[:, 0] = 1.0 - kpts[:, 0]
            arr = np.clip(arr * self._rng.uniform(0.7, 1.3), 0.0, 1.0)  # brightness jitter
        chw = np.transpose(arr, (2, 0, 1)).astype(np.float32)  # [C,H,W]
        return chw, kpts.reshape(-1).astype(np.float32)

    def next_batch(self) -> tuple[np.ndarray, np.ndarray]:
        imgs = np.zeros(
            (self.batch_size, self.in_channels, self.img_size, self.img_size), dtype=np.float32
        )
        kpts = np.zeros((self.batch_size, _OUT), dtype=np.float32)
        for b in range(self.batch_size):
            if self._cursor >= len(self._order):  # epoch boundary → reshuffle
                self._cursor = 0
                self.passes += 1
                self._rng.shuffle(self._order)
            imgs[b], kpts[b] = self._load_one(int(self._order[self._cursor]))
            self._cursor += 1
        return imgs, kpts

    # Vision data is generated/streamed on the fly: the text-stream skip/index is a no-op.
    def skip(self, n: int) -> int:
        self._cursor += n
        return n

    def save_skip_index(self, path) -> None:
        pass

    def load_skip_index(self, path) -> bool:
        return False
