"""Gesture dataset loader + download for hands_recognition stage 3.

Stage 3 trains a small gesture-classification head on the FROZEN multi-hand backbone: an image
→ the backbone's predicted hand keypoints → a gesture class (e.g. thumbs-up). The data is a
folder of labelled hand images laid out as one subdirectory per gesture:

    root/<gesture>/*.jpg        (e.g. root/thumbs_up/0001.jpg)

This matches a HaGRID subset (the public Hand Gesture Recognition Image Dataset) once extracted
to that layout. The loader yields (images [N, C, H, W] in [0,1], labels [N]) and mirrors the
synthetic loader's telemetry + no-op skip/index surface, so the ModelSpec swaps it in for
stage 3 transparently. `download_gestures` fetches + extracts the archive idempotently, reusing
the resumable downloader from data_freihand (the same `rdmca prepare` data step).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Fixed gesture vocabulary (index 0 = no gesture). Add a gesture by appending here and
# providing a matching `root/<gesture>/` folder; n_gestures is derived from this list.
GESTURES = ("no_gesture", "thumbs_up", "fist", "open_palm", "peace", "ok")
_GESTURE_INDEX = {name: i for i, name in enumerate(GESTURES)}
# A small public HaGRID-style subset would live here; the URL is configurable per call so the
# exact subset can be pinned without touching code. Left as a placeholder until one is chosen.
_GESTURES_URL = ""
_READY_MARKER = "thumbs_up"  # a gesture folder whose presence means "already prepared"


def n_gestures() -> int:
    return len(GESTURES)


def is_prepared(root: str | Path) -> bool:
    """True when at least the marker gesture folder (with images) is extracted under `root`."""
    marker = Path(root) / _READY_MARKER
    return marker.is_dir() and any(marker.glob("*.jpg") or marker.glob("*.png"))


def download_gestures(root: str | Path, *, url: str = _GESTURES_URL) -> Path:
    """Download + extract the gesture dataset into `root` (idempotent, resumable) — the same
    pattern as `download_freihand`, reusing its streaming downloader. Expects a zip that
    extracts to `root/<gesture>/*.jpg`. Re-running once a gesture folder exists is a no-op."""
    import zipfile

    from models.hands_recognition.data_freihand import _download_resumable

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if is_prepared(root):
        print(f"  Gesture dataset already prepared in {root} — skipping download.")
        return root
    if not url:
        raise RuntimeError(
            "No gesture-dataset URL configured. Set the dataset URL (a HaGRID subset) in "
            "data_gestures._GESTURES_URL or pass url=..., then re-run prepare. Expected layout "
            f"after extraction: {root}/<gesture>/*.jpg for gestures {GESTURES}."
        )
    zip_path = root / "gestures.zip"
    part_path = root / "gestures.zip.part"
    if not zip_path.exists():
        _download_resumable(url, part_path)
        part_path.rename(zip_path)
    print(f"  Extracting {zip_path.name} → {root} …")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    if not is_prepared(root):
        raise RuntimeError(f"Gesture extraction did not yield {root}/<gesture>/*.jpg.")
    zip_path.unlink()
    print(f"  Gesture dataset ready in {root}.")
    return root


class GestureLoader:
    """Yields (images [N, C, H, W] in [0,1], labels [N] gesture indices) from a deterministic
    train/val split of a `root/<gesture>/*.jpg` tree. Same surface as the synthetic loader."""

    _SPLIT_SEED = 4242

    def __init__(
        self,
        root: str,
        batch_size: int,
        img_size: int = 128,
        in_channels: int = 3,
        split: str = "train",
        seed: int = 0,
        val_fraction: float = 0.1,
    ):
        self.root = Path(root)
        self.batch_size = batch_size
        self.img_size = img_size
        self.in_channels = in_channels
        self._rng = np.random.default_rng(seed)
        self.passes = 0
        self.last_was_replay = False
        self.replay_fraction = 0.0

        samples: list[tuple[Path, int]] = []
        for name, idx in _GESTURE_INDEX.items():
            folder = self.root / name
            if folder.is_dir():
                for img in sorted([*folder.glob("*.jpg"), *folder.glob("*.png")]):
                    samples.append((img, idx))
        if not samples:
            raise FileNotFoundError(
                f"No gesture images under {self.root} (expected {self.root}/<gesture>/*.jpg "
                f"for {GESTURES}). See the GUIDE for the download."
            )
        order = np.arange(len(samples))
        np.random.default_rng(self._SPLIT_SEED).shuffle(order)
        n_val = max(1, int(len(order) * val_fraction))
        keep = order[:n_val] if split == "val" else order[n_val:]
        self._samples = [samples[i] for i in keep]
        # Same unit as the trainer's tokens_seen (batch_size · seq_len, seq_len = img_size), so
        # the max_corpus_passes cap counts real passes over the gesture set (see FreiHandLoader).
        self.epoch_tokens = len(self._samples) * self.img_size
        self._order = np.arange(len(self._samples))
        self._rng.shuffle(self._order)
        self._cursor = 0

    def num_batches(self) -> int:
        return max(1, len(self._samples) // self.batch_size)

    def _load_one(self, i: int) -> tuple[np.ndarray, int]:
        from PIL import Image

        path, label = self._samples[i]
        img = Image.open(path).convert("RGB").resize((self.img_size, self.img_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if self.in_channels == 1:
            arr = arr.mean(axis=2, keepdims=True)
        return np.transpose(arr, (2, 0, 1)).astype(np.float32), label

    def next_batch(self) -> tuple[np.ndarray, np.ndarray]:
        n, c, size = self.batch_size, self.in_channels, self.img_size
        imgs = np.zeros((n, c, size, size), dtype=np.float32)
        labels = np.zeros(n, dtype=np.int64)
        for b in range(n):
            if self._cursor >= len(self._order):
                self._cursor = 0
                self.passes += 1
                self._rng.shuffle(self._order)
            imgs[b], labels[b] = self._load_one(int(self._order[self._cursor]))
            self._cursor += 1
        return imgs, labels

    # Vision data is streamed on the fly: the text-stream skip/index is a no-op.
    def skip(self, n: int) -> int:
        self._cursor += n
        return n

    def save_skip_index(self, path) -> None:
        pass

    def load_skip_index(self, path) -> bool:
        return False
