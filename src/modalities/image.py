"""
Image Tokenizer — VQ-VAE (RDMCA §7.2 / Implementation Guide §3.1)

A small convolutional VQ-VAE, trained from scratch (no external vision model),
that maps an image to a grid of discrete tokens drawn from a learned codebook of
size IMAGE_VOCAB_SIZE. Those indices occupy the image range of the unified
vocabulary (offset applied by the perception layer / caller).

Pipeline:  image → encoder (conv ↓) → vector-quantize → indices  (encode)
           indices → codebook → decoder (conv ↑) → image          (decode)

Train with: python scripts/train_tokenizer.py --images-dir path/ (or --image-dataset)

Backend-neutral: convs use the channels-first (NCHW) convention via the backend
facade. NOTE: VQ-VAE weight checkpoints are not cross-backend (conv weight
layouts differ between MLX and PyTorch); train + load on the same backend.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional

import numpy as np

import src.backend as backend
from .vq import VectorQuantizer
from .vocab import IMAGE_VOCAB_SIZE

B = backend.current()
nn = B.nn
ops = B.ops

DEFAULT_IMG_SIZE = 32      # square; CIFAR-scale by default, configurable
EMB_DIM          = 64
HIDDEN           = 128


class _Encoder(nn.Module):
    """Two stride-2 convs → /4 spatial; final 3×3 conv to embedding dim."""
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, HIDDEN, 4, stride=2, padding=1)
        self.c2 = nn.Conv2d(HIDDEN, HIDDEN, 4, stride=2, padding=1)
        self.c3 = nn.Conv2d(HIDDEN, EMB_DIM, 3, stride=1, padding=1)

    def __call__(self, x):                       # x: [B, 3, H, W]
        x = ops.relu(self.c1(x))
        x = ops.relu(self.c2(x))
        return self.c3(x)


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(EMB_DIM, HIDDEN, 3, stride=1, padding=1)
        self.t1 = nn.ConvTranspose2d(HIDDEN, HIDDEN, 4, stride=2, padding=1)
        self.t2 = nn.ConvTranspose2d(HIDDEN, 3, 4, stride=2, padding=1)

    def __call__(self, x):
        x = ops.relu(self.c1(x))
        x = ops.relu(self.t1(x))
        return ops.sigmoid(self.t2(x))


class ImageVQVAE(nn.Module):
    """Convolutional VQ-VAE image tokenizer. Tokens per image = (img/4)^2."""

    def __init__(self, img_size: int = DEFAULT_IMG_SIZE,
                 codebook_size: int = IMAGE_VOCAB_SIZE):
        super().__init__()
        self.img_size      = img_size
        self.codebook_size = codebook_size
        self.grid          = img_size // 4
        self.n_tokens      = self.grid * self.grid
        self.encoder = _Encoder()
        self.vq      = VectorQuantizer(codebook_size, EMB_DIM)
        self.decoder = _Decoder()

    # -- training --------------------------------------------------------
    def loss(self, x):
        """x: [B, 3, H, W] in [0,1]. Reconstruction MSE + VQ loss."""
        z = self.encoder(x)                       # [B, EMB, g, g]
        z = ops.transpose(z, (0, 2, 3, 1))        # -> [B, g, g, EMB] for VQ
        z_q, _, vq_loss = self.vq(z)
        z_q = ops.transpose(z_q, (0, 3, 1, 2))    # -> NCHW for decoder
        recon = self.decoder(z_q)
        return ops.mean((recon - x) ** 2) + vq_loss

    # -- inference -------------------------------------------------------
    def encode_ids(self, image) -> List[int]:
        """np image [H,W,3] (0-255 or 0-1) → list of raw codebook indices."""
        chw = np.transpose(self._preprocess(image), (2, 0, 1))  # [3,H,W]
        x = ops.array(chw[None])                                 # [1,3,H,W]
        z = ops.transpose(self.encoder(x), (0, 2, 3, 1))         # [1,g,g,EMB]
        _, idx, _ = self.vq(z)
        B.engine.eval(idx)
        return [int(v) for v in ops.to_numpy(idx).reshape(-1)]

    def decode_ids(self, ids: List[int]) -> np.ndarray:
        """Raw codebook indices → reconstructed image [H,W,3] uint8."""
        g = self.grid
        idx = ops.array(np.array(ids[:g * g], dtype=np.int64).reshape(1, g, g))
        z_q = self.vq.lookup(idx)                                # [1,g,g,EMB]
        z_q = ops.transpose(z_q, (0, 3, 1, 2))                   # -> NCHW
        img = self.decoder(z_q)                                  # [1,3,H,W]
        B.engine.eval(img)
        out = np.transpose(ops.to_numpy(img)[0], (1, 2, 0))      # [H,W,3]
        return (np.clip(out, 0, 1) * 255).astype(np.uint8)

    def _preprocess(self, image) -> np.ndarray:
        arr = np.asarray(image, dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        arr = _resize(arr, self.img_size)
        return arr.astype(np.float32)

    # -- persistence -----------------------------------------------------
    def save(self, path: str) -> None:
        B.engine.save_weights(self, path)
        meta = {"img_size": self.img_size, "codebook_size": self.codebook_size}
        Path(path).with_suffix(".json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: str) -> Optional["ImageVQVAE"]:
        p = Path(path)
        if not p.exists():
            return None
        meta_path = p.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        model = cls(img_size=meta.get("img_size", DEFAULT_IMG_SIZE),
                    codebook_size=meta.get("codebook_size", IMAGE_VOCAB_SIZE))
        B.engine.load_weights(model, str(p))
        return model


def _resize(arr: np.ndarray, size: int) -> np.ndarray:
    """Resize [H,W,3] to [size,size,3]. Uses PIL when available, else nearest."""
    if arr.shape[0] == size and arr.shape[1] == size:
        return arr
    try:
        from PIL import Image
        im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
        im = im.resize((size, size), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0
    except ImportError:
        h, w = arr.shape[:2]
        yi = (np.arange(size) * h / size).astype(int)
        xi = (np.arange(size) * w / size).astype(int)
        return arr[yi][:, xi]
