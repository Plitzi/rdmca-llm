"""
Audio Tokenizer — VQ-VAE over log-mel spectrograms (RDMCA §7.2 / Guide §4.1)

A small 1-D convolutional VQ-VAE, trained from scratch in MLX, that turns a
waveform into a sequence of discrete tokens from a learned codebook of size
AUDIO_VOCAB_SIZE (~25 tokens/sec). Indices occupy the audio range of the unified
vocabulary (offset applied by the perception layer / caller).

This keeps the project self-contained and torch-free. EnCodec (Meta) is a valid
drop-in alternative if you prefer a pretrained codec — see docs/GUIDE.md.

Train with: python scripts/train_audio_tokenizer.py
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from .vq import VectorQuantizer
from .vocab import AUDIO_VOCAB_SIZE

SAMPLE_RATE = 16_000
N_MELS      = 64
FRAME_MS    = 25
HOP_MS      = 10
EMB_DIM     = 64
HIDDEN      = 128


# ---------------------------------------------------------------------------
# numpy log-mel feature extractor (no librosa dependency)
# ---------------------------------------------------------------------------

def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    def hz2mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel2hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    m_min, m_max = hz2mel(0), hz2mel(sr / 2)
    mels = np.linspace(m_min, m_max, n_mels + 2)
    hz   = mel2hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        for k in range(l, c):
            if c > l: fb[i - 1, k] = (k - l) / (c - l)
        for k in range(c, r):
            if r > c: fb[i - 1, k] = (r - k) / (r - c)
    return fb


def logmel(wav: np.ndarray, sr: int = SAMPLE_RATE, n_mels: int = N_MELS) -> np.ndarray:
    """waveform [T] float → log-mel spectrogram [frames, n_mels]."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    frame = int(sr * FRAME_MS / 1000)
    hop   = int(sr * HOP_MS / 1000)
    n_fft = 1
    while n_fft < frame:
        n_fft *= 2
    if len(wav) < frame:
        wav = np.pad(wav, (0, frame - len(wav)))
    window = np.hanning(frame).astype(np.float32)
    fb = _mel_filterbank(sr, n_fft, n_mels)
    frames = []
    for start in range(0, len(wav) - frame + 1, hop):
        seg = wav[start:start + frame] * window
        spec = np.abs(np.fft.rfft(seg, n=n_fft)) ** 2
        mel  = fb @ spec
        frames.append(np.log(mel + 1e-6))
    if not frames:
        frames = [np.log(fb @ (np.abs(np.fft.rfft(wav[:frame] * window, n=n_fft)) ** 2) + 1e-6)]
    return np.stack(frames, axis=0).astype(np.float32)   # [frames, n_mels]


# ---------------------------------------------------------------------------
# 1-D conv VQ-VAE over log-mel frames
# ---------------------------------------------------------------------------

class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(N_MELS, HIDDEN, 4, stride=2, padding=1)
        self.c2 = nn.Conv1d(HIDDEN, HIDDEN, 4, stride=2, padding=1)
        self.c3 = nn.Conv1d(HIDDEN, EMB_DIM, 3, stride=1, padding=1)

    def __call__(self, x: mx.array) -> mx.array:   # x: [B, T, N_MELS]
        x = nn.relu(self.c1(x))
        x = nn.relu(self.c2(x))
        return self.c3(x)


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(EMB_DIM, HIDDEN, 3, stride=1, padding=1)
        self.t1 = nn.ConvTranspose1d(HIDDEN, HIDDEN, 4, stride=2, padding=1)
        self.t2 = nn.ConvTranspose1d(HIDDEN, N_MELS, 4, stride=2, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.c1(x))
        x = nn.relu(self.t1(x))
        return self.t2(x)


class AudioVQVAE(nn.Module):
    """1-D VQ-VAE over log-mel frames (~/4 time downsampling → ~25 tok/s)."""

    def __init__(self, codebook_size: int = AUDIO_VOCAB_SIZE):
        super().__init__()
        self.codebook_size = codebook_size
        self.encoder = _Encoder()
        self.vq      = VectorQuantizer(codebook_size, EMB_DIM)
        self.decoder = _Decoder()

    def _crop(self, a: mx.array, b: mx.array):
        t = min(a.shape[1], b.shape[1])
        return a[:, :t, :], b[:, :t, :]

    # -- training --------------------------------------------------------
    def loss(self, mel: mx.array) -> mx.array:
        """mel: [B, T, N_MELS]. Reconstruction MSE + VQ loss."""
        z = self.encoder(mel)
        z_q, _, vq_loss = self.vq(z)
        recon = self.decoder(z_q)
        recon, target = self._crop(recon, mel)
        return mx.mean((recon - target) ** 2) + vq_loss

    # -- inference -------------------------------------------------------
    def encode_ids(self, wav: np.ndarray, sr: int = SAMPLE_RATE) -> List[int]:
        """waveform → list of raw codebook indices."""
        mel = mx.array(logmel(wav, sr))[None]     # [1, T, N_MELS]
        z = self.encoder(mel)
        _, idx, _ = self.vq(z)
        mx.eval(idx)
        return [int(v) for v in np.array(idx).reshape(-1)]

    def decode_mel(self, ids: List[int]) -> np.ndarray:
        """Raw codebook indices → reconstructed log-mel [T, N_MELS]."""
        idx = mx.array(np.array(ids, dtype=np.int32).reshape(1, -1))
        z_q = self.vq.lookup(idx)
        mel = self.decoder(z_q)
        mx.eval(mel)
        return np.array(mel)[0]

    # -- persistence -----------------------------------------------------
    def save(self, path: str) -> None:
        mx.savez(path, **dict(tree_flatten(self.parameters())))
        Path(path).with_suffix(".json").write_text(
            json.dumps({"codebook_size": self.codebook_size,
                        "n_mels": N_MELS, "sample_rate": SAMPLE_RATE}))

    @classmethod
    def load(cls, path: str) -> Optional["AudioVQVAE"]:
        p = Path(path)
        if not p.exists():
            return None
        meta_path = p.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        model = cls(codebook_size=meta.get("codebook_size", AUDIO_VOCAB_SIZE))
        model.update(tree_unflatten(list(mx.load(str(p)).items())))
        mx.eval(model.parameters())
        return model
