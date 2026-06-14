"""
Audio Tokenizer — VQ-VAE over log-mel spectrograms (RDMCA §7.2 / Guide §4.1)

A small 1-D convolutional VQ-VAE, trained from scratch in MLX, that turns a
waveform into a sequence of discrete tokens from a learned codebook of size
AUDIO_VOCAB_SIZE (~25 tokens/sec). Indices occupy the audio range of the unified
vocabulary (offset applied by the perception layer / caller).

This keeps the project self-contained and torch-free. EnCodec (Meta) is a valid
drop-in alternative if you prefer a pretrained codec — see docs/GUIDE.md.

Train with: python scripts/train_tokenizer.py --audio-dir path/to/wavs
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import src.core.backend as backend

from .vocab import AUDIO_VOCAB_SIZE
from .vq import VectorQuantizer

B = backend.current()
nn = B.nn
ops = B.ops

SAMPLE_RATE = 16_000
N_MELS = 64
FRAME_MS = 25
HOP_MS = 10
EMB_DIM = 64
HIDDEN = 128


# ---------------------------------------------------------------------------
# numpy log-mel feature extractor (no librosa dependency)
# ---------------------------------------------------------------------------


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    m_min, m_max = hz2mel(0), hz2mel(sr / 2)
    mels = np.linspace(m_min, m_max, n_mels + 2)
    hz = mel2hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        for k in range(l, c):
            if c > l:
                fb[i - 1, k] = (k - l) / (c - l)
        for k in range(c, r):
            if r > c:
                fb[i - 1, k] = (r - k) / (r - c)
    return fb


def logmel(wav: np.ndarray, sr: int = SAMPLE_RATE, n_mels: int = N_MELS) -> np.ndarray:
    """waveform [T] float → log-mel spectrogram [frames, n_mels]."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    frame = int(sr * FRAME_MS / 1000)
    hop = int(sr * HOP_MS / 1000)
    n_fft = 1
    while n_fft < frame:
        n_fft *= 2
    if len(wav) < frame:
        wav = np.pad(wav, (0, frame - len(wav)))
    window = np.hanning(frame).astype(np.float32)
    fb = _mel_filterbank(sr, n_fft, n_mels)
    frames = []
    for start in range(0, len(wav) - frame + 1, hop):
        seg = wav[start : start + frame] * window
        spec = np.abs(np.fft.rfft(seg, n=n_fft)) ** 2
        mel = fb @ spec
        frames.append(np.log(mel + 1e-6))
    if not frames:
        frames = [np.log(fb @ (np.abs(np.fft.rfft(wav[:frame] * window, n=n_fft)) ** 2) + 1e-6)]
    return np.stack(frames, axis=0).astype(np.float32)  # [frames, n_mels]


# ---------------------------------------------------------------------------
# 1-D conv VQ-VAE over log-mel frames
# ---------------------------------------------------------------------------


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(N_MELS, HIDDEN, 4, stride=2, padding=1)
        self.c2 = nn.Conv1d(HIDDEN, HIDDEN, 4, stride=2, padding=1)
        self.c3 = nn.Conv1d(HIDDEN, EMB_DIM, 3, stride=1, padding=1)

    def __call__(self, x):  # x: [B, N_MELS, T] (NCL)
        x = ops.relu(self.c1(x))
        x = ops.relu(self.c2(x))
        return self.c3(x)


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(EMB_DIM, HIDDEN, 3, stride=1, padding=1)
        self.t1 = nn.ConvTranspose1d(HIDDEN, HIDDEN, 4, stride=2, padding=1)
        self.t2 = nn.ConvTranspose1d(HIDDEN, N_MELS, 4, stride=2, padding=1)

    def __call__(self, x):
        x = ops.relu(self.c1(x))
        x = ops.relu(self.t1(x))
        return self.t2(x)


class AudioVQVAE(nn.Module):
    """1-D VQ-VAE over log-mel frames (~/4 time downsampling → ~25 tok/s).

    Internal tensor layout is channels-first (NCL = [B, N_MELS, T]) so the
    convs are backend-neutral. NOTE: weight checkpoints are not cross-backend
    (conv layouts differ) — train + load on the same backend."""

    def __init__(self, codebook_size: int = AUDIO_VOCAB_SIZE):
        super().__init__()
        self.codebook_size = codebook_size
        self.encoder = _Encoder()
        self.vq = VectorQuantizer(codebook_size, EMB_DIM)
        self.decoder = _Decoder()

    def _crop(self, a, b):
        t = min(a.shape[2], b.shape[2])  # time is the last axis (NCL)
        return a[:, :, :t], b[:, :, :t]

    def _quantize(self, z):
        """z encoder output [B, EMB, T'] → (z_q NCL, idx [B,T'], vq_loss)."""
        z = ops.transpose(z, (0, 2, 1))  # -> [B, T', EMB] for VQ
        z_q, idx, vq_loss = self.vq(z)
        return ops.transpose(z_q, (0, 2, 1)), idx, vq_loss

    # -- training --------------------------------------------------------
    def loss(self, mel):
        """mel: [B, N_MELS, T]. Reconstruction MSE + VQ loss."""
        z_q, _, vq_loss = self._quantize(self.encoder(mel))
        recon = self.decoder(z_q)
        recon, target = self._crop(recon, mel)
        return ops.mean((recon - target) ** 2) + vq_loss

    # -- inference -------------------------------------------------------
    def encode_ids(self, wav: np.ndarray, sr: int = SAMPLE_RATE) -> list[int]:
        """waveform → list of raw codebook indices."""
        mel = ops.array(np.transpose(logmel(wav, sr), (1, 0))[None])  # [1, N_MELS, T]
        _, idx, _ = self._quantize(self.encoder(mel))
        B.engine.eval(idx)
        return [int(v) for v in ops.to_numpy(idx).reshape(-1)]

    def decode_mel(self, ids: list[int]) -> np.ndarray:
        """Raw codebook indices → reconstructed log-mel [T, N_MELS]."""
        idx = ops.array(np.array(ids, dtype=np.int64).reshape(1, -1))  # [1, L]
        z_q = ops.transpose(self.vq.lookup(idx), (0, 2, 1))  # [1, EMB, L]
        mel = self.decoder(z_q)  # [1, N_MELS, T]
        B.engine.eval(mel)
        return np.transpose(ops.to_numpy(mel)[0], (1, 0))  # [T, N_MELS]

    # -- persistence -----------------------------------------------------
    def save(self, path: str) -> None:
        B.engine.save_weights(self, path)
        Path(path).with_suffix(".json").write_text(
            json.dumps(
                {"codebook_size": self.codebook_size, "n_mels": N_MELS, "sample_rate": SAMPLE_RATE}
            )
        )

    @classmethod
    def load(cls, path: str) -> AudioVQVAE | None:
        p = Path(path)
        if not p.exists():
            return None
        meta_path = p.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        model = cls(codebook_size=meta.get("codebook_size", AUDIO_VOCAB_SIZE))
        B.engine.load_weights(model, str(p))
        return model
