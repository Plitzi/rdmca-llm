"""
Vector Quantizer — shared core of the image and audio VQ-VAE tokenizers.

Maps continuous encoder outputs to the nearest entry in a learned codebook
(VQ-VAE, van den Oord et al.) and returns the discrete indices that become
tokens in the unified vocabulary. Straight-through estimator passes gradients
back to the encoder; codebook + commitment losses train the codebook.

Codebook collapse: with a purely gradient-trained codebook, a few entries win
early and the rest receive almost no gradient and die — 90%+ of an 8192-entry
codebook can go unused within ~1000 steps, wrecking reconstruction. Two standard
fixes are supported, OPT-IN via `ema=True` (default off = the original behaviour,
zero regression):
  • EMA codebook updates — the codebook tracks an exponential moving average of the
    encoder vectors assigned to each entry (more stable than SGD on the codebook);
  • dead-code reset — entries whose usage falls below a threshold are reseeded to a
    random current-batch encoder vector, recycling dead capacity to where the data
    actually is.
Both run in `ema_update(z)`, called by the trainer ONCE per step AFTER the optimizer
step (outside autograd), so the forward graph is never mutated mid-pass — keeping it
backend-neutral. With `ema=True` the forward drops the codebook loss (the EMA owns
the codebook) and keeps only the commitment loss.

Backend-neutral (written against `src.backend.current()`).
"""

from __future__ import annotations

import numpy as np

import src.backend as backend

B = backend.current()
nn = B.nn
ops = B.ops


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        dim: int,
        beta: float = 0.25,
        ema: bool = False,
        decay: float = 0.99,
        eps: float = 1e-5,
        dead_threshold: float = 1.0,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.beta = beta
        self.codebook = nn.Parameter(ops.randn((codebook_size, dim)) * (1.0 / codebook_size))

        # Anti-collapse (EMA + dead-code reset). State lives on the HOST (numpy), so
        # it never enters the param/grad tree and the update never touches autograd.
        self.ema = ema
        self.decay = decay
        self.eps = eps
        self.dead_threshold = dead_threshold
        self._N = np.zeros((codebook_size,), dtype=np.float64)  # EMA cluster size
        self._m = np.asarray(ops.to_numpy(self.codebook), dtype=np.float64)  # EMA sum of members

    def _nearest(self, flat):
        """flat: [N, dim] → nearest codebook indices [N]."""
        cb = self.codebook
        # ||x - e||^2 = ||x||^2 - 2 x·e + ||e||^2
        d = (
            ops.sum(flat * flat, axis=1, keepdims=True)
            - 2.0 * (flat @ cb.T)
            + ops.sum(cb * cb, axis=1)[None, :]
        )
        return ops.argmin(d, axis=1)

    def __call__(self, z):
        """
        z: [..., dim] encoder output.
        Returns (z_q_straight_through, indices[...], vq_loss).
        With ema=True the codebook loss is omitted (the EMA owns the codebook); the
        commitment loss (which trains the ENCODER toward the codebook) always stays.
        """
        shape = z.shape
        flat = z.reshape(-1, self.dim)
        idx = self._nearest(flat)
        z_q = self.codebook[idx].reshape(shape)

        commitment_loss = ops.mean((z - ops.stop_gradient(z_q)) ** 2)
        if self.ema:
            vq_loss = self.beta * commitment_loss
        else:
            codebook_loss = ops.mean((ops.stop_gradient(z) - z_q) ** 2)
            vq_loss = codebook_loss + self.beta * commitment_loss

        z_q_st = z + ops.stop_gradient(z_q - z)  # straight-through
        return z_q_st, idx.reshape(shape[:-1]), vq_loss

    def ema_update(self, z) -> None:
        """Update the codebook from the encoder outputs `z` by EMA, and reset dead
        entries. Call ONCE per training step, AFTER the optimizer step (outside
        autograd). No-op unless `ema=True`. `z` is the same encoder output passed to
        `__call__` (any leading shape; last dim must be `self.dim`)."""
        if not self.ema:
            return
        flat = np.asarray(ops.to_numpy(z), dtype=np.float64).reshape(-1, self.dim)
        if flat.shape[0] == 0:
            return
        cb = np.asarray(ops.to_numpy(self.codebook), dtype=np.float64)
        d = (flat * flat).sum(1, keepdims=True) - 2.0 * flat @ cb.T + (cb * cb).sum(1)[None, :]
        idx = d.argmin(1)  # [N]
        onehot = np.zeros((flat.shape[0], self.codebook_size), dtype=np.float64)
        onehot[np.arange(flat.shape[0]), idx] = 1.0

        n = onehot.sum(0)  # batch counts [K]
        dw = onehot.T @ flat  # batch sums [K, dim]
        self._N = self.decay * self._N + (1.0 - self.decay) * n
        self._m = self.decay * self._m + (1.0 - self.decay) * dw

        # Laplace-smoothed cluster sizes so an empty entry never divides by zero.
        total = self._N.sum()
        N = (self._N + self.eps) / (total + self.codebook_size * self.eps) * total
        cb_new = self._m / np.maximum(N[:, None], self.eps)

        # Dead-code reset: recycle entries the data has abandoned to a random current
        # encoder vector, so dead capacity is moved to where the data actually is.
        dead = np.where(self.dead_threshold > self._N)[0]
        if dead.size:
            pick = np.random.randint(0, flat.shape[0], size=dead.size)
            cb_new[dead] = flat[pick]
            self._N[dead] = 1.0  # fresh budget
            self._m[dead] = cb_new[dead]

        self.codebook = nn.Parameter(ops.array(cb_new.astype(np.float32)))

    def perplexity(self, z) -> float:
        """Codebook usage perplexity on `z` (1 = total collapse, codebook_size =
        perfectly uniform use) — a quick health metric for monitoring collapse."""
        flat = np.asarray(ops.to_numpy(z), dtype=np.float64).reshape(-1, self.dim)
        cb = np.asarray(ops.to_numpy(self.codebook), dtype=np.float64)
        d = (flat * flat).sum(1, keepdims=True) - 2.0 * flat @ cb.T + (cb * cb).sum(1)[None, :]
        idx = d.argmin(1)
        p = np.bincount(idx, minlength=self.codebook_size).astype(np.float64)
        p = p / max(p.sum(), 1.0)
        nz = p[p > 0]
        return float(np.exp(-(nz * np.log(nz)).sum()))

    def lookup(self, idx):
        """Indices → codebook vectors (for decoding)."""
        return self.codebook[idx]
