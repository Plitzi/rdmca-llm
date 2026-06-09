"""
Vector Quantizer — shared core of the image and audio VQ-VAE tokenizers.

Maps continuous encoder outputs to the nearest entry in a learned codebook
(VQ-VAE, van den Oord et al.) and returns the discrete indices that become
tokens in the unified vocabulary. Straight-through estimator passes gradients
back to the encoder; codebook + commitment losses train the codebook.

Backend-neutral (written against `src.backend.current()`).
"""
from __future__ import annotations

import src.backend as backend

B = backend.current()
nn = B.nn
ops = B.ops


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, dim: int, beta: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim  = dim
        self.beta = beta
        self.codebook = nn.Parameter(ops.randn((codebook_size, dim)) * (1.0 / codebook_size))

    def _nearest(self, flat):
        """flat: [N, dim] → nearest codebook indices [N]."""
        cb = self.codebook
        # ||x - e||^2 = ||x||^2 - 2 x·e + ||e||^2
        d = (ops.sum(flat * flat, axis=1, keepdims=True)
             - 2.0 * (flat @ cb.T)
             + ops.sum(cb * cb, axis=1)[None, :])
        return ops.argmin(d, axis=1)

    def __call__(self, z):
        """
        z: [..., dim] encoder output.
        Returns (z_q_straight_through, indices[...], vq_loss).
        """
        shape = z.shape
        flat  = z.reshape(-1, self.dim)
        idx   = self._nearest(flat)
        z_q   = self.codebook[idx].reshape(shape)

        codebook_loss   = ops.mean((ops.stop_gradient(z) - z_q) ** 2)
        commitment_loss = ops.mean((z - ops.stop_gradient(z_q)) ** 2)
        vq_loss = codebook_loss + self.beta * commitment_loss

        z_q_st = z + ops.stop_gradient(z_q - z)   # straight-through
        return z_q_st, idx.reshape(shape[:-1]), vq_loss

    def lookup(self, idx):
        """Indices → codebook vectors (for decoding)."""
        return self.codebook[idx]
