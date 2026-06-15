"""
Tests for VQ-VAE codebook-collapse prevention (EMA + dead-code reset) in
src/core/modalities/vq.py. Default (ema=False) must be unchanged; ema=True must move the
codebook toward the data and recycle dead entries instead of letting them die.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import numpy as np

from src.core.modalities.vq import VectorQuantizer


def test_default_behavior_unchanged():
    """ema=False keeps the original loss (codebook_loss + commitment) and ema_update
    is a no-op — no regression for the existing image/audio tokenizers."""
    vq = VectorQuantizer(16, 4)  # ema defaults to False
    z = mx.array(np.random.randn(2, 5, 4).astype(np.float32))
    cb_before = np.asarray(vq.codebook).copy()
    z_q, idx, loss = vq(z)
    assert z_q.shape == z.shape and idx.shape == (2, 5)
    assert float(loss) > 0
    vq.ema_update(z)  # no-op when ema=False
    assert np.array_equal(np.asarray(vq.codebook), cb_before)


def test_ema_forward_drops_codebook_loss():
    """With ema=True the forward returns only the commitment loss (the EMA owns the
    codebook), which is strictly smaller than the full vq_loss for the same input."""
    np.random.seed(0)
    z = mx.array(np.random.randn(3, 4).astype(np.float32))
    vq_full = VectorQuantizer(16, 4, ema=False)
    vq_ema = VectorQuantizer(16, 4, ema=True)
    vq_ema.codebook = vq_full.codebook  # same codebook → compare losses
    _, _, l_full = vq_full(z)
    _, _, l_ema = vq_ema(z)
    assert float(l_ema) < float(l_full)


def test_ema_moves_codebook_toward_data():
    """An EMA update must pull the codebook toward the encoder vectors assigned to
    it: after updates on a fixed cluster, the nearest code sits closer to the cluster
    mean than it did at init."""
    np.random.seed(1)
    vq = VectorQuantizer(8, 3, ema=True, decay=0.8, dead_threshold=0.0)
    center = np.array([5.0, -3.0, 2.0], dtype=np.float32)
    data = mx.array((center + 0.01 * np.random.randn(64, 3)).astype(np.float32))
    # nearest code to the center at init
    cb0 = np.asarray(vq.codebook)
    near0 = int(np.argmin(((cb0 - center) ** 2).sum(1)))
    d0 = float(((cb0[near0] - center) ** 2).sum())
    for _ in range(40):
        vq.ema_update(data)
    cb1 = np.asarray(vq.codebook)
    d1 = float(((cb1[np.argmin(((cb1 - center) ** 2).sum(1))] - center) ** 2).sum())
    assert d1 < d0, f"codebook did not move toward data ({d1} !< {d0})"


def test_dead_code_reset_recycles_unused_entries():
    """A codebook entry the data never selects (its EMA cluster size decays below the
    threshold) must be reseeded onto a current encoder vector, not left stranded."""
    np.random.seed(2)
    vq = VectorQuantizer(6, 2, ema=True, decay=0.5, dead_threshold=0.5)
    # Strand one entry far away; feed data clustered elsewhere so it is never chosen.
    cb = np.asarray(vq.codebook).copy()
    cb[0] = np.array([100.0, 100.0], np.float32)
    vq.codebook = mx.array(cb)
    data = mx.array((np.array([0.0, 0.0]) + 0.01 * np.random.randn(50, 2)).astype(np.float32))
    for _ in range(20):
        vq.ema_update(data)
    cb_after = np.asarray(vq.codebook)
    # entry 0 must have been pulled in from (100,100) to near the data cluster (~0,0)
    assert np.linalg.norm(cb_after[0]) < 1.0, f"dead entry not reset: {cb_after[0]}"


def test_perplexity_improves_with_ema_and_reset():
    """End-to-end: with EMA + reset, codebook usage (perplexity) on multi-cluster
    data is meaningfully higher than a frozen random codebook's — i.e. more codes are
    actually used (collapse is mitigated)."""
    np.random.seed(3)
    centers = np.array([[3, 0], [-3, 0], [0, 3], [0, -3]], dtype=np.float32)

    def batch():
        c = centers[np.random.randint(0, 4, size=128)]
        return mx.array((c + 0.05 * np.random.randn(128, 2)).astype(np.float32))

    vq = VectorQuantizer(16, 2, ema=True, decay=0.9, dead_threshold=1.0)
    ppl_before = vq.perplexity(batch())
    for _ in range(60):
        vq.ema_update(batch())
    ppl_after = vq.perplexity(batch())
    assert ppl_after >= ppl_before  # usage should not collapse; typically improves
    assert ppl_after > 2.0  # at least a few codes meaningfully used
