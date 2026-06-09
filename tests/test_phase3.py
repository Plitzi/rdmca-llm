"""
Phase 3 Acceptance Tests — Image Modality (VQ-VAE + unified vocab)
"""
import numpy as np

from src.modalities.image import ImageVQVAE
from src.modalities.vocab import build_modality_layout, IMAGE_VOCAB_SIZE


def test_vqvae_roundtrip_shapes():
    """encode→decode produces the right token count and image shape."""
    m = ImageVQVAE(img_size=32)
    img = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    ids = m.encode_ids(img)
    assert len(ids) == m.n_tokens == 64          # (32/4)^2
    assert all(0 <= i < IMAGE_VOCAB_SIZE for i in ids)
    rec = m.decode_ids(ids)
    assert rec.shape == (32, 32, 3)


def test_vqvae_trains_one_step():
    """A single gradient step computes a finite, non-negative loss.

    Backend-neutral: the VQ-VAE is channels-first (NCHW), so the input batch is
    [B, 3, H, W]. Runs on whichever backend is active (mlx | torch)."""
    import src.backend as backend
    B = backend.current()
    m = ImageVQVAE(img_size=32)
    B.engine.set_precision(m, "fp32")
    x = B.ops.array(np.random.rand(4, 3, 32, 32).astype(np.float32))   # NCHW
    lg = B.engine.value_and_grad(m, lambda mdl, b: mdl.loss(b))
    opt = B.engine.make_optimizer(m, lr=1e-3, weight_decay=0.0)
    l0, g = lg(m, x); B.engine.optimizer_step(opt, m, g)
    assert float(B.engine.item(l0)) >= 0.0


def test_unified_vocab_layout_disjoint():
    """Image tokens occupy a disjoint range above the text vocab."""
    layout = build_modality_layout(text_vocab_size=8000)
    assert layout["image"]["offset"] == 8000
    assert layout["audio"]["offset"] == 8000 + IMAGE_VOCAB_SIZE
    assert layout["total"] == layout["audio"]["offset"] + layout["audio"]["size"]


def test_image_tokens_offset_into_unified_range():
    """Perception offsets raw codebook ids into the image range + boundaries."""
    from src.modalities.perception import MultimodalPerception
    info = {"modality_layout": build_modality_layout(8000),
            "modality_tokens": {"mod_image": 7, "mod_end": 9}}
    mpl = MultimodalPerception(image_tok=ImageVQVAE(img_size=32), info=info)
    seq = mpl.encode_image((np.random.rand(32, 32, 3) * 255).astype(np.uint8))
    assert seq[0] == 7 and seq[-1] == 9
    assert all(8000 <= t < 8000 + IMAGE_VOCAB_SIZE for t in seq[1:-1])
