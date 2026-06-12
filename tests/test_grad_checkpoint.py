"""
Gradient (activation) checkpointing must produce the SAME loss and gradients as the
normal forward — it only changes WHEN activations are computed, not the math. Tested
with dropout=0 so the forward and the recompute are identical on both backends.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import mlx.core as mx
import mlx.nn as nn

from src.model.transformer import RDMCAFoundational, ModelConfig


def _cfg(ckpt):
    return ModelConfig(d_model=32, n_layers=4, n_heads=4, n_kv_heads=2, ffn_dim=64,
                       context_len=32, vocab_size=64, mrl_dims=[16, 32], dropout=0.0,
                       gradient_checkpointing=ckpt)


def test_checkpoint_forward_matches():
    np.random.seed(0); mx.random.seed(0)
    m_off = RDMCAFoundational(_cfg(False))
    m_on  = RDMCAFoundational(_cfg(True))
    # copy weights so both models are identical
    m_on.update(m_off.parameters())
    toks = mx.array(np.random.randint(0, 64, (2, 16)))
    assert np.allclose(np.asarray(m_off(toks)), np.asarray(m_on(toks)), atol=1e-5)


def test_checkpoint_grads_match():
    np.random.seed(1); mx.random.seed(1)
    m_off = RDMCAFoundational(_cfg(False))
    m_on  = RDMCAFoundational(_cfg(True))
    m_on.update(m_off.parameters())
    toks = mx.array(np.random.randint(0, 64, (2, 16)))

    def loss_fn(model):
        return model.mrl_loss(toks)

    l_off, g_off = nn.value_and_grad(m_off, loss_fn)(m_off)
    l_on,  g_on  = nn.value_and_grad(m_on,  loss_fn)(m_on)
    assert abs(float(l_off) - float(l_on)) < 1e-4

    # compare a representative gradient (first block's attention q_proj)
    go = np.asarray(g_off["blocks"][0]["attn"]["q_proj"]["weight"])
    gn = np.asarray(g_on["blocks"][0]["attn"]["q_proj"]["weight"])
    assert np.allclose(go, gn, atol=1e-4), f"max diff {np.abs(go-gn).max():.2e}"


def test_checkpoint_with_ple_mtp():
    """Checkpointing must also be correct with the optional PLE/MTP modules on."""
    np.random.seed(2); mx.random.seed(2)
    base = dict(d_model=32, n_layers=3, n_heads=4, n_kv_heads=2, ffn_dim=64,
                context_len=32, vocab_size=64, mrl_dims=[16, 32], dropout=0.0,
                n_mtp_heads=1, mtp_hidden_dim=16, ple_dim=4)
    m_off = RDMCAFoundational(ModelConfig(**base, gradient_checkpointing=False))
    m_on  = RDMCAFoundational(ModelConfig(**base, gradient_checkpointing=True))
    m_on.update(m_off.parameters())
    toks = mx.array(np.random.randint(0, 64, (2, 16)))
    lo = float(nn.value_and_grad(m_off, lambda m: m.mrl_loss(toks))(m_off)[0])
    ln = float(nn.value_and_grad(m_on,  lambda m: m.mrl_loss(toks))(m_on)[0])
    assert abs(lo - ln) < 1e-4
