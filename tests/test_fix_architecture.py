"""Regression tests — model architecture: config validation, weight tying, causality,
MRL loss weighting + completion masking, and the MoE gate top_k. (Split from the old
test_fixes.py.) Self-contained: MLX backend + tiny models, no checkpoints needed."""

import numpy as np
import pytest
from fixes_common import B, ModelConfig, RDMCAFoundational, tiny_model

# ─────────────────────────── config validation (M5/L8) ───────────────────────


def test_modelconfig_rejects_unsorted_mrl():
    with pytest.raises(ValueError):
        ModelConfig(d_model=256, n_heads=4, mrl_dims=[256, 128])


def test_modelconfig_rejects_duplicate_mrl():
    with pytest.raises(ValueError):
        ModelConfig(d_model=256, n_heads=4, mrl_dims=[128, 128, 256])


def test_modelconfig_rejects_mrl_over_dmodel():
    with pytest.raises(ValueError):
        ModelConfig(d_model=256, n_heads=4, mrl_dims=[128, 512])


def test_modelconfig_rejects_indivisible_heads():
    with pytest.raises(ValueError):
        ModelConfig(d_model=100, n_heads=8, mrl_dims=[100])


def test_attention_rejects_odd_head_dim():
    # d_model 48 / n_heads 16 = head_dim 3 (odd) → RoPE assert in attention init.
    # Match the message so a DIFFERENT assertion firing first can't mask a regression.
    cfg = ModelConfig(
        d_model=48, n_heads=16, ffn_dim=96, context_len=16, vocab_size=128, mrl_dims=[48]
    )
    with pytest.raises(AssertionError, match="head_dim must be even"):
        RDMCAFoundational(cfg)


# ─────────────────────────── weight tying + causality ────────────────────────


def test_output_head_is_weight_tied_to_embedding():
    """The output projection must REUSE embed.weight (weight tying): no separate
    'head' param in the tree, and head_at_dim(h, d) == h[:, :d] @ embed.weight[:, :d].T."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    m = tiny_model()
    names = [k for k, _ in tree_flatten(m.parameters())]
    assert not any(".head." in k or k.startswith("head.") for k in names), (
        f"a separate output head leaked into the param tree: {[k for k in names if 'head' in k]}"
    )
    h = mx.array(np.random.randn(2, 5, 64).astype(np.float32))
    for d in (32, 64):
        got = np.array(m.head_at_dim(h, d).tolist())
        want = np.array((h[..., :d] @ m.embed.weight[:, :d].T).tolist())
        assert np.allclose(got, want, atol=1e-5), f"head_at_dim not tied to embed at d={d}"


def test_model_is_causal():
    """Output at position k must NOT depend on tokens after k (causal masking):
    changing a future token leaves all earlier positions' hidden states identical."""
    import mlx.core as mx

    m = tiny_model()
    m.train(False)
    base = np.random.randint(1, 256, (1, 16))
    a = base.copy()
    b = base.copy()
    b[0, 10:] = (b[0, 10:] + 7) % 256  # perturb only positions ≥10
    ha = np.array(m(mx.array(a)).tolist())
    hb = np.array(m(mx.array(b)).tolist())
    # positions 0..9 saw identical context → must be bit-close; position ≥10 differs.
    assert np.allclose(ha[:, :10], hb[:, :10], atol=1e-5), "future token leaked into past"
    assert not np.allclose(ha[:, 10:], hb[:, 10:], atol=1e-5), "perturbation had no effect"


# ─────────────────────────── MRL uniform weights + masking (C1) ──────────────


def test_mrl_weights_are_uniform():
    """mrl_loss must equal the simple mean of per-dim cross-entropies (uniform
    weighting), not a 1/d-weighted sum that starves the full head."""
    import mlx.core as mx

    cfg = ModelConfig(
        d_model=64,
        n_heads=2,
        ffn_dim=128,
        context_len=16,
        vocab_size=256,
        mrl_dims=[32, 64],
        dropout=0.0,
    )  # deterministic
    m = RDMCAFoundational(cfg)
    toks = mx.array(np.random.randint(0, 256, (2, 17)))
    total = m.mrl_loss(toks)
    inputs, targets = toks[:, :-1], toks[:, 1:]
    h = m(inputs)
    per = []
    for d in cfg.mrl_dims:
        lg = m.head_at_dim(h, d)
        Bsz, S, V = lg.shape
        per.append(
            B.ops.cross_entropy(lg.reshape(Bsz * S, V), targets.reshape(Bsz * S), reduction="mean")
        )
    expected = (per[0] + per[1]) / 2.0
    assert abs(float(total.item()) - float(expected.item())) < 1e-3


def test_eval_ce_mask_matches_training_objective():
    """Validation eval_ce with a completion mask must equal the masked mean over the
    unmasked (assistant) targets — the SAME objective training optimizes — and must
    differ from the unmasked full-sequence mean. Otherwise the gate measures context
    tokens the model is never trained to predict and perplexity is inflated ~7×."""
    import mlx.core as mx

    cfg = ModelConfig(
        d_model=64,
        n_heads=2,
        ffn_dim=128,
        context_len=16,
        vocab_size=256,
        mrl_dims=[32, 64],
        dropout=0.0,
    )
    m = RDMCAFoundational(cfg)
    toks = mx.array(np.random.randint(0, 256, (2, 17)))
    mask = np.zeros((2, 17), dtype=np.int32)
    mask[:, 9:] = 1  # only the tail trains
    mask_mx = mx.array(mask)

    masked = float(m.eval_ce(toks, mask=mask_mx).item())
    unmasked = float(m.eval_ce(toks).item())

    # Manual masked-mean reference at full dim.
    inputs, targets = toks[:, :-1], toks[:, 1:]
    lg = m.head_at_dim(m(inputs), cfg.mrl_dims[-1])
    Bsz, S, V = lg.shape
    ce = B.ops.cross_entropy(lg.reshape(Bsz * S, V), targets.reshape(Bsz * S), reduction="none")
    mm = mask_mx[:, 1:].reshape(Bsz * S).astype(ce.dtype)
    ref = float((B.ops.sum(ce * mm) / B.ops.sum(mm)).item())
    assert abs(masked - ref) < 1e-3
    assert abs(masked - unmasked) > 1e-3  # masking actually changes the metric

    # An all-ones mask reduces to the plain mean (prose stages are unaffected).
    allone = float(m.eval_ce(toks, mask=mx.ones((2, 17))).item())
    assert abs(allone - unmasked) < 1e-3


def test_mrl_loss_all_ones_mask_equals_unmasked():
    """An all-ones mask must reproduce the plain mean cross-entropy exactly."""
    import mlx.core as mx

    cfg = ModelConfig(
        d_model=64,
        n_heads=2,
        ffn_dim=128,
        context_len=16,
        vocab_size=256,
        mrl_dims=[32, 64],
        dropout=0.0,
    )
    m = RDMCAFoundational(cfg)
    toks = mx.array(np.random.randint(0, 256, (2, 17)))
    plain = float(m.mrl_loss(toks).item())
    masked = float(m.mrl_loss(toks, mx.ones((2, 17), dtype=mx.int32)).item())
    assert abs(plain - masked) < 1e-4


def test_mrl_loss_mask_equals_manual_restricted_mean():
    """The masked loss must equal the per-dim CE averaged over ONLY the unmasked
    target positions (the completion-only contract), independently recomputed."""
    import mlx.core as mx

    cfg = ModelConfig(
        d_model=64,
        n_heads=2,
        ffn_dim=128,
        context_len=16,
        vocab_size=256,
        mrl_dims=[32, 64],
        dropout=0.0,
    )
    m = RDMCAFoundational(cfg)
    toks = mx.array(np.random.randint(0, 256, (2, 17)))
    mask = np.zeros((2, 17), dtype=np.int32)
    mask[:, 9:] = 1  # train only the tail
    got = float(m.mrl_loss(toks, mx.array(mask)).item())
    inputs, targets = toks[:, :-1], toks[:, 1:]
    tmask = mx.array(mask[:, 1:]).reshape(-1).astype(mx.float32)
    h = m(inputs)
    per = []
    for d in cfg.mrl_dims:
        lg = m.head_at_dim(h, d)
        Bsz, S, V = lg.shape
        ce = B.ops.cross_entropy(lg.reshape(Bsz * S, V), targets.reshape(Bsz * S), reduction="none")
        per.append(float((mx.sum(ce * tmask) / mx.sum(tmask)).item()))
    expected = (per[0] + per[1]) / 2.0
    assert abs(got - expected) < 1e-3


# ─────────────────────────── MoE: top_k restored on grow (M4) ────────────────


def test_moe_top_k_restored_on_grow():
    from src.core.model.moe import SectorGate

    g = SectorGate(d_model=32, n_experts=1, top_k=2)
    assert g.top_k == 1  # capped to available experts
    g.grow_experts(5)
    assert g.n_experts == 6
    assert g.top_k == 2  # restored toward the configured target
