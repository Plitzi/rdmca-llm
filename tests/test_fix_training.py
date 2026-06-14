"""Regression tests — training internals: true grad accumulation, gradient clipping,
optimizer-state warm resume, and backend-surface completeness. (Split from the old
test_fixes.py.)"""

import tempfile
from pathlib import Path

import numpy as np
from fixes_common import B, tiny_model

import src.core.backend as backend

# ─────────────────────────── grad accumulation (C4) ──────────────────────────


def test_accumulate_finalize_is_mean_of_microbatches():
    """finalize(accumulate(g1,g2), 1/2) must equal (g1+g2)/2 ELEMENT-WISE — not
    just differ in norm (a test that only checked the norm would pass even if
    finalize returned g1 unchanged)."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    m = tiny_model()
    lg = B.engine.value_and_grad(m, lambda mm, t: mm.mrl_loss(t))
    b1 = B.ops.array(np.random.randint(0, 256, (3, 33)).astype(np.int64))
    b2 = B.ops.array(np.random.randint(0, 256, (3, 33)).astype(np.int64))
    _, g1 = lg(m, b1)
    _, g2 = lg(m, b2)
    run = B.engine.accumulate_grads(None, g1, m)
    run = B.engine.accumulate_grads(run, g2, m)
    acc = B.engine.finalize_grads(run, 0.5, m)
    # Element-wise: every leaf of `acc` equals the mean of the two micro-batch grads.
    f1, f2, fa = dict(tree_flatten(g1)), dict(tree_flatten(g2)), dict(tree_flatten(acc))
    leaves = [k for k in fa if isinstance(fa[k], mx.array)]
    assert leaves, "no gradient leaves to compare"
    for k in leaves:
        expected = (f1[k].astype(mx.float32) + f2[k].astype(mx.float32)) / 2.0
        d = float(mx.max(mx.abs(fa[k].astype(mx.float32) - expected)).item())
        assert d < 1e-5, f"{k}: accumulated grad is not the mean (max|Δ|={d:.2e})"


def test_clip_grads_caps_norm():
    m = tiny_model()
    lg = B.engine.value_and_grad(m, lambda mm, t: mm.mrl_loss(t))
    b = B.ops.array(np.random.randint(0, 256, (3, 33)).astype(np.int64))
    _, g = lg(m, b)
    clipped = B.engine.clip_grads(m, g, 0.1)
    assert B.engine.grad_norm(m, clipped) <= 0.1 + 1e-3


def test_clip_grads_noop_when_under_threshold():
    m = tiny_model()
    lg = B.engine.value_and_grad(m, lambda mm, t: mm.mrl_loss(t))
    b = B.ops.array(np.random.randint(0, 256, (3, 33)).astype(np.int64))
    _, g = lg(m, b)
    n0 = B.engine.grad_norm(m, g)
    same = B.engine.clip_grads(m, g, 1e9)
    assert abs(B.engine.grad_norm(m, same) - n0) < 1e-4


# ─────────────────────────── backend surface completeness ────────────────────


def test_backend_surface_complete_both():
    import warnings

    from src.core.backend.base import check_surface

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # benign "switching backend" notice
        for name in ("mlx", "torch"):
            try:
                backend.select(name)
            except Exception:
                continue  # backend not installed in this env
            assert check_surface(backend.current()) == []
        backend.select("mlx")  # restore


# ─────────────────────────── optimizer state resume (M1) ─────────────────────


def test_optimizer_state_roundtrip():
    """save_optimizer → load_optimizer restores AdamW moments exactly (warm resume)."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    m = tiny_model()
    opt = B.engine.make_optimizer(m, 5e-4, 0.1)
    lg = B.engine.value_and_grad(m, lambda mm, t: mm.mrl_loss(t))
    for _ in range(4):  # populate optimizer state
        b = B.ops.array(np.random.randint(0, 256, (3, 33)).astype(np.int64))
        loss, g = lg(m, b)
        B.engine.optimizer_step(opt, m, g)
        B.engine.eval(loss)
    with tempfile.TemporaryDirectory() as td:
        p = str(Path(td) / "s.opt")
        B.engine.save_optimizer(opt, p)
        opt2 = B.engine.make_optimizer(m, 5e-4, 0.1)
        assert B.engine.load_optimizer(opt2, p) is True
        s1 = dict(tree_flatten(opt.state))
        s2 = dict(tree_flatten(opt2.state))
        keys = [k for k in s1 if isinstance(s1[k], mx.array) and isinstance(s2.get(k), mx.array)]
        assert keys, "optimizer state should have array leaves"
        for k in keys:
            d = float(mx.max(mx.abs(s1[k].astype(mx.float32) - s2[k].astype(mx.float32))).item())
            assert d < 1e-5
        # absent file → graceful False (cold start)
        assert B.engine.load_optimizer(opt2, str(Path(td) / "missing.opt")) is False
