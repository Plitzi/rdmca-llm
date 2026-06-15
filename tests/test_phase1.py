"""
Phase 1 Acceptance Tests — Core Text Model
Run after all 5 curriculum stages complete and the foundational core is frozen.
All tests must pass before beginning Phase 2.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import numpy as np
import pytest

from src.model.transformer import ModelConfig, RDMCAFoundational
from src.relevance.engine import RelevanceEngine

# ---------------------------------------------------------------------------
# Model sanity
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model():
    cfg = ModelConfig()
    return RDMCAFoundational(cfg)


def test_model_forward_shape(model):
    batch = mx.array(np.random.randint(0, 32000, (2, 64)))
    h = model(batch)
    assert h.shape == (2, 64, 256), f"unexpected shape {h.shape}"


def test_mrl_loss_decreases(model):
    """MRL loss must decrease over 10 steps on a fixed batch. Seeded so the
    random init + batch are deterministic (no stochastic flakiness in CI)."""
    import mlx.nn as nn
    import mlx.optimizers as optim

    np.random.seed(0)
    mx.random.seed(0)
    opt = optim.AdamW(learning_rate=1e-3)
    batch = mx.array(np.random.randint(1, 32000, (4, 33)))

    loss_and_grad = nn.value_and_grad(model, lambda m, t: m.mrl_loss(t))
    losses = []
    for _ in range(10):
        loss, grads = loss_and_grad(model, batch)
        mx.eval(loss)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        losses.append(loss.item())

    assert losses[-1] < losses[0], "loss did not decrease"


def test_mrl_prefix_valid(model):
    """Truncating to 128 dims via the shared head must produce finite logits."""
    batch = mx.array(np.random.randint(0, 32000, (1, 32)))
    h = model(batch)
    logits = model.head_at_dim(h, 128)  # shared-head prefix projection
    arr = np.array(logits.tolist())
    assert np.all(np.isfinite(arr)), "non-finite logits at 128-dim prefix"


def test_memory_footprint(model):
    """Training pass must stay under 8 GB (rough check via param count)."""
    n = model.count_params()
    # BF16: 2 bytes/param + optimizer states ~3x → ~6x total
    estimated_gb = n * 2 * 6 / 1e9
    assert estimated_gb < 8.0, f"estimated memory {estimated_gb:.2f} GB > 8 GB"


# ---------------------------------------------------------------------------
# Relevance Engine
# ---------------------------------------------------------------------------


def test_re_latency():
    """RE scoring must be fast. Threshold is generous (25ms avg over 100 calls)
    so the test asserts 'not pathologically slow' without flaking on a loaded
    machine / cold CI — a real regression makes scoring orders slower, not 2x."""
    import time

    from src.memory.episodic_buffer import Experience
    from src.memory.ltss import LTSS

    ltss = LTSS(db_path=":memory:")
    re = RelevanceEngine(ltss=ltss)

    exp = Experience(
        text="Hello world",
        embedding=np.random.randn(256).astype(np.float32),
    )
    exp.episodic_context = []

    start = time.perf_counter()
    for _ in range(100):
        re.score(exp)
    elapsed_ms = (time.perf_counter() - start) / 100 * 1000
    assert elapsed_ms < 25.0, f"RE latency {elapsed_ms:.2f}ms > 25ms"


def test_re_novelty():
    """Highly novel experience must score N > 0.7."""
    from src.relevance.engine import novelty

    e = np.random.randn(256).astype(np.float32)
    s = -e  # opposite direction → max novelty
    assert novelty(e, s) > 0.7


# ---------------------------------------------------------------------------
# Sector wiring + isolation
# ---------------------------------------------------------------------------


def _small_model():
    cfg = ModelConfig(
        vocab_size=512, d_model=64, n_layers=2, n_heads=2, ffn_dim=128, mrl_dims=[32, 64]
    )
    return RDMCAFoundational(cfg)


def test_sector_zero_output_init():
    """Attaching zero-init sectors must not change logits (Guide §1.6.2)."""
    from src.model.lora import build_all_sectors

    m = _small_model()
    m.train(False)
    batch = mx.array(np.random.randint(1, 512, (2, 16)))
    base = np.array(m.logits(batch).tolist())
    m.attach_sectors(build_all_sectors(d_model=64, n_layers=2))
    m.set_active_sectors([(1, 1.0), (2, 1.0)])
    after = np.array(m.logits(batch).tolist())
    assert np.allclose(base, after, atol=1e-5), "sectors changed init output"


def test_sector_isolation():
    """An update to S1 must leave the core and S2-S7 bit-identical (§1.6.1)."""
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    from src.model.lora import build_all_sectors, masked_sector_update

    m = _small_model()
    m.attach_sectors(build_all_sectors(d_model=64, n_layers=2))
    batch = mx.array(np.random.randint(1, 512, (4, 17)))

    before = {k: np.array(v.tolist()) for k, v in tree_flatten(m.parameters())}

    def loss_fn(model):
        model.set_active_sectors([(1, 1.0)])
        return model.mrl_loss(batch)

    _loss, _gnorm = masked_sector_update(m, 1, loss_fn, optim.SGD(learning_rate=0.1))
    after = {k: np.array(v.tolist()) for k, v in tree_flatten(m.parameters())}

    s1_changed = any("sectors.1." in k and not np.array_equal(before[k], after[k]) for k in before)
    others_changed = any(
        ("sectors." in k and "sectors.1." not in k) and not np.array_equal(before[k], after[k])
        for k in before
    )
    core_changed = any(
        "sectors." not in k and not np.array_equal(before[k], after[k]) for k in before
    )

    assert s1_changed, "S1 did not update"
    assert not others_changed, "another sector changed during S1 update"
    assert not core_changed, "foundational core changed during S1 update"


@pytest.mark.skip(reason="requires trained foundational checkpoint")
def test_blim_accuracy():
    """BLiMP grammaticality >= 70%."""
    pass


@pytest.mark.skip(reason="requires trained foundational checkpoint")
def test_gsm8k_accuracy():
    """GSM8K accuracy >= 15%."""
    pass


@pytest.mark.skip(reason="requires trained foundational checkpoint")
def test_bcf_accuracy():
    """BCF probe set accuracy >= 90%."""
    pass
