"""
Phase 1 Acceptance Tests — Core Text Model
Run after all 5 curriculum stages complete and the foundational core is frozen.
All tests must pass before beginning Phase 2.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import mlx.core as mx

from src.model.transformer import RDMCAFoundational, ModelConfig
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
    """MRL loss must decrease over 10 steps on a fixed batch."""
    import mlx.nn as nn
    import mlx.optimizers as optim

    opt   = optim.AdamW(learning_rate=1e-3)
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
    """Truncating to 128 dims must produce valid (finite) logits."""
    batch   = mx.array(np.random.randint(0, 32000, (1, 32)))
    h       = model(batch)
    h_small = h[..., :128]
    head    = model.heads[1]   # 128-dim head
    logits  = head(h_small)
    arr     = np.array(logits.tolist())
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
    """RE must score an experience in < 5ms."""
    import time
    from src.memory.episodic_buffer import Experience
    from src.memory.ltss import LTSS

    ltss = LTSS(db_path=":memory:")
    re   = RelevanceEngine(ltss=ltss)

    exp = Experience(
        text="Hello world",
        embedding=np.random.randn(256).astype(np.float32),
    )
    exp.episodic_context = []

    start = time.perf_counter()
    for _ in range(100):
        re.score(exp)
    elapsed_ms = (time.perf_counter() - start) / 100 * 1000
    assert elapsed_ms < 5.0, f"RE latency {elapsed_ms:.2f}ms > 5ms"


def test_re_novelty():
    """Highly novel experience must score N > 0.7."""
    from src.relevance.engine import novelty
    e = np.random.randn(256).astype(np.float32)
    s = -e   # opposite direction → max novelty
    assert novelty(e, s) > 0.7


# ---------------------------------------------------------------------------
# Sector isolation (placeholder — requires LoRA integration)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="requires trained sectors — run after Phase 1 complete")
def test_sector_isolation():
    """S1 update must not change S2-S7 parameter checksums."""
    pass


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
