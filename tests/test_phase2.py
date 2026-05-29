"""
Phase 2 Acceptance Tests — Memory & Safety Systems
Run after Phase 2 implementation is complete.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.memory.episodic_buffer import EpisodicBuffer, Experience
from src.memory.ltss import LTSS, LTSSNode
from src.memory.mrf import mrf, THETA_RETAIN


@pytest.fixture
def ltss(tmp_path):
    return LTSS(db_path=str(tmp_path / "test.db"), emb_dim=256)


def _make_exp(text="test", emb=None):
    return Experience(
        text=text,
        embedding=emb if emb is not None else np.random.randn(256).astype(np.float32),
    )


def test_ltss_persistence(tmp_path):
    """LTSS must survive process restart (re-load from disk)."""
    db = str(tmp_path / "ltss.db")
    store = LTSS(db_path=db, emb_dim=256)
    emb = np.random.randn(256).astype(np.float32)
    store.add(LTSSNode(id="n1", embedding=emb, content="test", modality="text"))
    assert len(store) == 1

    # Simulate restart
    store2 = LTSS(db_path=db, emb_dim=256)
    rows = store2._conn.execute("SELECT id FROM ltss_nodes").fetchall()
    assert len(rows) == 1, "node lost after reload"


def test_ltss_search(ltss):
    """Search must return correct nearest neighbor."""
    base = np.ones(256, dtype=np.float32)
    ltss.add(LTSSNode(id="a", embedding=base, content="a"))
    ltss.add(LTSSNode(id="b", embedding=-base, content="b"))

    results = ltss.search(base, k=1)
    assert results[0][0] == "a", "wrong nearest neighbor"


def test_mrf_frequent_retain(ltss):
    """Experience retrieved 10x over 3 days must stay above THETA_RETAIN."""
    exp = _make_exp()
    exp.retrieval_count = 10
    exp.age_days = 3.0

    fate = mrf(exp, relevance_score=0.6, ltss=ltss)
    assert fate in ("retain", "promote"), f"expected retain/promote, got {fate}"


def test_mrf_stale_expire(ltss):
    """Routine experience with 0 retrievals and old timestamp must expire."""
    import time
    exp = _make_exp()
    exp.retrieval_count = 0
    exp.age_days = 10.0
    exp.timestamp = time.time() - 10 * 86400

    fate = mrf(exp, relevance_score=0.2, ltss=ltss)
    assert fate == "expire", f"expected expire, got {fate}"


@pytest.mark.skip(reason="requires rollback system wired to real sector adapters")
def test_rollback_integrity():
    """Rollback must restore bit-identical parameters."""
    pass


@pytest.mark.skip(reason="requires BCF + consolidation pipeline")
def test_bcf_adversarial_routing():
    """Adversarial experiences must land in adversarial buffer, never in LTSS."""
    pass
