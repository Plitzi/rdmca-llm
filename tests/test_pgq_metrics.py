"""
PGQ growth metrics (issue C1): the consolidation pipeline must feed PGQ REAL
[0,1] growth signals derived from the cycle, not the old hardcoded 0.0s (which kept
PGQ permanently "stable" so it never grew capacity). We test the metric computation
(`_growth_metrics`) directly with a light stub pipeline, plus the PGQ decision logic.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.core.consolidation import pipeline as P
from src.core.consolidation.pgq import PGQ
from src.core.consolidation.pipeline import ConsolidationPipeline
from src.core.memory.episodic_buffer import Experience


class _StubLTSS:
    """max_cosine_similarity returns a fixed value so we can drive cluster_novel."""

    def __init__(self, coh):
        self._coh = coh

    def max_cosine_similarity(self, emb):
        return self._coh


class _Cfg:  # minimal model.cfg with vocab_size
    vocab_size = 8192


class _Model:
    cfg = _Cfg()


def _pipe(ltss_coh=1.0, last_loss=None, with_model=True):
    """A ConsolidationPipeline shell with only the fields _growth_metrics touches."""
    p = ConsolidationPipeline.__new__(ConsolidationPipeline)  # skip __init__
    p.ltss = _StubLTSS(ltss_coh)
    p.model = _Model() if with_model else None
    p._last_loss = last_loss
    return p


def _exp(rel=0.0):
    e = Experience(text="x", embedding=np.ones(4, dtype=np.float32), sector_assignment=1)
    e.relevance_score = rel
    return e


def test_pred_error_from_loss():
    # pred_error = loss / log(vocab); loss == log(vocab) ⇒ at chance ⇒ 1.0
    p = _pipe(last_loss=float(np.log(8192)))
    gm = p._growth_metrics([_exp()], {"S1": 0.0}, 1)
    assert gm["pred_error"] == pytest.approx(1.0, abs=1e-6)  # at chance
    # half of max entropy → ~0.5
    p_half = _pipe(last_loss=float(np.log(8192)) / 2)
    assert p_half._growth_metrics([_exp()], {}, 1)["pred_error"] == pytest.approx(0.5, abs=1e-6)
    p2 = _pipe(last_loss=0.0)
    assert p2._growth_metrics([_exp()], {}, 1)["pred_error"] == 0.0  # perfect


def test_saturation_from_gradnorm():
    p = _pipe()
    lo = p._growth_metrics([_exp()], {"S3": 0.0}, 3)["saturation"]
    hi = p._growth_metrics([_exp()], {"S3": 50.0}, 3)["saturation"]
    assert lo == pytest.approx(0.0, abs=1e-6)
    assert 0.9 < hi <= 1.0  # saturates toward 1


def test_cluster_novel_fraction():
    far = _pipe(ltss_coh=0.0)  # below PGQ_NOVEL_COH ⇒ all novel
    near = _pipe(ltss_coh=0.99)  # above ⇒ none novel
    exps = [_exp(), _exp(), _exp()]
    assert far._growth_metrics(exps, {}, 1)["cluster_novel"] == 1.0
    assert near._growth_metrics(exps, {}, 1)["cluster_novel"] == 0.0


def test_exc_rate_from_relevance():
    p = _pipe()
    exps = [_exp(rel=0.9), _exp(rel=0.1), _exp(rel=0.6), _exp(rel=0.0)]  # 2 ≥ 0.5
    assert p._growth_metrics(exps, {}, 1)["exc_rate"] == pytest.approx(0.5)


def test_empty_cycle_is_all_zero():
    p = _pipe(last_loss=None)
    gm = p._growth_metrics([], {}, 1)
    assert gm == {"saturation": 0.0, "exc_rate": 0.0, "pred_error": 0.0, "cluster_novel": 0.0}


def test_metrics_can_drive_pgq_off_stable():
    """The whole point of C1: real signals can push GNS past 'stable' and trigger
    growth — impossible when everything was hardcoded to 0."""
    pgq = PGQ()
    # high signals across the board → should NOT be 'stable'
    res = pgq.evaluate(
        "c1",
        saturation=0.9,
        exc_rate=0.9,
        pred_error=0.9,
        cluster_novel=0.9,
        busiest_sector_id=1,
        sectors={},
        model=None,
    )
    assert res.decision != "stable" and res.gns > P.PGQ_SAT_REF * 0  # gns>0
    # all-zero (old behavior) → stable
    res0 = pgq.evaluate(
        "c0",
        saturation=0.0,
        exc_rate=0.0,
        pred_error=0.0,
        cluster_novel=0.0,
        busiest_sector_id=1,
        sectors={},
        model=None,
    )
    assert res0.decision == "stable"
