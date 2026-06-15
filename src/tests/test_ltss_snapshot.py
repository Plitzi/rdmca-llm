"""LTSS store retrieval branches (src/memory/ltss.py) + sector snapshot/catastrophe
manager (src/consolidation/snapshot.py)."""

import numpy as np

from src.consolidation.snapshot import SectorSnapshotManager, _mean, _std
from src.memory.ltss import LTSS, LTSSNode


def _store(tmp_path, dim=8):
    return LTSS(db_path=str(tmp_path / "ltss.db"), emb_dim=dim)


def test_ltss_add_search_get_content(tmp_path):
    s = _store(tmp_path)
    a = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    b = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    s.add(LTSSNode(id="a", embedding=a, content="alpha"))
    s.add(LTSSNode(id="b", embedding=b, content="beta"))
    assert len(s) == 2
    hits = s.search(a, k=1)
    assert hits and hits[0][0] == "a"  # nearest is itself
    assert s.get_content("a") == "alpha"
    assert s.get_content("missing") is None
    assert s.max_cosine_similarity(a) > 0.9
    s.close()
    s.close()  # idempotent


def test_ltss_empty_and_edges_and_centroid(tmp_path):
    s = _store(tmp_path)
    assert s.search(np.ones(8, dtype=np.float32)) == []  # empty store
    assert s.max_cosine_similarity(np.ones(8, dtype=np.float32)) == 0.0
    assert s.global_centroid is None  # property
    s.add(LTSSNode(id="x", embedding=np.ones(8, dtype=np.float32), content="x"))
    s.add(LTSSNode(id="y", embedding=np.zeros(8, dtype=np.float32), content="y"))
    s.add_edge("x", "y", "related", 0.5)
    assert s.global_centroid is not None  # property
    assert s.global_std is not None  # property
    s.close()


def test_snapshot_mean_std_helpers():
    assert _mean([]) == 0.0 and _std([]) == 0.0
    assert _mean([2.0, 4.0]) == 3.0
    assert _std([2.0, 4.0]) > 0.0


def test_detect_catastrophe_triggers_and_freezes(tmp_path):
    m = SectorSnapshotManager(snapshot_dir=str(tmp_path / "snaps"))
    # a BCF delta beyond 0.001 is a catastrophe trigger
    assert (
        m.detect_catastrophe(
            1, benchmark_delta=0.0, kl_divergence=0.0, bcf_delta=0.5, grad_norm=1.0
        )
        is True
    )
    # a fully clean cycle is not
    assert (
        m.detect_catastrophe(
            2, benchmark_delta=0.0, kl_divergence=0.0, bcf_delta=0.0, grad_norm=1.0
        )
        is False
    )
    assert m.is_frozen(2) is False


def test_snapshot_write_and_rollback_without_snapshot(tmp_path):
    m = SectorSnapshotManager(snapshot_dir=str(tmp_path / "snaps"))
    p = m.snapshot_before_update(7, {"w": np.zeros(4, dtype=np.float32)}, cycle_t=1000.0)
    assert p.exists()
    # rolling back a sector with no snapshot returns False (logged), doesn't raise
    assert m.rollback(999, sector_adapter=None) is False
