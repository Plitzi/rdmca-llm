"""Ambiguity handler (src/consolidation/ambiguity.py): score from sector affinities,
the clear/defer/queue decision, the human-review queue + persistence, and expiry."""

import time

import numpy as np

from src.consolidation.ambiguity import MAX_DEFER_CYCLES, AmbiguityHandler
from src.memory.episodic_buffer import Experience


def _handler(tmp_path):
    return AmbiguityHandler(queue_path=str(tmp_path / "queue.jsonl"))


def _exp(text):
    return Experience(text=text, embedding=np.zeros(8, dtype=np.float32))


def test_ambiguity_score():
    h = AmbiguityHandler(queue_path="logs/human_queue.jsonl")
    assert h.ambiguity_score([]) == 1.0
    assert abs(h.ambiguity_score([(1, 0.9), (2, 0.4)]) - 0.1) < 1e-9


def test_handle_clear(tmp_path):
    h = _handler(tmp_path)
    assert h.handle(_exp("x"), [(1, 0.95)], cycle_id="c0") == "clear"


def test_handle_defer_then_queue_after_max(tmp_path):
    h = _handler(tmp_path)
    exp = _exp("ambiguous content")
    mid = [(1, 0.5), (2, 0.45)]  # score 0.5 → defer band
    for _ in range(MAX_DEFER_CYCLES):
        assert h.handle(exp, mid, cycle_id="c") == "defer"
    # once defer_count hits the cap, the same mid-ambiguity escalates to the queue
    assert h.handle(exp, mid, cycle_id="c") == "queue"
    assert h.pending_count() == 1


def test_handle_direct_queue_on_high_ambiguity(tmp_path):
    h = _handler(tmp_path)
    assert h.handle(_exp("y"), [(1, 0.1)], cycle_id="c") == "queue"
    # persisted to disk
    assert (tmp_path / "queue.jsonl").read_text().strip()


def test_queue_for_review_direct(tmp_path):
    h = _handler(tmp_path)
    h.queue_for_review(_exp("z"), score=0.8, rationale="low confidence")
    assert h.pending_count() == 1


def test_expire_old_entries(tmp_path):
    h = _handler(tmp_path)
    h.queue_for_review(_exp("old"), score=0.9)
    h._queue[0].added_at = time.time() - 999 * 86400  # force-age it
    assert h.expire_old_entries() == 1
    assert h.pending_count() == 0
