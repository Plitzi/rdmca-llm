"""
Cross-cutting data-pipeline guards (framework + plugin SDK, no specific stage):

  - the empty-corpus loader guard (an empty *.val.jsonl must NOT hang next_batch);
  - the SDK `blend`/`cycle_records` helpers (synthetic fill, interleave-then-fill,
    cycling and the empty-input guard);
  - size-weighted rehearsal selection favouring the largest earlier stage.

Per-stage source guards (arithmetic / causal / CoT / ethics / language) live with their
stage under models/cognition/stageNN_*/tests/, so deleting a stage takes its tests.
"""

from typing import ClassVar

import pytest

from src.data.loader import DataLoader, TextDataset
from src.plugins.sdk import blend, cycle_records


def _dummy_synth(n: int = 1000):
    """A stage-agnostic synthetic generator for the SDK blend tests — keeps these
    framework tests independent of any stage's data sources."""
    for i in range(n):
        yield {"text": f"SYNTH{i}", "lang": "en"}


# ── 1. empty-corpus loader must terminate, not hang ──────────────────────────
def test_empty_corpus_does_not_hang(tmp_path):
    """A dir whose only data file is EMPTY must yield StopIteration from
    next_batch() instead of spinning forever (the stage-7 hang)."""
    (tmp_path / "empty.jsonl").write_text("")  # zero records

    class _StubTok:  # never called (no records)
        ready = True
        lang_tokens: ClassVar[dict] = {}

        def encode(self, *a, **k):
            return []

        def encode_raw(self, *a, **k):
            return []

        def decode(self, *a, **k):
            return ""

    ds = TextDataset(str(tmp_path), _StubTok(), batch_size=2, seq_len=8)
    loader = DataLoader(ds)
    with pytest.raises(StopIteration):
        loader.next_batch()


# ── 2. blend is pure-synthetic when the real seed is empty (offline) ─────────
def test_blend_pure_synthetic_when_real_empty():
    out = list(blend(iter([]), _dummy_synth(50), n_examples=20))
    assert len(out) == 20
    assert all(r["text"].strip() for r in out)


def test_blend_interleaves_then_fills():
    real = iter([{"text": f"REAL{i}", "lang": "en"} for i in range(3)])
    out = list(blend(real, _dummy_synth(100), n_examples=10))
    assert len(out) == 10
    assert sum(r["text"].startswith("REAL") for r in out) == 3  # all real used, spread in


def test_cycle_records_fills_n_and_handles_empty():
    recs = [{"text": f"r{i}", "lang": "en"} for i in range(5)]
    out = list(cycle_records(list(recs), 17))
    assert len(out) == 17  # cycles 5 → 17
    assert list(cycle_records([], 10)) == []  # empty never hangs


# ── 3. rehearsal selection is weighted by corpus size ────────────────────────
def test_replay_weights_track_corpus_size(tmp_path):
    big, small = tmp_path / "big", tmp_path / "small"
    big.mkdir()
    small.mkdir()
    (big / "d.jsonl").write_text('{"text": "x"}\n' * 1000)
    (small / "d.jsonl").write_text('{"text": "x"}\n')

    class _Stub:
        def __init__(self, files):
            self._files = files

    wb = DataLoader._corpus_bytes(_Stub(list(big.glob("*.jsonl"))))
    ws = DataLoader._corpus_bytes(_Stub(list(small.glob("*.jsonl"))))
    assert wb > ws * 100  # big corpus dominates the draw
