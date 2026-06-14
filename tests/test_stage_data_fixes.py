"""
Tests for the Level-1 stages 3–7 training-failure fixes:

  - the empty-corpus loader guard (an empty *.val.jsonl must NOT hang next_batch);
  - synthetic fill generators (arithmetic / causal / CoT / ethics) producing valid,
    format-correct, level-graded records — esp. CoT closing its <think> block;
  - `_blend` staying robust when the real seed is empty (offline);
  - size-weighted rehearsal selection favouring the largest earlier stage.
"""

import sys
from pathlib import Path
from typing import ClassVar

sys.path.insert(0, str(Path(__file__).parent.parent))

import itertools

import numpy as np
import pytest

from src.agent import THINK_CLOSE, THINK_OPEN, visible_stream_text
from src.data import graded as g
from src.data.loader import DataLoader, TextDataset


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


# ── 1b. clean conversational providers parse into quality transcripts ────────
def test_dailydialog_extractor_builds_quality_transcript():
    """The DailyDialog corpus entry must turn alternating utterances into a clean
    User:/Assistant: exchange that passes the conversational-quality gate — and be
    exception-free on a missing/renamed schema (so a bad mirror is just empty)."""
    from src.data.textnorm import conversational_quality_ok

    extractors = dict(g._DIALOGUE_CORPORA["en"])
    assert "roskoN/dailydialog" in extractors
    dd = extractors["roskoN/dailydialog"]
    turns = dd({"utterances": ["Hi there!", "Hello! How are you?", "Good, thanks."]})
    assert turns == [(0, "Hi there!"), (1, "Hello! How are you?"), (0, "Good, thanks.")]
    text = g._format_dialogue(turns)
    assert text.startswith("User: Hi there!\nAssistant: Hello!")
    assert conversational_quality_ok(text)
    assert dd({}) == [] and dd({"dialog": []}) == []  # schema-miss safe, no raise


# ── 2. synthetic CoT closes its <think> block and gives an answer ────────────
def test_gen_cot_closes_think_and_answers():
    for rec in itertools.islice(g.gen_cot(200, seed=7), 200):
        t = rec["text"]
        assert THINK_OPEN in t and THINK_CLOSE in t  # block opened AND closed
        assert t.index(THINK_OPEN) < t.index(THINK_CLOSE)  # in the right order
        assert "The answer is" in t.split(THINK_CLOSE, 1)[1]  # answer AFTER the block
        # think OFF would show exactly the answer line — never empty, never the scratchpad
        vis = visible_stream_text(t)
        assert vis.strip().startswith("The answer is")
        assert THINK_OPEN not in vis


# ── 3. arithmetic is level-graded (L1 = single-digit, never negative) ────────
def test_gen_arithmetic_level1_single_digit_nonneg():
    import re

    eq = re.compile(r"^(\d+) ([+\-]) (\d+) = (-?\d+)$")
    seen_eq = 0
    for rec in itertools.islice(g.gen_arithmetic(2000, level=1, seed=3), 2000):
        t = rec["text"]
        m = eq.match(t)
        if not m:  # counting / comparison / worded / Q&A
            continue
        a, op, b, c = int(m[1]), m[2], int(m[3]), int(m[4])
        assert c >= 0  # never negative
        assert (a + b if op == "+" else a - b) == c  # arithmetic is correct
        # Atomic BORROW primitive '(d+10) - d = …' (the worked subtraction step) is an
        # intentional exception with a 10–18 minuend; graded equations stay single-digit.
        if op == "-" and 10 <= a <= 19 and b < 10:
            continue
        seen_eq += 1
        assert a < 10 and b < 10  # single-digit operands
    assert seen_eq > 0


# ── 4. _blend is pure-synthetic when the real seed is empty (offline) ────────
def test_blend_pure_synthetic_when_real_empty():
    out = list(g._blend(iter([]), g.gen_causal(50), n_examples=20))
    assert len(out) == 20
    assert all(r["text"].strip() for r in out)


def test_blend_interleaves_then_fills():
    real = iter([{"text": f"REAL{i}", "lang": "en"} for i in range(3)])
    out = list(g._blend(real, g.gen_cot(100), n_examples=10))
    assert len(out) == 10
    assert sum(r["text"].startswith("REAL") for r in out) == 3  # all real used, spread in


# ── 5. ethics fills budget and respects requested languages ──────────────────
def test_gen_ethics_respects_langs():
    en = list(itertools.islice(g.gen_ethics(50, langs=["en"]), 50))
    assert en and all(r["lang"] == "en" for r in en)
    both = {r["lang"] for r in itertools.islice(g.gen_ethics(200, langs=["en", "es"]), 200)}
    assert both == {"en", "es"}


# ── 5b. graded dictionary: meanings, valid format, per-level vocab growth ────
def test_gen_definitions_are_well_formed():
    for rec in itertools.islice(g.gen_definitions(150, level=1, seed=4), 150):
        t = rec["text"]
        assert rec["lang"] == "en" and t.strip()
        # every entry is either a definition statement or a "what does X mean?" Q&A
        assert " is " in t or " means " in t
        if t.startswith("User:"):
            assert "\nAssistant:" in t  # Q&A has an answer turn


def test_definitions_vocabulary_grows_per_level():
    """A level includes every tier ≤ its number, so higher levels DEFINE more words
    (vocabulary + definitions grow per level — the user's requirement)."""

    def defined_words(level):
        seen = set()
        for rec in itertools.islice(g.gen_definitions(4000, level=level, seed=1), 4000):
            # the headword is the subject: 'A <w> is', 'To <w> means', '<W> means'
            for w in list(g._DICT_TIER1) + list(g._DICT_TIER2):
                if (
                    f" {w} is " in rec["text"]
                    or f"to {w} means" in rec["text"].lower()
                    or rec["text"].lower().startswith(f"{w} means")
                    or f"is {w}?" in rec["text"]
                    or f"to {w}?" in rec["text"]
                    or f"'{w}'" in rec["text"]
                ):
                    seen.add(w)
        return seen

    l1, l2 = defined_words(1), defined_words(2)
    assert set(g._DICT_TIER1).issubset(l1)  # level 1 defines tier-1 words
    assert l1.isdisjoint(g._DICT_TIER2)  # but NONE of tier-2
    assert set(g._DICT_TIER2) & l2  # level 2 adds tier-2 words
    assert len(l2) > len(l1)  # vocabulary grew


# ── 5c. clean everyday-conversation anchor (basic_chat) ──────────────────────
def test_gen_basic_chat_clean_qa_and_cycles_to_budget():
    """basic_chat is clean User/Assistant Q&A, and a SMALL unique set CYCLED to fill
    a budget (controlled clean repetition — the fluency anchor)."""
    sample = list(itertools.islice(g.gen_basic_chat(40), 40))
    for r in sample:
        assert r["text"].startswith("User: ") and "\nAssistant: " in r["text"]
        assert r["lang"] == "en"
    # greeting must map to a greeting-style reply, never an apology ("hi"→"I'm sorry")
    greet = [
        r["text"]
        for r in itertools.islice(g.gen_basic_chat(3000), 3000)
        if r["text"].startswith("User: Hi\n")
    ]
    assert greet and all("sorry" not in t.lower() for t in greet)
    # cycles: far more records than unique exchanges, but bounded unique count
    big = list(itertools.islice(g.gen_basic_chat(4000), 4000))
    assert len(big) == 4000 and 20 < len({r["text"] for r in big}) < 400


def test_cycle_records_fills_n_and_handles_empty():
    recs = [{"text": f"r{i}", "lang": "en"} for i in range(5)]
    out = list(g._cycle_records(list(recs), 17))
    assert len(out) == 17  # cycles 5 → 17
    assert list(g._cycle_records([], 10)) == []  # empty never hangs


# ── 6. rehearsal selection is weighted by corpus size ────────────────────────
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
