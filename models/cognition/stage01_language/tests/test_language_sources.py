"""
Stage 1 (language) data-source guards.

Clean conversational providers parse into quality transcripts; the graded dictionary
produces well-formed definitions whose vocabulary GROWS per level; and basic_chat is
clean User/Assistant Q&A cycled to fill a budget (the fluency anchor — a greeting maps
to a greeting, never an apology). Lives with the stage so deleting it removes its test.
"""

import itertools

from models.cognition.stage01_language.dictionary import DICT_TIER1, DICT_TIER2
from models.cognition.stage01_language.sources import (
    _DIALOGUE_CORPORA,
    _format_dialogue,
    gen_basic_chat,
    gen_definitions,
)


def test_dailydialog_extractor_builds_quality_transcript():
    """The DailyDialog corpus entry must turn alternating utterances into a clean
    User:/Assistant: exchange that passes the conversational-quality gate — and be
    exception-free on a missing/renamed schema (so a bad mirror is just empty)."""
    from src.data.textnorm import conversational_quality_ok

    extractors = dict(_DIALOGUE_CORPORA["en"])
    assert "roskoN/dailydialog" in extractors
    dd = extractors["roskoN/dailydialog"]
    turns = dd({"utterances": ["Hi there!", "Hello! How are you?", "Good, thanks."]})
    assert turns == [(0, "Hi there!"), (1, "Hello! How are you?"), (0, "Good, thanks.")]
    text = _format_dialogue(turns)
    assert text.startswith("User: Hi there!\nAssistant: Hello!")
    assert conversational_quality_ok(text)
    assert dd({}) == [] and dd({"dialog": []}) == []  # schema-miss safe, no raise


def test_gen_definitions_are_well_formed():
    for rec in itertools.islice(gen_definitions(150, level=1, seed=4), 150):
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
        for rec in itertools.islice(gen_definitions(4000, level=level, seed=1), 4000):
            # the headword is the subject: 'A <w> is', 'To <w> means', '<W> means'
            for w in list(DICT_TIER1) + list(DICT_TIER2):
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
    assert set(DICT_TIER1).issubset(l1)  # level 1 defines tier-1 words
    assert l1.isdisjoint(DICT_TIER2)  # but NONE of tier-2
    assert set(DICT_TIER2) & l2  # level 2 adds tier-2 words
    assert len(l2) > len(l1)  # vocabulary grew


def test_gen_basic_chat_clean_qa_and_cycles_to_budget():
    """basic_chat is clean User/Assistant Q&A, and a SMALL unique set CYCLED to fill
    a budget (controlled clean repetition — the fluency anchor)."""
    sample = list(itertools.islice(gen_basic_chat(40), 40))
    for r in sample:
        assert r["text"].startswith("User: ") and "\nAssistant: " in r["text"]
        assert r["lang"] == "en"
    # greeting must map to a greeting-style reply, never an apology ("hi"→"I'm sorry")
    greet = [
        r["text"]
        for r in itertools.islice(gen_basic_chat(3000), 3000)
        if r["text"].startswith("User: Hi\n")
    ]
    assert greet and all("sorry" not in t.lower() for t in greet)
    # cycles: far more records than unique exchanges, but bounded unique count
    big = list(itertools.islice(gen_basic_chat(4000), 4000))
    assert len(big) == 4000 and 20 < len({r["text"] for r in big}) < 400
