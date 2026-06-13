"""
Compositional grammar generation (stage 1) — teach HOW LANGUAGE WORKS (rules + word usage),
not just vocabulary. Rules are stated and applied across many words so the model learns the
RULE (and generalizes), and the vocabulary SCALES with level (richer word bank → same rules,
wider words — toward a university-graduate base).

Guards: morphology is CORRECT (curated, never naive +s/+ed), the article a/an rule is right,
many distinct rule types appear, and level widens the part-of-speech vocabulary.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.graded import gen_grammar, _PLURAL_IRREG, _PAST_IRREG, _COMPARATIVE


def test_no_naive_morphology_errors():
    """The classic synthetic-grammar bugs must NOT appear: 'runned', 'gooses', 'a apple',
    'a adjective', verb-synonym in an adjective frame."""
    texts = [r["text"] for r in gen_grammar(2000, level=1, seed=1)]
    blob = "\n".join(texts)
    for bad in ["runned", "goed", "eated", "gooses", "mans", "childs",
                "a apple", "a egg", "a adjective", "a insect", "a elephant",
                "very begin", "very start"]:
        assert bad not in blob, f"grammar produced an error: {bad!r}"


def test_irregulars_are_correct_when_present():
    texts = "\n".join(r["text"] for r in gen_grammar(3000, level=1, seed=2))
    # if an irregular plural/past is mentioned, it must be the correct form
    for sg, pl in _PLURAL_IRREG.items():
        if f"one {sg}," in texts:
            assert f"two {pl}." in texts.split(f"one {sg},")[1][:40] or pl in texts
    for v, p in _PAST_IRREG.items():
        if f"past tense of '{v}'" in texts:
            assert p in texts


def test_many_rule_types_and_completion_form():
    texts = [r["text"] for r in gen_grammar(2000, level=1, seed=3)]
    blob = "\n".join(texts)
    # a spread of distinct grammar faculties is taught
    for marker in ["is a noun", "is a verb", "is an adjective",       # parts of speech
                   "before a vowel", "plural", "past tense",          # a/an, plural, tense
                   "compare two things", "comes before the noun",     # comparative, adjective
                   "subject and a verb", "opposite of"]:              # sentence, antonym
        assert marker in blob, f"missing rule type: {marker}"
    assert any(t.startswith("User:") and "Assistant:" in t for t in texts)   # completion Q&A


def test_vocab_scales_with_level():
    """Higher level draws part-of-speech vocab from more dictionary tiers → a richer word
    set, while teaching the SAME rules (the per-level enrichment the design relies on)."""
    from src.data.graded import _DICT_TIERS
    if len(_DICT_TIERS) < 2:
        return                                   # only one tier defined; nothing to compare
    words = lambda lvl: {w for r in gen_grammar(4000, level=lvl, seed=4)
                         for w in r["text"].replace("'", " ").split()}
    assert len(words(2)) >= len(words(1))
