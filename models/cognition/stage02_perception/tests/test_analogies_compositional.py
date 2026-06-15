"""
Compositional analogies (stage 2) — LEARN the relation, don't memorize tuples.

The old generator had 8 fixed analogy tuples → pure memorization. The new one samples two
instances of the SAME relation, so the model sees each relation generalize across many
pairs. Guards: every analogy's two pairs share one relation, the generated set is far more
varied than a fixed tuple list, and the completion Q&A form teaches the analogy skill.
"""

import re

from models.cognition.stage02_perception.sources import _ANALOGY_RELATIONS, gen_analogies


def _pair_to_relation():
    """(a,b) -> relation name, to verify both halves of an analogy share a relation."""
    idx = {}
    for rel, pairs in _ANALOGY_RELATIONS.items():
        for p in pairs:
            idx[p] = rel
    return idx


def test_analogies_are_relation_consistent_and_varied():
    rows = list(gen_analogies(1500, seed=5))
    idx = _pair_to_relation()
    seen_analogies, seen_relations, completions = set(), set(), 0
    for r in rows:
        t = r["text"]
        m = re.search(r"([a-z]+) is to ([a-z]+) as ([a-z]+) is to (?:what\?|([a-z]+))", t)
        if not m:
            continue  # numeric pattern
        a, b, c = m[1], m[2], m[3]
        d = m[4]
        if d is None:  # completion Q&A: answer on next line
            d = re.search(r"Assistant: ([a-z]+)\.", t)[1]
            completions += 1
        rel_ab, rel_cd = idx.get((a, b)), idx.get((c, d))
        assert rel_ab is not None and rel_ab == rel_cd, f"pairs not same relation: {t}"
        seen_analogies.add((a, b, c, d))
        seen_relations.add(rel_ab)

    # Combinatorial variety: far beyond the old 8 fixed tuples, spanning most relations.
    assert len(seen_analogies) > 100, f"too few distinct analogies: {len(seen_analogies)}"
    assert len(seen_relations) >= 6, "should span many relation types"
    assert completions > 0, "the completion (skill) form must appear"


def test_each_relation_supports_generalization():
    """Every relation has enough instances that held-out OOD pairs are possible (you can
    train on some pairs and test on unseen ones) — the basis for an OOD generalization
    probe, impossible with a single fixed tuple per relation."""
    for rel, pairs in _ANALOGY_RELATIONS.items():
        assert len(pairs) >= 8, f"{rel} too small to train+hold-out for generalization"
        assert len(set(pairs)) == len(pairs), f"{rel} has duplicate instance pairs"
