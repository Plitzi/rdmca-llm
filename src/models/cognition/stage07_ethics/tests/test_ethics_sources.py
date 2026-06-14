"""
Stage 7 (ethics / BCF) data-source guards.

Synthetic ethics fills its budget and honors the requested languages. Lives with the
stage so deleting the stage takes its test too.
"""

import itertools

from src.models.cognition.stage07_ethics.sources import gen_ethics


def test_gen_ethics_respects_langs():
    en = list(itertools.islice(gen_ethics(50, langs=["en"]), 50))
    assert en and all(r["lang"] == "en" for r in en)
    both = {r["lang"] for r in itertools.islice(gen_ethics(200, langs=["en", "es"]), 200)}
    assert both == {"en", "es"}
