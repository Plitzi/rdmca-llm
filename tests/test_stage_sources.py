"""
Stage source-resolution tests — the registry's stream_source must own the source
keys the old graded.stream_source dispatcher did, route each to the right stage, and
yield {'text','lang'} records. Only offline (synthetic) keys are exercised so the
test never touches the network; the real-corpus keys are checked for ownership only.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stages import owns_source, stream_source

# key -> owning stage number (mirrors the old dispatcher's routing).
KEY_OWNER = {
    "tinystories": 1,
    "instruct": 1,
    "instructions": 1,
    "dialogue": 1,
    "daily_dialog": 1,
    "simple_wikipedia": 1,
    "definitions": 1,
    "dictionary": 1,
    "grammar": 1,
    "basic_chat": 1,
    "smalltalk": 1,
    "analogies": 2,
    "arithmetic": 3,
    "causal": 4,
    "causal_synth": 4,
    "reasoning": 5,
    "cot": 5,
    "memory": 6,
    "memory_synth": 6,
    "ethics": 7,
    "agentic": 8,
    "tools": 8,
    "mcp": 9,
    "skills": 10,
}

# Offline synthetic keys (no dataset download) we can fully consume in a test.
OFFLINE_KEYS = [
    "definitions",
    "dictionary",
    "grammar",
    "basic_chat",
    "analogies",
    "memory",
    "ethics",
]


def test_every_known_key_is_owned_by_expected_stage():
    for key, number in KEY_OWNER.items():
        plugin = owns_source(key)
        assert plugin is not None, f"no stage owns source key {key!r}"
        assert plugin.number == number, f"{key!r} owned by stage {plugin.number}, expected {number}"


def test_unknown_key_returns_none():
    assert stream_source("does_not_exist", langs=["en"], n_tokens=10_000) is None


def test_offline_sources_yield_text_lang_records():
    for key in OFFLINE_KEYS:
        it = stream_source(key, langs=["en"], n_tokens=12_000, arithmetic_level=1)
        assert it is not None, f"{key!r} resolved to None"
        rec = next(iter(it))
        assert isinstance(rec.get("text"), str) and rec["text"].strip()
        assert isinstance(rec.get("lang"), str)


def test_ethics_blends_without_real_seed_offline():
    # No extra_streamers → the real ethics seed is empty; the builder must still
    # produce synthetic records (the _blend robustness the dispatcher relied on).
    recs = []
    for rec in stream_source("ethics", langs=["en", "es"], n_tokens=6_000):
        recs.append(rec)
        if len(recs) >= 5:
            break
    assert len(recs) == 5 and all(r["text"].strip() for r in recs)
