"""Pure SDK helpers shared by stages: readability/hash filters (textfilter) and the
hermes tool-call parser (agentic). Both are deterministic and model-free."""

import json

from src.plugins.sdk import agentic, textfilter


# ─────────────────────────── textfilter ──────────────────────────────────────
def test_stable_hash_is_deterministic_and_hex():
    h1 = textfilter.stable_hash("hello world")
    h2 = textfilter.stable_hash("hello world")
    assert h1 == h2 and h1 != textfilter.stable_hash("other")
    int(h1, 16)  # valid hex


def test_syllable_count_basic():
    assert textfilter.syllable_count("cat") == 1
    assert textfilter.syllable_count("banana") >= 3


def test_flesch_kincaid_grade_orders_simple_below_complex():
    simple = textfilter.flesch_kincaid_grade("The cat sat on the mat. It is fun.")
    complex_ = textfilter.flesch_kincaid_grade(
        "Constitutional jurisprudence necessitates meticulous deliberation regarding precedent."
    )
    assert complex_ > simple


def test_passes_filter_grade_and_word_len_and_none():
    assert textfilter.passes_filter("The dog ran fast.", None) is True  # no spec → always
    spec = {"max_grade": 4, "max_word_len": 10}
    assert textfilter.passes_filter("The dog ran.", spec) is True
    # an overlong word fails the word-length bound
    assert textfilter.passes_filter("Antidisestablishmentarianism today.", spec) is False


# ─────────────────────────── agentic (hermes) ────────────────────────────────
_TOOLS = [
    {
        "type": "function",
        "function": {"name": "get_time", "description": "now", "parameters": {"type": "object"}},
    }
]
_EX = {
    "tools": _TOOLS,
    "conversations": [
        {"from": "human", "value": "what time is it?"},
        {
            "from": "gpt",
            "value": 'let me check <tool_call>{"name": "get_time", "arguments": {}}</tool_call>',
        },
        {"from": "tool", "value": "<tool_response>12:00</tool_response>"},
        {"from": "gpt", "value": "It is 12:00."},
    ],
}


def test_hermes_tools_normalizes_to_claude_shape():
    out = json.loads(agentic.hermes_tools(_TOOLS))
    assert out[0]["name"] == "get_time" and "input_schema" in out[0]
    assert agentic.hermes_tools("not json") is None
    assert agentic.hermes_tools([]) is None


def test_hermes_events_yields_ordered_events():
    _tools, events = agentic.hermes_events(_EX)
    kinds = [k for k, _ in events]
    assert kinds == ["user", "assistant", "call", "result", "assistant"]
    call = next(p for k, p in events if k == "call")
    assert call["name"] == "get_time"


def test_hermes_to_transcript_requires_a_call():
    txt = agentic.hermes_to_transcript(_EX)
    assert txt and "Action:" in txt and "Observation: 12:00" in txt
    # an example with no tool call → None
    no_call = {
        "conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello"}]
    }
    assert agentic.hermes_to_transcript(no_call) is None
