"""Stage 8 data sources — action and tool use.

Real function-calling conversations (hermes) re-serialized into a Claude-style
agentic loop with JSON Action/Observation turns. EN only.
"""

from __future__ import annotations

from collections.abc import Iterator

from src.models.sdk import hermes_to_transcript, stable_hash


def stream_agentic(langs: list[str], limit_mb: int | None = None) -> Iterator[dict]:
    """Stream real agentic tool-use transcripts (EN) as a Claude-style loop."""
    if "en" not in {lang.lower() for lang in langs}:
        return
    from datasets import load_dataset

    try:
        ds = load_dataset("NousResearch/hermes-function-calling-v1", split="train", streaming=True)
    except Exception as e:
        print(f"    [agentic] {e}")
        return
    seen: set = set()
    for ex in ds:
        text = hermes_to_transcript(ex)
        if not text:
            continue
        h = stable_hash(text)
        if h in seen:
            continue
        seen.add(h)
        yield {"text": text, "lang": "en"}


def _build_agentic(*, langs, limit_mb=None, **_):
    return stream_agentic(langs, limit_mb)


SOURCES = {"agentic": _build_agentic, "tools": _build_agentic}
