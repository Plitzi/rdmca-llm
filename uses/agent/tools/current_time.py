"""Example tool: get the current date/time.

A good example tool is one the model genuinely CANNOT answer on its own — the
real wall-clock time is exactly that (unlike, say, arithmetic, which the model
learns in stage 3 and should do itself). This keeps tool-use testing separate
from arithmetic testing.

Self-contained (no network). To add your own tool, copy this file: expose a
`TOOL` of type `src.agent.Tool` with a name, description, JSON input_schema and a
`run(input: dict)` function returning a JSON-serializable result.
"""
from __future__ import annotations
from datetime import datetime, timezone

from src.agent import Tool


def _get_current_time(inp: dict) -> dict:
    tz = str(inp.get("timezone", "")).strip().lower()
    now = datetime.now(timezone.utc) if tz in ("", "utc") else datetime.now()
    return {
        "iso": now.isoformat(timespec="seconds"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": now.strftime("%A"),
    }


TOOL = Tool(
    name="get_current_time",
    description="Get the current date and time (the model cannot know this on its own).",
    input_schema={
        "type": "object",
        "properties": {
            "timezone": {"type": "string", "description": "Optional; 'utc' or 'local'."}
        },
        "required": [],
    },
    run=_get_current_time,
)
