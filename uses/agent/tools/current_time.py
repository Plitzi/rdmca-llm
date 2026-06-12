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
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.agent import Tool


def _get_current_time(inp: dict) -> dict:
    tz = str(inp.get("timezone", "")).strip()
    if tz.lower() in ("", "utc"):
        now = datetime.now(timezone.utc)
    else:
        try:                                    # IANA name, e.g. "America/New_York"
            now = datetime.now(ZoneInfo(tz))
        except (ZoneInfoNotFoundError, ValueError):
            now = datetime.now()                # unknown zone → machine local time
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
            "timezone": {"type": "string", "description": "Optional; 'utc' (default), "
                         "'local', or an IANA name like 'America/New_York'."}
        },
        "required": [],
    },
    run=_get_current_time,
)
