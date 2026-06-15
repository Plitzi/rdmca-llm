"""Parser for NousResearch/hermes-function-calling-v1, shared by the tool-use,
MCP and reasoning stages — they all reframe the SAME real tool interactions into
different wire formats, so the parse-once step lives here."""

from __future__ import annotations

import json
import re

# Instruction shown to the model on how to call a tool (Claude-style Action loop).
AGENTIC_SYSTEM_PROMPT = (
    "You can use tools. To call one, output a line "
    'Action: {"name": <tool>, "input": {<args>}} and you will then '
    "receive an Observation with the result; otherwise answer directly."
)
_TOOLCALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TOOLRESP_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)


def hermes_tools(raw) -> str | None:
    """Normalize hermes `tools` to a compact JSON array of
    {name, description, input_schema} (Claude tool-definition shape)."""
    try:
        arr = raw if isinstance(raw, list) else json.loads(raw)
    except Exception:
        return None
    out = []
    for tool in arr if isinstance(arr, list) else []:
        fn = tool.get("function", tool) if isinstance(tool, dict) else None
        if fn and fn.get("name"):
            out.append(
                {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {}),
                }
            )
    return json.dumps(out, ensure_ascii=False) if out else None


def hermes_events(ex: dict):
    """Parse one hermes example into (tools_json, events). `events` is a list of
    ('user'|'assistant'|'call'|'result', payload) — shared by the agentic, MCP and
    reasoning serializers."""
    tools = hermes_tools(ex.get("tools"))
    events: list = []
    for turn in ex.get("conversations") or []:
        sender, value = turn.get("from"), (turn.get("value") or "").strip()
        if not value:
            continue
        if sender == "human":
            events.append(("user", " ".join(value.split())))
        elif sender == "gpt":
            text = _TOOLCALL_RE.sub("", value).strip()
            if text:
                events.append(("assistant", " ".join(text.split())))
            for call in _TOOLCALL_RE.findall(value):
                try:
                    obj = json.loads(call)
                except Exception:
                    continue
                events.append(
                    (
                        "call",
                        {
                            "name": obj.get("name"),
                            "input": obj.get("arguments", obj.get("input", {})),
                        },
                    )
                )
        elif sender == "tool":
            match = _TOOLRESP_RE.search(value)
            events.append(("result", " ".join((match.group(1) if match else value).split())))
    return tools, events


def hermes_to_transcript(ex: dict) -> str | None:
    """Serialize one hermes example into the canonical agentic transcript
    (Claude-style Action/Observation). None unless it has ≥1 real tool call."""
    tools, events = hermes_events(ex)
    lines = [f"System: {AGENTIC_SYSTEM_PROMPT}"]
    if tools:
        lines.append(f"Tools: {tools}")
    saw_call = False
    for kind, payload in events:
        if kind == "user":
            lines.append(f"User: {payload}")
        elif kind == "assistant":
            lines.append(f"Assistant: {payload}")
        elif kind == "call":
            lines.append(f"Action: {json.dumps(payload, ensure_ascii=False)}")
            saw_call = True
        elif kind == "result":
            lines.append(f"Observation: {payload}")
    return "\n".join(lines) if saw_call and len(lines) >= 4 else None
