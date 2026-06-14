"""Stage 9 data sources — Model Context Protocol (MCP).

The SAME real tool interactions (hermes) re-serialized into the MCP wire format
(JSON-RPC 2.0): a tools/list result, then tools/call requests and result messages.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from src.stages._shared.agentic import hermes_events
from src.stages._shared.text import stable_hash

_MCP_SYS = (
    "You interact with tools over MCP (Model Context Protocol, JSON-RPC 2.0): "
    'send {"jsonrpc":"2.0","method":"tools/call","params":{"name","arguments"}} '
    "and receive a matching result message; otherwise answer directly."
)


def _mcp_to_transcript(ex: dict) -> str | None:
    """Serialize one hermes example into an MCP JSON-RPC session. None unless it
    contains at least one real tool call."""
    tools, events = hermes_events(ex)
    lines = [f"System: {_MCP_SYS}"]
    if tools:
        listing = {"jsonrpc": "2.0", "id": 0, "result": {"tools": json.loads(tools)}}
        lines.append("Server: " + json.dumps(listing, ensure_ascii=False))
    request_id, saw_call = 0, False
    for kind, payload in events:
        if kind == "user":
            lines.append(f"User: {payload}")
        elif kind == "assistant":
            lines.append(f"Assistant: {payload}")
        elif kind == "call":
            request_id += 1
            req = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {"name": payload["name"], "arguments": payload["input"]},
            }
            lines.append("Client: " + json.dumps(req, ensure_ascii=False))
            saw_call = True
        elif kind == "result":
            res = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": [{"type": "text", "text": payload}]},
            }
            lines.append("Server: " + json.dumps(res, ensure_ascii=False))
    return "\n".join(lines) if saw_call and len(lines) >= 4 else None


def stream_mcp(langs: list[str], limit_mb: int | None = None) -> Iterator[dict]:
    """Stream real tool interactions as MCP (JSON-RPC 2.0) sessions (EN)."""
    if "en" not in {lang.lower() for lang in langs}:
        return
    from datasets import load_dataset

    try:
        ds = load_dataset("NousResearch/hermes-function-calling-v1", split="train", streaming=True)
    except Exception as e:
        print(f"    [mcp] {e}")
        return
    seen: set = set()
    for ex in ds:
        text = _mcp_to_transcript(ex)
        if not text:
            continue
        h = stable_hash(text)
        if h in seen:
            continue
        seen.add(h)
        yield {"text": text, "lang": "en"}


def _build_mcp(*, langs, limit_mb=None, **_):
    return stream_mcp(langs, limit_mb)


SOURCES = {"mcp": _build_mcp}
