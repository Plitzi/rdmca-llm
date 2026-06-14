"""Example tool: a to-do list for planning (Claude Code's TodoWrite, in miniature).

When a task is multi-step, a model that plans well first writes down the steps,
then works through them. This tool lets the agent record and update that plan as
structured state — so planning is observable and the model can track progress
instead of holding everything in the scratchpad.

It is just an *example* of a planning aid the agent uses **when available**: the
reasoning stage (stage 5) is trained on transcripts both with and without a todo
tool, so the model learns to reach for one only if the session offers it.

Self-contained (in-memory, per process). `run(input)` replaces the list with the
given `items` (or updates statuses) and returns the current plan.
"""

from __future__ import annotations

from typing import Any

from src.agent import Tool

_VALID = ("pending", "in_progress", "done")


def make_todo_tool() -> Tool:
    """Build a fresh todo tool with its OWN plan state. Each call is independent —
    a server handling concurrent sessions should make one per session rather than
    share the module singleton (whose state would otherwise leak across runs)."""
    plan: list[dict] = []  # closure-local — not a module global

    def _todo(inp: dict) -> dict[str, Any]:
        """Set or update the plan. `items` is a list of step strings or
        {content, status} dicts; an empty/omitted `items` just returns the plan."""
        items = inp.get("items")
        if isinstance(items, list):
            plan.clear()
            for it in items:
                if isinstance(it, dict):
                    content = str(it.get("content", it.get("step", ""))).strip()
                    status = str(it.get("status", "pending")).lower()
                else:
                    content, status = str(it).strip(), "pending"
                if content:
                    plan.append(
                        {"content": content, "status": status if status in _VALID else "pending"}
                    )
        return {"plan": list(plan), "remaining": sum(1 for s in plan if s["status"] != "done")}

    return Tool(
        name="todo",
        description=(
            "Record or update a short plan of steps for a multi-step task. "
            "Use it when planning, before acting. Each item has content and an "
            "optional status (pending|in_progress|done)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "description": "The ordered plan steps.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "status": {"type": "string", "enum": list(_VALID)},
                        },
                        "required": ["content"],
                    },
                }
            },
            "required": [],
        },
        run=_todo,
    )


# Default singleton for the single-process CLI (one plan per process).
TOOL = make_todo_tool()
