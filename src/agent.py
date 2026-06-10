"""Agent output formatting — the model can return plain text or JSON, selectable
by the consumer. Today the chat CLI exposes it via `--format` (and the `/format`
command); a serving API would expose the same as a request field. Centralized
here so every consumer stays in sync.

The base model learned both registers: natural language (stages 1-5) and JSON
tool-use transcripts (stage 6, "Action and tool use"). `text` mode leaves
generation untouched; `json` mode primes the model toward a JSON object and
parses the result into a structured payload.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

OUTPUT_FORMATS = ("text", "json")

# Short priming so the model emits JSON (mirrors the agentic stage). Kept brief
# for small models and small context windows.
_JSON_PRIMER = "\nRespond with a single JSON object.\n"
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def normalize_format(fmt: Optional[str]) -> str:
    """Validate/normalize an output-format name."""
    fmt = (fmt or "text").lower()
    if fmt not in OUTPUT_FORMATS:
        raise ValueError(f"Unknown output format '{fmt}' — choose from {OUTPUT_FORMATS}.")
    return fmt


def wrap_prompt(prompt: str, fmt: str) -> str:
    """Prepare the prompt text for the chosen output format."""
    if normalize_format(fmt) == "json":
        return prompt.rstrip() + _JSON_PRIMER
    return prompt


def parse_output(text: str, fmt: str) -> dict:
    """Turn a raw generation into the structured result the consumer expects.

    text → {"format": "text", "text": ...}
    json → {"format": "json", "json": <obj|None>, "valid": bool, "raw": ...}
    """
    if normalize_format(fmt) == "text":
        return {"format": "text", "text": text}
    obj, valid = None, False
    m = _JSON_OBJ_RE.search(text)               # first {...} span in the output
    if m:
        try:
            obj = json.loads(m.group(0))
            valid = True
        except (ValueError, TypeError):
            obj = None
    return {"format": "json", "json": obj, "valid": valid, "raw": text}


# ─────────────────────────────── agentic loop ───────────────────────────────
# Mirrors the Claude Code / Anthropic SDK tool loop: the model emits an
# `Action: {"name","input"}`; the runner executes the tool and feeds back an
# `Observation: {...}`; this repeats until the model answers without an Action.
# Skills (Claude-style SKILL.md) are injected as extra context. Same hook a
# serving API will reuse — see [[uses/api]].

AGENT_SYSTEM = ('You can use tools. To call one, output a line '
                'Action: {"name": <tool>, "input": {<args>}}; you then receive an '
                'Observation with the result. Otherwise answer the user directly.')

_ACTION_RE = re.compile(r"Action:\s*(\{.*\})", re.DOTALL)


@dataclass
class Tool:
    """An executable tool. `run(input_dict)` returns any JSON-serializable result."""
    name: str
    description: str
    input_schema: dict
    run: Callable[[dict], Any]


def tools_spec(tools: List[Tool]) -> str:
    """Claude-style tool definitions (name/description/input_schema) as JSON."""
    return json.dumps([{"name": t.name, "description": t.description,
                        "input_schema": t.input_schema} for t in tools],
                      ensure_ascii=False)


def parse_action(text: str) -> Optional[dict]:
    """Extract a tool call from a model turn, or None. Accepts an `Action:` line
    or a bare JSON object carrying a `name`."""
    candidates = []
    m = _ACTION_RE.search(text)
    if m:
        candidates.append(m.group(1))
    m2 = _JSON_OBJ_RE.search(text)
    if m2:
        candidates.append(m2.group(0))
    for c in candidates:
        try:
            obj = json.loads(c)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict) and obj.get("name"):
            obj.setdefault("input", {})
            return obj
    return None


def build_agent_prompt(tools: List[Tool], user: str,
                       skill_md: Optional[str] = None) -> str:
    """Assemble the initial agentic prompt (system + tools + optional skill + user)."""
    parts = [f"System: {AGENT_SYSTEM}"]
    if tools:
        parts.append(f"Tools: {tools_spec(tools)}")
    if skill_md:
        parts.append(skill_md.strip())
    parts.append(f"User: {user}")
    parts.append("Assistant:")
    return "\n".join(parts)


def run_agent(generate_fn: Callable[[str], str], tools: List[Tool], user: str,
              skill_md: Optional[str] = None, max_steps: int = 4) -> dict:
    """Drive the tool loop. `generate_fn(prompt_text) -> response_text` wraps the
    model. Returns {"final": <text|None>, "steps": [{"action","observation"}, …]}.
    """
    registry = {t.name: t for t in tools}
    transcript = build_agent_prompt(tools, user, skill_md)
    steps: list = []
    for _ in range(max_steps):
        out = generate_fn(transcript)
        action = parse_action(out)
        if action is None:                          # no tool call → final answer
            return {"final": out.strip(), "steps": steps}
        tool = registry.get(action["name"])
        if tool is None:
            obs: Any = {"error": f"unknown tool '{action['name']}'"}
        else:
            try:
                obs = tool.run(action.get("input", {}) or {})
            except Exception as e:                  # tools must never crash the loop
                obs = {"error": str(e)}
        steps.append({"action": action, "observation": obs})
        transcript += (f"\nAction: {json.dumps(action, ensure_ascii=False)}"
                       f"\nObservation: {json.dumps(obs, ensure_ascii=False)}"
                       f"\nAssistant:")
    return {"final": None, "steps": steps, "note": "max steps reached"}
