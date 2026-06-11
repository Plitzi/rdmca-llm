"""Agent output formatting — the model can return plain text or JSON, selectable
by the consumer. Today the chat CLI exposes it via `--format` (and the `/format`
command); a serving API would expose the same as a request field. Centralized
here so every consumer stays in sync.

The base model learned both registers: natural language (stages 1-5) and JSON
tool-use transcripts (stage 7, "Action and tool use"). `text` mode leaves
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


def wrap_prompt(prompt: str, fmt: str, think: str = "off") -> str:
    """Prepare the prompt text for the chosen output format and thinking level."""
    prompt = prompt.rstrip()
    if normalize_thinking(think) != "off":
        prompt += THINK_INSTRUCTION
    if normalize_format(fmt) == "json":
        prompt += _JSON_PRIMER
    return prompt


def parse_output(text: str, fmt: str) -> dict:
    """Turn a raw generation into the structured result the consumer expects.

    text → {"format": "text", "text": ...}
    json → {"format": "json", "json": <obj|None>, "valid": bool, "raw": ...}
    """
    if normalize_format(fmt) == "text":
        return {"format": "text", "text": clean_answer(text)}
    text = strip_thinking(text)                 # never parse JSON out of a scratchpad
    obj, valid = None, False
    m = _JSON_OBJ_RE.search(text)               # first {...} span in the output
    if m:
        try:
            obj = json.loads(m.group(0))
            valid = True
        except (ValueError, TypeError):
            obj = None
    return {"format": "json", "json": obj, "valid": valid, "raw": text}


# ─────────────────────────── thinking / reasoning ───────────────────────────
# A reasoning register the model learns in the Reasoning stage (stage 5, the
# capstone of the frozen cognitive base): it
# emits a <think>…</think> scratchpad and then the answer (mirrors Claude's
# thinking blocks). The "thinking level" is an effort dial — whether to think at
# all and how large a token budget the scratchpad gets — analogous to Claude's
# reasoning effort. Same hook every consumer (chat / agent / future API) reuses.
THINKING_LEVELS = ("off", "low", "medium", "high")
THINK_OPEN  = "<think>"          # MUST match the delimiters used by the
THINK_CLOSE = "</think>"         # reasoning data (see src/data/graded.py).

# Fraction of the per-turn token budget (`max_tokens`) the scratchpad may use.
# off → no thinking; high → up to the full budget (it closes at </think> sooner
# if the model finishes on its own). The dial is relative to max_tokens because
# small levels have tiny context windows, so an absolute count won't fit.
_THINK_BUDGET_FRAC = {"off": 0.0, "low": 0.25, "medium": 0.5, "high": 1.0}

# Resource ceiling on scratchpad tokens (NOT the loop defense). This only bounds
# how much memory/compute one turn's reasoning can request — it is deliberately
# generous so it never truncates genuine long reasoning. The actual defense
# against an infinite-thinking logic bomb is the *loop detector* in
# run_chat.generate() (`_looping` + the wall-clock deadline), which fires only on
# degenerate repetition/stalls and so cannot cut short real, progressing thought.
# A looping scratchpad is detected and closed early; the model then still answers.
MAX_THINK_TOKENS = 4096

THINK_INSTRUCTION = (f" First reason step by step inside {THINK_OPEN} {THINK_CLOSE}, "
                     "then give the final answer.")

# A closed scratchpad, and an unterminated one (budget hit / EOS before close).
_THINK_RE      = re.compile(re.escape(THINK_OPEN) + r"(.*?)" + re.escape(THINK_CLOSE),
                            re.DOTALL)
_THINK_OPEN_RE = re.compile(re.escape(THINK_OPEN) + r".*", re.DOTALL)


def normalize_thinking(level: Optional[str]) -> str:
    """Validate/normalize a thinking-level name."""
    level = (level or "off").lower()
    if level not in THINKING_LEVELS:
        raise ValueError(f"Unknown thinking level '{level}' — choose from {THINKING_LEVELS}.")
    return level


def think_budget(level: str, max_tokens: int) -> int:
    """Token budget for the <think> scratchpad at this level (0 = no thinking),
    clamped to MAX_THINK_TOKENS as an anti-logic-bomb ceiling."""
    budget = int(max_tokens * _THINK_BUDGET_FRAC[normalize_thinking(level)])
    return min(budget, MAX_THINK_TOKENS)


def split_thinking(text: str) -> tuple:
    """Split a raw generation into (thinking, answer).

    A closed <think>…</think> block is returned as `thinking` (its inner text);
    everything outside it is the answer. Returns (None, text) when no block is
    present."""
    m = _THINK_RE.search(text)
    if m:
        answer = (text[:m.start()] + text[m.end():]).strip()
        return m.group(1).strip(), answer
    return None, text


def strip_thinking(text: str) -> str:
    """Drop any <think> scratchpad (closed or unterminated) from a turn, leaving
    just the answer. Used before action/format parsing so reasoning is never
    mistaken for a tool call or the output payload."""
    thinking, answer = split_thinking(text)
    if thinking is not None:
        return answer
    return _THINK_OPEN_RE.sub("", text).strip()


# Role tags the training transcripts use to delimit turns (User:/Assistant:/…).
# A turn's reply ends where the next such tag begins; small/undertrained models
# tend to "keep going" and echo these, so we trim at a turn boundary.
_ROLES = "User|Assistant|System|Tools?|Observation|Client|Server"
# Newline-anchored boundary that starts a NEW turn → everything after it is leak.
ANSWER_STOP_STRINGS = ("\nUser:", "\nAssistant:", "\nSystem:", "\nTools:",
                       "\nObservation:", "\nClient:", "\nServer:")
_ROLE_BOUNDARY_RE = re.compile(r"\n\s*(?:" + _ROLES + r")\s*:", re.IGNORECASE)
_LEADING_ROLE_RE  = re.compile(r"^\s*(?:" + _ROLES + r")\s*:\s*", re.IGNORECASE)


def clean_answer(text: str) -> str:
    """Return just this turn's reply: drop the <think> scratchpad, a leading role
    tag (a primed 'Assistant:'), and any text from where the model starts echoing
    a new turn (a 'User:'/'System:'/… boundary). Defensive against the role-tag
    leakage small models produce; safe no-op on clean output."""
    text = strip_thinking(text)
    text = _LEADING_ROLE_RE.sub("", text.lstrip(), count=1)
    m = _ROLE_BOUNDARY_RE.search(text)
    if m:
        text = text[:m.start()]
    return text.strip()


def first_stop_index(text: str, stops=ANSWER_STOP_STRINGS) -> Optional[int]:
    """Char index of the earliest stop string in `text`, or None. Lets a streaming
    generator halt (and not print past) a turn-boundary leak as soon as it forms."""
    hits = [i for i in (text.find(s) for s in stops) if i != -1]
    return min(hits) if hits else None


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
    or a bare JSON object carrying a `name`. Any <think> scratchpad is stripped
    first so reasoning is never mistaken for a tool call."""
    text = strip_thinking(text)
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
                       skill_md: Optional[str] = None, think: str = "off") -> str:
    """Assemble the initial agentic prompt (system + tools + optional skill + user)."""
    system = AGENT_SYSTEM + (THINK_INSTRUCTION if normalize_thinking(think) != "off" else "")
    parts = [f"System: {system}"]
    if tools:
        parts.append(f"Tools: {tools_spec(tools)}")
    if skill_md:
        parts.append(skill_md.strip())
    parts.append(f"User: {user}")
    parts.append("Assistant:")
    return "\n".join(parts)


def run_agent(generate_fn: Callable[[str], str], tools: List[Tool], user: str,
              skill_md: Optional[str] = None, max_steps: int = 6,
              think: str = "off") -> dict:
    """Drive the tool loop — multiple think→act→observe rounds until the model
    answers (Claude Code-style). `generate_fn(prompt_text) -> response_text`
    wraps the model; it may return a `<think>…</think>` scratchpad before the
    answer/action, which is captured per step (the action/observation parsing
    ignores it). Returns
    {"final": <text|None>, "thinking": <text|None>,
     "steps": [{"thinking","action","observation"}, …]}.
    """
    registry = {t.name: t for t in tools}
    transcript = build_agent_prompt(tools, user, skill_md, think)
    steps: list = []
    for _ in range(max_steps):
        out = generate_fn(transcript)
        thinking, answer = split_thinking(out)
        action = parse_action(answer)
        if action is None:                          # no tool call → final answer
            return {"final": answer.strip(), "thinking": thinking, "steps": steps}
        tool = registry.get(action["name"])
        if tool is None:
            obs: Any = {"error": f"unknown tool '{action['name']}'"}
        else:
            try:
                obs = tool.run(action.get("input", {}) or {})
            except Exception as e:                  # tools must never crash the loop
                obs = {"error": str(e)}
        steps.append({"thinking": thinking, "action": action, "observation": obs})
        transcript += (f"\nAction: {json.dumps(action, ensure_ascii=False)}"
                       f"\nObservation: {json.dumps(obs, ensure_ascii=False)}"
                       f"\nAssistant:")
    return {"final": None, "steps": steps, "note": "max steps reached"}
