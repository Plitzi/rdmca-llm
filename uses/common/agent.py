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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.core.memory.recall import (
    MEM_CLOSE,  # noqa: F401  (re-exported for callers)
    MEM_OPEN,  # noqa: F401  (re-exported for callers)
    memory_block,
)
from src.core.modalities.vocab import REASONING_SPECIALS
from src.models.cognition.mood import mood_system_phrase

OUTPUT_FORMATS = ("text", "json")

# Short priming so the model emits JSON (mirrors the agentic stage). Kept brief
# for small models and small context windows.
_JSON_PRIMER = "\nRespond with a single JSON object.\n"
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def normalize_format(fmt: str | None) -> str:
    """Validate/normalize an output-format name."""
    fmt = (fmt or "text").lower()
    if fmt not in OUTPUT_FORMATS:
        raise ValueError(f"Unknown output format '{fmt}' — choose from {OUTPUT_FORMATS}.")
    return fmt


def system_preamble(system: str | None, mood: str = "neutral") -> str:
    """Build the `System:` line that opens a conversation, the SAME way the training
    data is framed (`System: <persona> (mood: <mood>)`). The mood rides on this same
    channel and is neutral-by-default (adds nothing), so an ordinary chat is
    unchanged. Returns "" when there is neither a system prompt nor an active mood.
    Kept SEPARATE from the running history so the caller can refresh it each turn as
    the conversation's mood shifts, while it always stays at the front (in-distribution)."""
    system = (system or "").strip()
    tag = mood_system_phrase(mood)
    if not system and not tag:
        return ""
    line = "System: " + " ".join(p for p in (system, tag) if p)
    return line + "\n"


def wrap_prompt(prompt: str, fmt: str, think: str = "off") -> str:
    """Frame one chat turn the way the training data is formatted: a `User:` line
    and a trailing `Assistant:` so the model continues as the assistant. This is
    the SAME `User:/Assistant:` convention used by the dialogue/reasoning/agentic
    corpora — priming bare text (no role) is why an undertrained model just rambles
    instead of replying. The leading newline lets turns concatenate cleanly in the
    running history."""
    prompt = prompt.rstrip()
    if normalize_thinking(think) != "off":
        prompt += THINK_INSTRUCTION
    if normalize_format(fmt) == "json":
        prompt += _JSON_PRIMER
    return f"\nUser: {prompt}\nAssistant:"


def parse_output(text: str, fmt: str) -> dict:
    """Turn a raw generation into the structured result the consumer expects.

    text → {"format": "text", "text": ...}
    json → {"format": "json", "json": <obj|None>, "valid": bool, "raw": ...}
    """
    if normalize_format(fmt) == "text":
        return {"format": "text", "text": clean_answer(text)}
    text = strip_thinking(text)  # never parse JSON out of a scratchpad
    obj, valid = None, False
    m = _JSON_OBJ_RE.search(text)  # first {...} span in the output
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
THINK_OPEN, THINK_CLOSE = REASONING_SPECIALS  # single source of truth (vocab.py);
# also registered as tokenizer symbols.

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

THINK_INSTRUCTION = (
    f" First reason step by step inside {THINK_OPEN} {THINK_CLOSE}, then give the final answer."
)

# A closed scratchpad, and an unterminated one (budget hit / EOS before close).
_THINK_RE = re.compile(re.escape(THINK_OPEN) + r"(.*?)" + re.escape(THINK_CLOSE), re.DOTALL)
_THINK_OPEN_RE = re.compile(re.escape(THINK_OPEN) + r".*", re.DOTALL)


def normalize_thinking(level: str | None) -> str:
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
        answer = (text[: m.start()] + text[m.end() :]).strip()
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


def visible_stream_text(text: str) -> str:
    """The portion of a partial generation to SHOW when thinking must stay hidden
    (e.g. `--think off`): everything after a closed `</think>`; empty while still
    inside an unterminated `<think>`; the whole text when there is no scratchpad.

    A model trained on the reasoning stage emits a `<think>…</think>` block on its
    own even when not asked to — without this the raw scratchpad streams to the
    screen. Streaming `visible_stream_text(running_text)` instead shows only the
    answer (after the block), so 'think off' never leaks a scratchpad."""
    if THINK_CLOSE in text:
        tail = text.rsplit(THINK_CLOSE, 1)[1]
    elif THINK_OPEN in text:
        return ""
    else:
        return text
    # A NEW scratchpad opened after the last close → hide it (and everything after).
    return tail.split(THINK_OPEN, 1)[0] if THINK_OPEN in tail else tail


# Role tags the training transcripts use to delimit turns (User:/Assistant:/…).
# A turn's reply ends where the next such tag begins; small/undertrained models
# tend to "keep going" and echo these, so we trim at a turn boundary.
_ROLES = "User|Assistant|System|Tools?|Observation|Client|Server"
# Newline-anchored boundary that starts a NEW turn → everything after it is leak.
ANSWER_STOP_STRINGS = (
    "\nUser:",
    "\nAssistant:",
    "\nSystem:",
    "\nTools:",
    "\nObservation:",
    "\nClient:",
    "\nServer:",
)
_LEADING_ROLE_RE = re.compile(r"^\s*(?:" + _ROLES + r")\s*:\s*", re.IGNORECASE)
# A turn boundary is a role tag — newline-anchored OR inline. Small/undertrained
# models echo the next turn mid-line ("...not sure. User: ...") without a newline,
# so a \n-anchored match alone leaves the whole run-on blob in the reply. Matched
# case-sensitively (transcripts always capitalize the tag) to avoid cutting a
# natural lowercase "user:" in ordinary prose.
_ROLE_BOUNDARY_RE = re.compile(r"\b(?:" + _ROLES + r")\s*:")


def clean_answer(text: str) -> str:
    """Return just this turn's reply: drop the <think> scratchpad, a leading role
    tag (a primed 'Assistant:'), and any text from where the model starts echoing
    a new turn (a 'User:'/'System:'/… boundary). Defensive against the role-tag
    leakage small models produce; safe no-op on clean output."""
    text = strip_thinking(text)
    text = _LEADING_ROLE_RE.sub("", text.lstrip(), count=1)
    m = _ROLE_BOUNDARY_RE.search(text)
    if m:
        text = text[: m.start()]
    return text.strip()


_ROLE_NAMES = ("User", "Assistant", "System", "Tools", "Tool", "Observation", "Client", "Server")


def safe_stream_len(text: str) -> int:
    """Length of `text` that is safe to stream right now. Holds back a trailing
    fragment that could be the start of a role tag (e.g. a bare 'User' that the
    next token may turn into 'User:'), so a forming turn boundary is never printed
    before first_stop_index can cut it. Flush the remainder when generation ends."""
    longest = 0
    for r in _ROLE_NAMES:
        tag = r + ":"
        for k in range(min(len(tag), len(text)), 0, -1):
            if text.endswith(tag[:k]):
                j = len(text) - k
                if j == 0 or not text[j - 1].isalnum():  # sits on a word boundary
                    longest = max(longest, k)
                break
    return len(text) - longest


def first_stop_index(text: str, stops=ANSWER_STOP_STRINGS) -> int | None:
    """Char index of the earliest turn-boundary in `text`, or None. Lets a streaming
    generator halt (and not print past) a turn-boundary leak as soon as it forms —
    including an inline `User:`/`Assistant:` the model echoes without a newline. A
    boundary at index 0 is ignored (that's the primed role tag, not a leak)."""
    hits = [i for i in (text.find(s) for s in stops) if i != -1]
    m = _ROLE_BOUNDARY_RE.search(text)
    if m and m.start() > 0:
        hits.append(m.start())
    return min(hits) if hits else None


# ─────────────────────────────── agentic loop ───────────────────────────────
# Mirrors the Claude Code / Anthropic SDK tool loop: the model emits an
# `Action: {"name","input"}`; the runner executes the tool and feeds back an
# `Observation: {...}`; this repeats until the model answers without an Action.
# Skills (Claude-style SKILL.md) are injected as extra context. Same hook a
# serving API will reuse — see [[uses/api]].

AGENT_SYSTEM = (
    "You can use tools. To call one, output a line "
    'Action: {"name": <tool>, "input": {<args>}}; you then receive an '
    "Observation with the result. Otherwise answer the user directly."
)

_ACTION_RE = re.compile(r"Action:\s*(\{.*\})", re.DOTALL)


@dataclass
class Tool:
    """An executable tool. `run(input_dict)` returns any JSON-serializable result."""

    name: str
    description: str
    input_schema: dict
    run: Callable[[dict], Any]


def tools_spec(tools: list[Tool]) -> str:
    """Claude-style tool definitions (name/description/input_schema) as JSON."""
    return json.dumps(
        [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ],
        ensure_ascii=False,
    )


def parse_action(text: str) -> dict | None:
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


def build_agent_prompt(
    tools: list[Tool],
    user: str,
    skill_md: str | None = None,
    think: str = "off",
    system: str | None = None,
    memory: str = "",
) -> str:
    """Assemble the initial agentic prompt (memory + system + tools + optional skill
    + user). An optional `system` persona is prepended to the tool-use instructions
    so the same system-prompt channel works in the agent as in the chat. Recalled
    `memory` (if any) leads the prompt as a `<mem>…</mem>` block — same as the chat,
    so the agent recalls past context too."""
    base = AGENT_SYSTEM + (THINK_INSTRUCTION if normalize_thinking(think) != "off" else "")
    system = f"{system.strip()} {base}" if system and system.strip() else base
    parts = []
    mem = memory_block(memory)
    if mem:
        parts.append(mem.rstrip("\n"))
    parts.append(f"System: {system}")
    if tools:
        parts.append(f"Tools: {tools_spec(tools)}")
    if skill_md:
        parts.append(skill_md.strip())
    parts.append(f"User: {user}")
    parts.append("Assistant:")
    return "\n".join(parts)


def run_agent(
    generate_fn: Callable[[str], str],
    tools: list[Tool],
    user: str,
    skill_md: str | None = None,
    max_steps: int = 6,
    think: str = "off",
    max_context_chars: int = 8000,
    system: str | None = None,
    memory: str = "",
    context_mgr=None,
    encode=None,
    decode=None,
    should_stop: Callable[[], bool] | None = None,
    get_steering: Callable[[], list[str]] | None = None,
) -> dict:
    """Drive the tool loop — multiple think→act→observe rounds until the model
    answers (Claude Code-style). `generate_fn(prompt_text) -> response_text`
    wraps the model; it may return a `<think>…</think>` scratchpad before the
    answer/action, which is captured per step (the action/observation parsing
    ignores it). Returns
    {"final": <text|None>, "thinking": <text|None>,
     "steps": [{"thinking","action","observation"}, …]}.

    The prompt is rebuilt each round as the static header (system+tools+skill+user)
    plus the step tail. Appending blindly would grow the transcript without bound;
    the model's own context-length trim then drops from the FRONT, silently
    discarding the system/tools spec. So the header is always PINNED and only the
    tail is bounded. Two tail strategies:
      • default — keep as many RECENT step blocks as fit `max_context_chars`;
      • STR sector context-slots (§12), when a `context_mgr` (+ `encode`/`decode`)
        is supplied — route each completed step block to its sector slot(s), evict
        overflow to the episodic buffer (not discarded), and assemble the active
        tail from the slots. Same mechanism the chat uses for its history body, so
        the agent recalls/forgets context the same way every other surface does."""
    registry = {t.name: t for t in tools}
    header = build_agent_prompt(tools, user, skill_md, think, system=system, memory=memory)
    steps: list = []

    # STR slots need a tokenizer round-trip (the manager works in token space); the
    # header is encoded once so the assembled tail can be capped to leave room for
    # it + the next generation, keeping header+tail within the positional window.
    use_slots = context_mgr is not None and encode is not None and decode is not None
    header_budget = len(encode(header)) if use_slots else 0

    # Mid-run STEERING: messages the user types while the agent works (queued via
    # get_steering) are injected as the latest User turn, so they can CORRECT an
    # agent heading the wrong way — Claude Code-style. The prompt ends with the
    # correction + a fresh "Assistant:" cue, so the next step answers it.
    steering: list = []

    def _apply_steering(prompt: str) -> str:
        if not steering:
            return prompt
        prompt = prompt.rstrip()
        if prompt.endswith("Assistant:"):  # drop the stale cue
            prompt = prompt[: -len("Assistant:")].rstrip()
        return prompt + "".join(f"\nUser: {m}" for m in steering) + "\nAssistant:"

    def _block(st: dict) -> str:
        return (
            f"\nAction: {json.dumps(st['action'], ensure_ascii=False)}"
            f"\nObservation: {json.dumps(st['observation'], ensure_ascii=False)}"
            f"\nAssistant:"
        )

    def _transcript() -> str:
        if use_slots:  # §12 sector slots assemble the tail
            cap = max(64, context_mgr.context_len - header_budget - 128)
            return header + decode(context_mgr.assemble(cap))
        kept, total = [], len(header)
        for st in reversed(steps):  # keep the most recent steps that fit
            b = _block(st)
            if kept and total + len(b) > max_context_chars:
                break
            kept.append(b)
            total += len(b)
        return header + "".join(reversed(kept))

    for _ in range(max_steps):
        if should_stop is not None and should_stop():  # user aborted the run
            return {"final": None, "steps": steps, "thinking": None, "note": "interrupted"}
        if get_steering is not None:  # pull queued corrections
            steering.extend(m for m in get_steering() if m.strip())
        out = generate_fn(_apply_steering(_transcript()))
        thinking, answer = split_thinking(out)
        action = parse_action(answer)
        if action is None:  # no tool call → final answer
            return {"final": answer.strip(), "thinking": thinking, "steps": steps}
        tool = registry.get(action["name"])
        if tool is None:
            obs: Any = {"error": f"unknown tool '{action['name']}'"}
        else:
            try:
                obs = tool.run(action.get("input", {}) or {})
            except Exception as e:  # tools must never crash the loop
                obs = {"error": str(e)}
        steps.append({"thinking": thinking, "action": action, "observation": obs})
        if use_slots:  # route this step's block to its slot(s)
            context_mgr.add(encode(_block(steps[-1])))
    return {"final": None, "steps": steps, "note": "max steps reached"}
