"""Stage 5 data sources — reasoning (chain-of-thought).

Real GSM8K CoT (reframed into <think>…</think> + answer) and multi-step hermes tool
sessions (reframed with a <think> plan) blended with synthetic CoT word problems in
the exact stage-5 format, so the model learns to think → plan → use tools → answer.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterator

from src.plugins.sdk import (
    AGENTIC_SYSTEM_PROMPT,
    REASONING_SPECIALS,
    blend,
    hermes_events,
    stable_hash,
)

# Single source of truth: the tokenizer registers these via vocab.CONTROL_SPECIALS,
# and agent.THINK_OPEN/CLOSE use the same list — keep all three in sync through it.
_THINK_OPEN, _THINK_CLOSE = REASONING_SPECIALS
_GSM_FINAL_RE = re.compile(r"####\s*(.+?)\s*$", re.DOTALL)

_SYN_NAMES = [
    "Tom",
    "Mia",
    "Ana",
    "Leo",
    "Sam",
    "Noa",
    "Eva",
    "Ravi",
    "Kai",
    "Lia",
    "Omar",
    "Zoe",
    "Ben",
    "Ivy",
    "Max",
    "Nia",
]
_SYN_OBJECTS = [
    "apples",
    "oranges",
    "cookies",
    "marbles",
    "stickers",
    "crayons",
    "blocks",
    "balloons",
    "coins",
    "pencils",
    "books",
    "candies",
    "toys",
    "cards",
    "shells",
    "buttons",
]


def _cot_transcript(question: str, answer: str) -> str | None:
    """Reframe one (question, GSM8K-answer) pair as a <think>…</think> + answer
    transcript. None if it lacks either working steps or a final answer."""
    question = " ".join((question or "").split())
    answer = (answer or "").strip()
    if not question or not answer:
        return None
    match = _GSM_FINAL_RE.search(answer)
    final = match.group(1).strip() if match else ""
    steps = " ".join(_GSM_FINAL_RE.sub("", answer).split())
    if not steps or not final:
        return None
    return (
        f"User: {question}\nAssistant: {_THINK_OPEN} {steps} {_THINK_CLOSE}\nThe answer is {final}."
    )


# A planning aid the agent can call WHEN AVAILABLE (mirrors models/cognition/uses/agent/tools/todo.py).
_TODO_TOOL_DEF = {
    "name": "todo",
    "description": "Record a short plan of steps before acting (use when planning).",
    "input_schema": {
        "type": "object",
        "properties": {"items": {"type": "array", "items": {"type": "string"}}},
        "required": [],
    },
}


def _reasoning_tool_transcript(ex: dict, use_todo: bool = False) -> str | None:
    """Reframe a *multi-step* hermes tool session into a reasoning-with-tools trace:
    a <think>…</think> that states a PLAN (the real, ordered tools the session uses)
    before the Action/Observation loop. Optionally records the plan with a `todo` tool.
    None unless ≥2 tool calls."""
    tools, events = hermes_events(ex)
    ordered = list(dict.fromkeys(p["name"] for k, p in events if k == "call" and p.get("name")))
    if len(ordered) < 2:  # planning only matters when multi-step
        return None
    plan = "; ".join(f"{i}) {name}" for i, name in enumerate(ordered, 1))

    tools_listing, todo_on = tools, False
    if use_todo and tools:
        try:
            arr = json.loads(tools)
            arr.append(_TODO_TOOL_DEF)
            tools_listing = json.dumps(arr, ensure_ascii=False)
            todo_on = True
        except Exception:
            todo_on = False

    lines = [f"System: {AGENTIC_SYSTEM_PROMPT}"]
    if tools_listing:
        lines.append(f"Tools: {tools_listing}")
    injected = False
    for kind, payload in events:
        if kind == "user":
            lines.append(f"User: {payload}")
            if not injected:  # plan right after the goal is stated
                lines.append(
                    f"Assistant: {_THINK_OPEN} To do this I will call, in order: {plan}. {_THINK_CLOSE}"
                )
                if todo_on:  # record the plan with the todo tool
                    call = {"name": "todo", "input": {"items": ordered}}
                    obs = {
                        "plan": [{"content": name, "status": "pending"} for name in ordered],
                        "remaining": len(ordered),
                    }
                    lines.append(f"Action: {json.dumps(call, ensure_ascii=False)}")
                    lines.append(f"Observation: {json.dumps(obs, ensure_ascii=False)}")
                injected = True
        elif kind == "assistant":
            lines.append(f"Assistant: {payload}")
        elif kind == "call":
            lines.append(f"Action: {json.dumps(payload, ensure_ascii=False)}")
        elif kind == "result":
            lines.append(f"Observation: {payload}")
    return "\n".join(lines) if injected and len(lines) >= 5 else None


def stream_reasoning(langs: list[str], limit_mb: int | None = None) -> Iterator[dict]:
    """Stage-5 reasoning: blends three real signals — chain-of-thought (GSM8K), basic
    planning, and tool use (multi-step hermes sessions reframed with a <think> plan).
    Interleaved ~2 CoT : 1 plan+tools so step-by-step reasoning stays dominant. EN only."""
    if "en" not in {lang.lower() for lang in langs}:
        return
    from datasets import load_dataset

    def _load(name, *args, **kw):
        try:
            return iter(load_dataset(name, *args, split="train", streaming=True, **kw))
        except Exception as e:
            print(f"    [reasoning] {name}: {e}")
            return iter(())

    cot_it = _load("openai/gsm8k", "main")
    tool_it = _load("NousResearch/hermes-function-calling-v1")
    seen: set = set()
    n_tool = 0
    while True:
        produced = False
        for _ in range(2):  # 2 chain-of-thought traces
            ex = next(cot_it, None)
            if ex is None:
                break
            text = _cot_transcript(ex.get("question", ""), ex.get("answer", ""))
            if text:
                produced = True
                yield {"text": text, "lang": "en"}
        ex = next(tool_it, None)  # 1 plan + tool-use trace
        if ex is not None:
            text = _reasoning_tool_transcript(ex, use_todo=(n_tool % 2 == 0))
            if text and (h := stable_hash(text)) not in seen:
                seen.add(h)
                n_tool += 1
                produced = True
                yield {"text": text, "lang": "en"}
        if not produced:  # both sources exhausted
            return


def gen_cot(n: int, seed: int = 1) -> Iterator[dict]:
    """Synthetic chain-of-thought word problems in the EXACT stage-5 format — a
    <think>…</think> scratchpad with arithmetic steps, then 'The answer is N.' This
    teaches the model to OPEN and then CLOSE the think block and emit an answer."""
    rng = random.Random(seed)
    for _ in range(n):
        name, obj = rng.choice(_SYN_NAMES), rng.choice(_SYN_OBJECTS)
        kind = rng.randint(0, 3)
        if kind == 0:  # addition
            a, b = rng.randint(1, 20), rng.randint(1, 20)
            result = a + b
            q = f"{name} has {a} {obj} and gets {b} more. How many {obj} does {name} have now?"
            steps = f"{name} starts with {a} {obj} and gets {b} more. {a} + {b} = {result}."
        elif kind == 1:  # subtraction
            a = rng.randint(5, 20)
            b = rng.randint(1, a)
            result = a - b
            q = f"{name} has {a} {obj} and gives away {b}. How many {obj} are left?"
            steps = f"{name} starts with {a} {obj} and gives away {b}. {a} - {b} = {result}."
        elif kind == 2:  # multiplication (equal groups)
            a, b = rng.randint(2, 9), rng.randint(2, 9)
            result = a * b
            q = f"There are {a} boxes with {b} {obj} in each. How many {obj} in total?"
            steps = f"Each of the {a} boxes has {b} {obj}. {a} x {b} = {result}."
        else:  # division (equal sharing)
            b, per = rng.randint(2, 6), rng.randint(2, 6)
            a = b * per
            result = per
            q = f"{name} shares {a} {obj} equally among {b} friends. How many does each get?"
            steps = f"{a} {obj} are shared among {b} friends. {a} / {b} = {result}."
        yield {
            "text": f"User: {q}\nAssistant: {_THINK_OPEN} {steps} {_THINK_CLOSE}\nThe answer is {result}.",
            "lang": "en",
        }


def _build_reasoning(*, langs, approx_examples, limit_mb=None, **_):
    return blend(stream_reasoning(langs, limit_mb), gen_cot(approx_examples), approx_examples)


SOURCES = {"reasoning": _build_reasoning, "cot": _build_reasoning}
