"""
Graded data for the LEVEL curriculum.

Each level teaches information of increasing complexity. This module provides:

  - `flesch_kincaid_grade` / `passes_filter` — a readability gate (US grade level)
    used to keep text at/below a level's reading difficulty (level 5 = no gate).
  - streamers over real HuggingFace corpora for the lower levels: TinyStories,
    conversation (`stream_dialogue`), arithmetic (`stream_arithmetic`), causal
    reasoning (`stream_causal`), chain-of-thought (`stream_reasoning`) and
    Simple-English Wikipedia.
  - `gen_analogies` is the one remaining TEMPORARY synthetic generator (no real
    graded analogy corpus loads on datasets>=3 yet); see its TODO.

`stream_source(key, ...)` dispatches a source name (as listed in a level's
`curriculum.stageN.data.sources`) to the right generator/streamer. The full
corpora (`wikipedia`, `arc_*`, `gsm8k`, `math`, `ethics`) live in
scripts/prepare_data.py and are passed in via `extra_streamers` to avoid a
circular import.
"""
from __future__ import annotations
import json
import random
import re
from typing import Callable, Dict, Iterator, List, Optional

# ──────────────────────────── readability gate ──────────────────────────────
_VOWELS = "aeiouy"


def _syllables(word: str) -> int:
    """Approximate syllable count: number of vowel groups (min 1)."""
    word = word.lower()
    groups = re.findall(r"[aeiouy]+", word)
    n = len(groups)
    if word.endswith("e") and n > 1:          # silent final 'e'
        n -= 1
    return max(n, 1)


def flesch_kincaid_grade(text: str) -> float:
    """Flesch-Kincaid US grade level. Higher = harder to read.
    grade = 0.39·(words/sentence) + 11.8·(syllables/word) − 15.59."""
    words = re.findall(r"[A-Za-zÀ-ÿ']+", text)
    if not words:
        return 0.0
    sentences = max(len(re.findall(r"[.!?]+", text)), 1)
    syl = sum(_syllables(w) for w in words)
    wps = len(words) / sentences
    spw = syl / len(words)
    return 0.39 * wps + 11.8 * spw - 15.59


def passes_filter(text: str, spec: Optional[dict]) -> bool:
    """True if `text` is simple enough for the filter spec. `spec` is None at
    level 5 (everything passes). Keys: `max_grade`, `max_word_len`."""
    if not spec:
        return True
    if "max_word_len" in spec:
        if any(len(w) > spec["max_word_len"] for w in text.split()):
            return False
    if "max_grade" in spec:
        if flesch_kincaid_grade(text) > spec["max_grade"]:
            return False
    return True


# ──────────────────────────── arithmetic (real) ─────────────────────────────
# Real symbolic arithmetic from AtlasUnified/atlas-math-sets ("a op b = c").
# Language-agnostic (only digits/operators) and graded by operand magnitude:
#   level 1 single-digit +/−, 2 two-digit +/−/×/÷, 3+ larger / all operators.
_ARITH_RE = re.compile(r"\s*(\d+)\s*([+\-x×*/])\s*(\d+)\s*=")
# atlas-math has ~17.8M rows; the graded subset (esp. single-digit at level 1) is
# a tiny, finite space, so scanning a bounded prefix covers it many times over
# without churning the whole dataset every run.
_ARITH_SCAN_CAP = 400_000


def _arith_difficulty(a: int, b: int, op: str) -> int:
    """Coarse difficulty from operand magnitude + operation type."""
    mx = max(a, b)
    if op in "+-":
        return 1 if mx < 10 else (2 if mx < 100 else 3)
    return 2 if mx < 100 else 3                     # ×/÷ never count as level 1


def stream_arithmetic(langs: List[str], level: int,
                      limit_mb: Optional[int] = None) -> Iterator[dict]:
    """Stream real arithmetic equations graded to `level`. Symbolic content, so
    tagged with the primary configured language (it is language-agnostic). Scans
    a bounded prefix of the dataset (the graded space is small and finite)."""
    from datasets import load_dataset
    lang = langs[0] if langs else "en"
    try:
        ds = load_dataset("AtlasUnified/atlas-math-sets", split="train", streaming=True)
    except Exception as e:
        print(f"    [arithmetic] {e}")
        return
    for scanned, ex in enumerate(ds):
        if scanned >= _ARITH_SCAN_CAP:
            break
        out = (ex.get("output") or "").strip()
        m = _ARITH_RE.match(out)
        if not m:
            continue                                # skip roots/powers/word problems
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        if _arith_difficulty(a, b, op) <= level:
            yield {"text": out, "lang": lang}


# ──────────────────────────── analogies (TEMPORARY synthetic) ───────────────
# TODO: replace with a real graded analogy corpus once one is available in a
# loadable (parquet) form — current public analogy datasets are script-based and
# no longer load on datasets>=3. Kept synthetic for now so stage 2 (perception /
# pattern recognition) still has data at the lower levels.
_ANALOGY_PAIRS = [
    ("dog", "puppy", "cat", "kitten"), ("big", "small", "tall", "short"),
    ("hot", "cold", "up", "down"), ("day", "night", "sun", "moon"),
    ("happy", "sad", "fast", "slow"), ("bird", "fly", "fish", "swim"),
    ("king", "queen", "man", "woman"), ("hand", "glove", "foot", "sock"),
]


def gen_analogies(n: int, seed: int = 1) -> Iterator[dict]:
    rng = random.Random(seed)
    for _ in range(n):
        if rng.random() < 0.5:
            a, b, c, d = rng.choice(_ANALOGY_PAIRS)
            yield {"text": f"{a} is to {b} as {c} is to {d}.", "lang": "en"}
        else:
            start = rng.randint(1, 9); step = rng.randint(1, 5)
            seq = [start + step * i for i in range(4)]
            yield {"text": f"Pattern: {seq[0]} {seq[1]} {seq[2]} {seq[3]} -> {seq[3]+step}", "lang": "en"}


# ──────────────────────────── memory (synthetic, EN) ────────────────────────
# Trains the Memory-management stage (stage 6, frozen cognitive core) to CONSUME
# recalled memory. Each example leads with a `<mem>…</mem>` block — the SAME framing
# src/agent.py injects at inference (agent.MEM_OPEN / MEM_CLOSE) — holding the
# relevant fact among distractors, then a User question and an Assistant answer
# that USES the fact. ~20% are negatives where the answer is NOT in memory, so the
# model learns to recall from the block instead of hallucinating. The User:/Assistant:
# framing matches the rest of the corpus, so completion-only loss masking applies.
_MEM_NAMES = ["Maria", "Tom", "Aisha", "Kenji", "Lucia", "Omar",
              "Sven", "Priya", "Diego", "Lena", "Nora", "Hugo"]
_MEM_FACTS = [
    ("favorite color", ["blue", "green", "red", "purple", "orange", "teal", "yellow"]),
    ("pet",            ["a cat", "a dog", "a parrot", "a rabbit", "a turtle", "a hamster"]),
    ("home city",      ["Lima", "Cairo", "Oslo", "Kyoto", "Madrid", "Accra", "Quito"]),
    ("job",            ["a teacher", "a nurse", "a baker", "an engineer", "a pilot", "a chef"]),
    ("favorite food",  ["pasta", "mango", "sushi", "tacos", "lentils", "ramen"]),
    ("birthday month", ["March", "July", "October", "January", "May", "September"]),
]


def _mem_fact_line(name: str, attr: str, val: str) -> str:
    return f"{name}'s {attr} is {val}."


def gen_memory(n: int, seed: int = 1) -> Iterator[dict]:
    """Synthetic recall-and-use examples: a <mem> block of facts + distractors, a
    question, and an answer that uses (or correctly disclaims) the memory."""
    rng = random.Random(seed)
    for _ in range(n):
        k = rng.randint(2, 4)                        # facts in the <mem> block
        names = rng.sample(_MEM_NAMES, k)
        attrs = [rng.choice(_MEM_FACTS) for _ in range(k)]
        facts = [(names[i], attrs[i][0], rng.choice(attrs[i][1])) for i in range(k)]
        lines = [f"- {_mem_fact_line(*f)}" for f in facts]
        rng.shuffle(lines)
        block = "<mem>\n" + "\n".join(lines) + "\n</mem>"
        if rng.random() < 0.8:                       # positive: answer lives in memory
            tgt = rng.choice(facts)
            q, a = f"What is {tgt[0]}'s {tgt[1]}?", _mem_fact_line(*tgt)
        else:                                        # negative: not in memory
            outsider = rng.choice([nm for nm in _MEM_NAMES if nm not in names])
            attr = rng.choice(_MEM_FACTS)[0]
            q, a = f"What is {outsider}'s {attr}?", "I don't have that in my memory."
        yield {"text": f"{block}\nUser: {q}\nAssistant: {a}", "lang": "en"}


# ──────────────────────────── causal (real, EN) ─────────────────────────────
# Real cause→effect statements from the e-CARE dataset. English only — no real
# multilingual causal corpus yet — so emitted only when English is requested.
def _causal_statement(cause: str, effect: str) -> str:
    cause = cause.strip().rstrip(".")
    effect = effect.strip()
    if effect:
        effect = effect[0].lower() + effect[1:]
    return f"{cause}, so {effect}"


def stream_causal(langs: List[str], limit_mb: Optional[int] = None) -> Iterator[dict]:
    """Stream real cause→effect statements (EN) reconstructed from e-CARE."""
    if "en" not in {l.lower() for l in langs}:
        return
    from datasets import load_dataset
    try:
        ds = load_dataset("12ml/e-CARE", split="train", streaming=True)
    except Exception as e:
        print(f"    [causal] {e}")
        return
    for ex in ds:
        correct = ex.get("choice1") if str(ex.get("label")) == "0" else ex.get("choice2")
        premise = ex.get("premise") or ""
        if not (correct and premise):
            continue
        if ex.get("question") == "cause":           # premise is the effect
            cause, effect = correct, premise
        else:                                        # premise is the cause
            cause, effect = premise, correct
        yield {"text": _causal_statement(cause, effect), "lang": "en"}


# ──────────────────────────── agentic tool use (real, EN) ───────────────────
# Real function-calling conversations (NousResearch/hermes-function-calling-v1)
# re-serialized into a Claude-style agentic loop with JSON tool calls:
#   System: <how to call tools>
#   Tools: [{"name","description","input_schema"}, ...]
#   User: ...
#   Assistant: <optional text>
#   Action: {"name": "...", "input": {...}}
#   Observation: {...}
#   Assistant: <final answer>
# The model learns to emit an `Action` JSON and consume an `Observation` — the
# tool(args)→result loop used by Claude Code / the Anthropic SDK. JSON is used
# throughout (universal). English only — no multilingual tool-use corpus yet.
_AGENTIC_SYS = ('You can use tools. To call one, output a line '
                'Action: {"name": <tool>, "input": {<args>}} and you will then '
                'receive an Observation with the result; otherwise answer directly.')
_TOOLCALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TOOLRESP_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)


def _hermes_tools(raw) -> Optional[str]:
    """Normalize hermes `tools` to a compact JSON array of
    {name, description, input_schema} (Claude tool-definition shape)."""
    try:
        arr = raw if isinstance(raw, list) else json.loads(raw)
    except Exception:
        return None
    out = []
    for t in arr if isinstance(arr, list) else []:
        fn = t.get("function", t) if isinstance(t, dict) else None
        if fn and fn.get("name"):
            out.append({"name": fn["name"],
                        "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters", {})})
    return json.dumps(out, ensure_ascii=False) if out else None


def _hermes_events(ex: dict):
    """Parse one hermes example into (tools_json, events). `events` is a list of
    ('user'|'assistant'|'call'|'result', payload) — shared by the agentic (stage 7)
    and MCP (stage 8) serializers below."""
    tools = _hermes_tools(ex.get("tools"))
    events: list = []
    for turn in ex.get("conversations") or []:
        frm, val = turn.get("from"), (turn.get("value") or "").strip()
        if not val:
            continue
        if frm == "human":
            events.append(("user", " ".join(val.split())))
        elif frm == "gpt":
            text = _TOOLCALL_RE.sub("", val).strip()
            if text:
                events.append(("assistant", " ".join(text.split())))
            for c in _TOOLCALL_RE.findall(val):
                try:
                    obj = json.loads(c)
                except Exception:
                    continue
                events.append(("call", {"name": obj.get("name"),
                                        "input": obj.get("arguments", obj.get("input", {}))}))
        elif frm == "tool":
            m = _TOOLRESP_RE.search(val)
            events.append(("result", " ".join((m.group(1) if m else val).split())))
    return tools, events


def _hermes_to_transcript(ex: dict) -> Optional[str]:
    """Serialize one hermes example into the canonical agentic transcript
    (Claude-style Action/Observation). None unless it has ≥1 real tool call."""
    tools, events = _hermes_events(ex)
    lines = [f"System: {_AGENTIC_SYS}"]
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


def stream_agentic(langs: List[str], limit_mb: Optional[int] = None) -> Iterator[dict]:
    """Stream real agentic tool-use transcripts (EN) as a Claude-style loop."""
    if "en" not in {l.lower() for l in langs}:
        return
    from datasets import load_dataset
    try:
        ds = load_dataset("NousResearch/hermes-function-calling-v1",
                          split="train", streaming=True)
    except Exception as e:
        print(f"    [agentic] {e}")
        return
    seen: set = set()
    for ex in ds:
        text = _hermes_to_transcript(ex)
        if not text:
            continue
        h = hash(text)
        if h in seen:
            continue
        seen.add(h)
        yield {"text": text, "lang": "en"}


# ──────────────────────────── reasoning / CoT (real, EN) ────────────────────
# Stage 5: chain-of-thought traces (capstone of the frozen cognitive base). GSM8K
# answers already contain step-by-step
# working ending in "#### <final>"; we reframe each into the canonical thinking
# transcript — a <think>…</think> scratchpad followed by the answer — so the
# model learns to reason before answering (mirrors Claude's thinking blocks).
# Single source of truth: the tokenizer registers these via vocab.CONTROL_SPECIALS,
# and agent.THINK_OPEN/CLOSE use the same list — keep all three in sync through it.
from src.modalities.vocab import REASONING_SPECIALS
_THINK_OPEN, _THINK_CLOSE = REASONING_SPECIALS
_GSM_FINAL_RE = re.compile(r"####\s*(.+?)\s*$", re.DOTALL)


def _cot_transcript(question: str, answer: str) -> Optional[str]:
    """Reframe one (question, GSM8K-answer) pair as a <think>…</think> + answer
    transcript. None if it lacks either working steps or a final answer."""
    question = " ".join((question or "").split())
    answer = (answer or "").strip()
    if not question or not answer:
        return None
    m = _GSM_FINAL_RE.search(answer)
    final = m.group(1).strip() if m else ""
    steps = " ".join(_GSM_FINAL_RE.sub("", answer).split())
    if not steps or not final:
        return None
    return (f"User: {question}\n"
            f"Assistant: {_THINK_OPEN} {steps} {_THINK_CLOSE}\n"
            f"The answer is {final}.")


# A planning aid the agent can call WHEN AVAILABLE (mirrors uses/agent/tools/todo.py
# and Claude Code's TodoWrite). Half the reasoning tool-transcripts expose it and
# half don't, so the model learns to record a plan with `todo` only if the session
# offers the tool — never to hallucinate a tool that isn't listed.
_TODO_TOOL_DEF = {
    "name": "todo",
    "description": "Record a short plan of steps before acting (use when planning).",
    "input_schema": {"type": "object",
                     "properties": {"items": {"type": "array",
                                              "items": {"type": "string"}}},
                     "required": []},
}


def _reasoning_tool_transcript(ex: dict, use_todo: bool = False) -> Optional[str]:
    """Reframe a *multi-step* hermes tool session into a reasoning-with-tools trace:
    a <think>…</think> that states a PLAN (the real, ordered list of tools the
    session uses) before the Action/Observation loop runs. This teaches the stage-5
    behaviour the user asked for — think → plan → (use a `todo` tool if available)
    → use tools → answer — using the real call sequence as the plan (nothing
    invented). When `use_todo`, a `todo` tool is added to the listing and the plan
    is recorded with it first. None unless ≥2 tool calls."""
    tools, events = _hermes_events(ex)
    ordered = list(dict.fromkeys(p["name"] for k, p in events
                                 if k == "call" and p.get("name")))   # de-dup, keep order
    if len(ordered) < 2:                       # planning only matters when multi-step
        return None
    plan = "; ".join(f"{i}) {n}" for i, n in enumerate(ordered, 1))

    # Optionally expose a `todo` tool (only if we have a coherent tool listing to
    # extend — never advertise a tool the rest of the transcript can't reference).
    tools_listing, todo_on = tools, False
    if use_todo and tools:
        try:
            arr = json.loads(tools)
            arr.append(_TODO_TOOL_DEF)
            tools_listing = json.dumps(arr, ensure_ascii=False)
            todo_on = True
        except Exception:
            todo_on = False

    lines = [f"System: {_AGENTIC_SYS}"]
    if tools_listing:
        lines.append(f"Tools: {tools_listing}")
    injected = False
    for kind, payload in events:
        if kind == "user":
            lines.append(f"User: {payload}")
            if not injected:                   # plan right after the goal is stated
                lines.append(f"Assistant: {_THINK_OPEN} To do this I will call, in order: "
                             f"{plan}. {_THINK_CLOSE}")
                if todo_on:                    # record the plan with the todo tool
                    call = {"name": "todo", "input": {"items": ordered}}
                    obs = {"plan": [{"content": n, "status": "pending"} for n in ordered],
                           "remaining": len(ordered)}
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


def stream_reasoning(langs: List[str], limit_mb: Optional[int] = None) -> Iterator[dict]:
    """Stage-5 reasoning: blends three real signals so the model learns to think
    before answering — (1) chain-of-thought (GSM8K), (2) basic planning and (3) tool
    use (multi-step hermes sessions reframed with a <think> plan). Interleaved ~2
    CoT : 1 plan+tools so step-by-step reasoning stays dominant. EN only."""
    if "en" not in {l.lower() for l in langs}:
        return
    from datasets import load_dataset

    def _load(name, *args, **kw):
        try:
            return iter(load_dataset(name, *args, split="train", streaming=True, **kw))
        except Exception as e:
            print(f"    [reasoning] {name}: {e}")
            return iter(())

    cot_it  = _load("openai/gsm8k", "main")
    tool_it = _load("NousResearch/hermes-function-calling-v1")
    seen: set = set()
    n_tool = 0
    while True:
        produced = False
        for _ in range(2):                     # 2 chain-of-thought traces
            ex = next(cot_it, None)
            if ex is None:
                break
            text = _cot_transcript(ex.get("question", ""), ex.get("answer", ""))
            if text:
                produced = True
                yield {"text": text, "lang": "en"}
        ex = next(tool_it, None)               # 1 plan + tool-use trace
        if ex is not None:
            # Alternate: half the tool traces record the plan with a `todo` tool,
            # half don't — so the model learns to use it only when it's available.
            text = _reasoning_tool_transcript(ex, use_todo=(n_tool % 2 == 0))
            if text and (h := hash(text)) not in seen:
                seen.add(h)
                n_tool += 1
                produced = True
                yield {"text": text, "lang": "en"}
        if not produced:                       # both sources exhausted
            return


# ──────────────────────────── MCP protocol (real, EN) ───────────────────────
# Stage 8: the SAME real tool interactions, re-serialized into the Model Context
# Protocol wire format (JSON-RPC 2.0): a tools/list result, then tools/call
# requests and their result messages. Real underlying data (no synthetic), just
# in MCP's envelope — so the model learns the protocol Claude/MCP servers speak.
_MCP_SYS = ('You interact with tools over MCP (Model Context Protocol, JSON-RPC 2.0): '
            'send {"jsonrpc":"2.0","method":"tools/call","params":{"name","arguments"}} '
            'and receive a matching result message; otherwise answer directly.')


def _mcp_to_transcript(ex: dict) -> Optional[str]:
    """Serialize one hermes example into an MCP JSON-RPC session. None unless it
    contains at least one real tool call."""
    tools, events = _hermes_events(ex)
    lines = [f"System: {_MCP_SYS}"]
    if tools:
        listing = {"jsonrpc": "2.0", "id": 0,
                   "result": {"tools": json.loads(tools)}}
        lines.append("Server: " + json.dumps(listing, ensure_ascii=False))
    rid, saw_call = 0, False
    for kind, payload in events:
        if kind == "user":
            lines.append(f"User: {payload}")
        elif kind == "assistant":
            lines.append(f"Assistant: {payload}")
        elif kind == "call":
            rid += 1
            req = {"jsonrpc": "2.0", "id": rid, "method": "tools/call",
                   "params": {"name": payload["name"], "arguments": payload["input"]}}
            lines.append("Client: " + json.dumps(req, ensure_ascii=False))
            saw_call = True
        elif kind == "result":
            res = {"jsonrpc": "2.0", "id": rid,
                   "result": {"content": [{"type": "text", "text": payload}]}}
            lines.append("Server: " + json.dumps(res, ensure_ascii=False))
    return "\n".join(lines) if saw_call and len(lines) >= 4 else None


def stream_mcp(langs: List[str], limit_mb: Optional[int] = None) -> Iterator[dict]:
    """Stream real tool interactions as MCP (JSON-RPC 2.0) sessions (EN)."""
    if "en" not in {l.lower() for l in langs}:
        return
    from datasets import load_dataset
    try:
        ds = load_dataset("NousResearch/hermes-function-calling-v1",
                          split="train", streaming=True)
    except Exception as e:
        print(f"    [mcp] {e}")
        return
    seen: set = set()
    for ex in ds:
        text = _mcp_to_transcript(ex)
        if not text:
            continue
        h = hash(text)
        if h in seen:
            continue
        seen.add(h)
        yield {"text": text, "lang": "en"}


# ──────────────────────────── skills (real, EN) ─────────────────────────────
# Stage 9: skills work like Claude Code's — a SKILL.md with YAML frontmatter
# (name, description = when to use it) plus instructions. When a request matches
# a skill's description, the agent follows its instructions. Real procedures from
# Super-NaturalInstructions: each task's `definition` is the instruction body,
# applied to a real input→target. Capped per task so coverage is broad (many
# skills) rather than deep (one skill drilled).
_SKILL_SYS = ("You have Skills — reusable procedures defined with YAML frontmatter "
              "(name, description) and instructions. When a request matches a Skill's "
              "description, use it and follow its instructions.")
_SKILL_CAP_PER_TASK = 40


def _skill_slug(task_name: str) -> str:
    """'task001_quoref_question_generation' → 'quoref-question-generation'."""
    s = re.sub(r"^task\d+_", "", task_name or "").replace("_", "-").strip("-")
    return s or "skill"


def stream_skills(langs: List[str], limit_mb: Optional[int] = None) -> Iterator[dict]:
    """Stream real skills (EN) as Claude-style SKILL.md + an applied input→target."""
    if "en" not in {l.lower() for l in langs}:
        return
    from datasets import load_dataset
    try:
        ds = load_dataset("Muennighoff/natural-instructions", split="train", streaming=True)
    except Exception as e:
        print(f"    [skills] {e}")
        return
    seen: set = set()
    per_task: dict = {}
    for ex in ds:
        defn = " ".join((ex.get("definition") or "").split())
        inp = " ".join((ex.get("inputs") or "").split())
        tgt = " ".join((ex.get("targets") or "").split())
        if not (defn and inp and tgt):
            continue
        task = ex.get("task_name") or ""
        if per_task.get(task, 0) >= _SKILL_CAP_PER_TASK:   # breadth over depth
            continue
        slug = _skill_slug(task)
        text = (f"System: {_SKILL_SYS}\n"
                f"Skill:\n---\nname: {slug}\n"
                f"description: Use this skill to {slug.replace('-', ' ')}.\n---\n"
                f"{defn}\n"
                f"User: {inp}\n"
                f"Assistant: {tgt}")
        h = hash(text)
        if h in seen:
            continue
        seen.add(h)
        per_task[task] = per_task.get(task, 0) + 1
        yield {"text": text, "lang": "en"}


# ── Conversational enrichment: system personas, mood, story-on-request ───────
# These shape REGISTER, not facts. A fraction of the conversational/instruction
# data is given a `System:` persona so the model learns to CONDITION on a system
# prompt (real system-prompt support); emotional dialogues carry a `(mood: …)`
# annotation on that SAME channel so tone is driven by an explicit, neutral-by-
# default mood (src/modalities/moods.py); and a fraction of stories are reframed
# as a request so narration is something the model does ON DEMAND, not only as
# free continuation. All plain ASCII — no new tokenizer symbols.
import hashlib
from src.modalities.moods import emotion_to_mood, mood_system_phrase

_SYSTEM_PERSONAS: List[str] = [
    "You are a helpful, friendly assistant. Answer simply and directly.",
    "You are a kind assistant who talks to young children. Keep words simple.",
    "You are a cheerful helper. Be warm and encouraging.",
    "You are a calm, patient assistant. Explain things gently.",
    "You are a concise assistant. Give short, clear answers.",
    "You are a curious, playful assistant who loves to chat.",
    "You are a storyteller who tells short, simple stories.",
    "You are a thoughtful assistant. Be honest and clear.",
]

_STORY_PROMPTS: List[str] = [
    "Tell me a story.", "Can you tell me a short story?",
    "Tell me a little story please.", "I want to hear a story.",
    "Tell me a bedtime story.",
]


def _hash01(key: str) -> float:
    """Deterministic value in [0,1) keyed on text — stable selection across runs."""
    return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) / 0x100000000


def _persona_for(key: str) -> str:
    return _SYSTEM_PERSONAS[int(_hash01(key) * len(_SYSTEM_PERSONAS)) % len(_SYSTEM_PERSONAS)]


def _prepend_system(text: str, persona: str, mood: str = "neutral") -> str:
    """Add a `System:` line (with an optional non-neutral `(mood: …)` tag) above a
    User:/Assistant: transcript so the model learns to condition on it."""
    tag = mood_system_phrase(mood)
    sys_line = f"System: {persona}" + (f" {tag}" if tag else "")
    return f"{sys_line}\n{text}"


# ──────────────────────────── HF graded corpora ─────────────────────────────
def stream_tinystories(limit_mb: Optional[int] = None,
                       story_request_frac: float = 0.25) -> Iterator[dict]:
    """TinyStories — short, simple children's stories (EN). Level 1 language.
    A fraction are reframed as a `User: <story prompt>` → `Assistant: <story>` turn
    (completion-masked at train time) so the model learns to TELL a story when
    asked, not only to continue prose."""
    try:
        from datasets import load_dataset
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        for ex in ds:
            t = ex.get("text", "")
            if not t.strip():
                continue
            if _hash01(t) < story_request_frac:                 # reframe as a request
                # NO system prompt: telling a story should be a NATURAL response to
                # the request, not a behaviour that needs a persona to unlock.
                prompt = _STORY_PROMPTS[int(_hash01("p" + t) * len(_STORY_PROMPTS))
                                        % len(_STORY_PROMPTS)]
                yield {"text": f"User: {prompt}\nAssistant: {t.strip()}", "lang": "en"}
            else:
                yield {"text": t, "lang": "en"}                 # plain narrative (grammar)
    except Exception as e:
        print(f"    [tinystories] {e}")


# ── real conversation corpora ───────────────────────────────────────────────
# Conversational ability is learned from REAL dialogue (no synthetic templates):
# the model picks up greetings, small talk, world facts, etc. from human chats.
# At low levels the readability filter (passes_filter) keeps only the simplest
# conversations; higher levels admit more. Each conversation is normalized to a
# two-party A:/B: transcript. The legacy `daily_dialog` is script-based and no
# longer loads on datasets>=3, so we use parquet-native everyday-chat corpora.
def _format_dialogue(turns: List[tuple]) -> Optional[str]:
    """Normalize (speaker, text) turns to a "User:/Assistant:" transcript — the
    SAME turn convention the reasoning/agentic/MCP data and the chat runtime use,
    so a model trained on dialogue is primed to reply as the assistant. The first
    speaker is the User, the second the Assistant. Returns None for empty or 3+
    speaker conversations (kept strictly dyadic)."""
    label: Dict[str, str] = {}
    lines: List[str] = []
    for spk, txt in turns:
        txt = " ".join(str(txt).split())
        if not txt:
            continue
        if spk not in label:
            if len(label) >= 2:                    # third speaker → drop conversation
                return None
            label[spk] = "User" if not label else "Assistant"
        lines.append(f"{label[spk]}: {txt}")
    return "\n".join(lines) if len(lines) >= 2 else None


def _parse_person_dialogue(raw: str) -> List[tuple]:
    """Parse DialogSum '#Person1#: ...' transcripts into (speaker, text) turns."""
    turns: List[tuple] = []
    for line in raw.splitlines():
        m = re.match(r"\s*#(Person\d+)#\s*:\s*(.*)", line)
        if m:
            turns.append((m.group(1), m.group(2)))
        elif turns and line.strip():               # wrapped continuation line
            turns[-1] = (turns[-1][0], turns[-1][1] + " " + line.strip())
    return turns


# Per-language everyday-conversation corpora: {lang: [(HF id, extractor → turns)]}.
# These are GENERAL (mixed-register) conversation; the emotion-LABELLED corpus
# (EmpatheticDialogues) is streamed separately and mood-balanced — see
# `_stream_empathetic_balanced`. Everything is round-robin interleaved in
# `stream_dialogue` so no corpus/mood forms a front-loaded block.
_DIALOGUE_CORPORA: Dict[str, List[tuple]] = {
    "en": [
        ("knkarthick/dialogsum",
         lambda ex: _parse_person_dialogue(ex.get("dialogue", ""))),
        # General everyday chit-chat / greetings / small talk — parquet-native
        # social-commonsense dialogue (zip parallel `speakers`/`dialogue` into turns).
        # Provides the NEUTRAL register that keeps "hi" → "hi!" likely (vs apologising).
        ("allenai/soda",
         lambda ex: list(zip(ex.get("speakers") or [], ex.get("dialogue") or []))),
    ],
}

# Multilingual backbone: real human assistant conversations tagged by language.
# Covers any requested language present in the data (en, es, de, fr, ru, zh, …),
# so adding a language to a level's `model.languages` needs no code change.
_OASST_REPOS = ("OpenAssistant/oasst1", "OpenAssistant/oasst2")


def _stream_oasst(langs: set) -> Iterator[dict]:
    """Reconstruct OpenAssistant message trees into User:/Assistant: transcripts,
    one per leaf path, keeping monolingual paths whose language is requested."""
    from datasets import load_dataset
    for repo in _OASST_REPOS:
        try:
            ds = load_dataset(repo, split="train")          # small; full load to chain trees
        except Exception as e:
            print(f"    [dialogue/{repo}] {e}")
            continue
        by_id = {m["message_id"]: m for m in ds}
        parents = {m["parent_id"] for m in ds if m.get("parent_id")}
        for m in ds:                                        # start from leaves (no children)
            if m["message_id"] in parents:
                continue
            lang = m.get("lang")
            if lang not in langs:
                continue
            chain, cur = [], m
            while cur is not None:
                chain.append(cur)
                cur = by_id.get(cur.get("parent_id")) if cur.get("parent_id") else None
            if any(c.get("lang") != lang for c in chain):   # keep paths in a single language
                continue
            chain.reverse()
            text = _format_dialogue([(c["role"], c["text"]) for c in chain])
            if text:
                yield {"text": text, "lang": lang}


def _stream_corpus(name: str, extract, lang: str) -> Iterator[dict]:
    """Stream one general dialogue corpus as {text, lang} records."""
    from datasets import load_dataset
    try:
        ds = load_dataset(name, split="train", streaming=True)
    except Exception as e:
        print(f"    [dialogue/{name}] {e}")
        return
    for ex in ds:
        text = _format_dialogue(extract(ex))
        if text:
            yield {"text": text, "lang": lang}


# EmpatheticDialogues carries an `emotion` label (32 categories — roughly HALF
# positive: joyful/proud/grateful/excited/hopeful/content…; half negative:
# sad/afraid/angry/anxious/lonely…) and pairs an emotional `situation` with an apt
# response, i.e. emotion-that-fits-context. We stream it BALANCED across that label
# (cap per emotion) so the model sees the full mood range evenly instead of
# over-fitting the support/apologetic subset (the "hi → I'm sorry…" failure). Deleting
# the corpus would only skip the problem; balancing it fixes the root.
def _stream_empathetic_balanced(per_emotion_cap: int = 700) -> Iterator[dict]:
    from collections import Counter
    from datasets import load_dataset
    try:
        ds = load_dataset("Estwld/empathetic_dialogues_llm", split="train", streaming=True)
    except Exception as e:
        print(f"    [dialogue/empathetic] {e}")
        return
    counts: Counter = Counter()
    for ex in ds:
        emo = (ex.get("emotion") or "unknown").strip().lower()
        if counts[emo] >= per_emotion_cap:                  # mood balance: even per emotion
            continue
        turns = [(c.get("role"), c.get("content")) for c in (ex.get("conversations") or [])]
        text = _format_dialogue(turns)
        if text:
            counts[emo] += 1
            # Annotate the SYSTEM channel with this dialogue's mood so the model
            # learns to set tone from an explicit, neutral-by-default mood (the
            # runtime mood head injects the same `(mood: …)` at inference).
            mood = emotion_to_mood(emo)
            yield {"text": _prepend_system(text, _persona_for(text), mood),
                   "mood": mood, "lang": "en"}


def _interleave(*streams: Iterator[dict]) -> Iterator[dict]:
    """Round-robin across live generators until all exhaust, so no single corpus (or
    mood) forms a front-loaded block in the output — the same anti-forgetting mixing
    the training loader does across files, applied here across dialogue sources."""
    live = [s for s in streams if s is not None]
    while live:
        nxt = []
        for s in live:
            try:
                rec = next(s)
            except StopIteration:
                continue
            nxt.append(s)
            yield rec
        live = nxt


def stream_dialogue(langs: List[str], limit_mb: Optional[int] = None) -> Iterator[dict]:
    """Stream real human conversations as User:/Assistant: transcripts, MOOD-BALANCED
    and round-robin INTERLEAVED across sources (emotion-balanced EmpatheticDialogues +
    general SODA/DialogSum + the OASST assistant backbone) so the model's default
    register isn't dominated by any one tone. Exact duplicates are dropped."""
    langs_set = set(langs)
    seen: set = set()

    def _fresh(rec: dict) -> bool:
        h = hash(rec["text"])
        if h in seen:
            return False
        seen.add(h)
        return True

    substreams: List[Iterator[dict]] = []
    if "en" in langs_set:                                   # emotion-labelled → balanced
        substreams.append(_stream_empathetic_balanced())
    for lang in langs:                                      # general conversation corpora
        for name, extract in _DIALOGUE_CORPORA.get(lang, []):
            substreams.append(_stream_corpus(name, extract, lang))
    substreams.append(_stream_oasst(langs_set))             # multilingual assistant backbone

    for rec in _interleave(*substreams):                    # mix moods throughout, no blocks
        if _fresh(rec):
            yield rec


def stream_instruct(langs: List[str], limit_mb: Optional[int] = None,
                    system_frac: float = 0.4) -> Iterator[dict]:
    """Simple instruction→response pairs (Alpaca, EN) framed as User:/Assistant: so
    the model learns to ANSWER a request directly — the conversational corpora are
    empathetic/narrative and don't teach 'reply to what was asked'. A fraction get a
    `System:` persona so the model also learns to CONDITION on a system prompt. Long
    answers are skipped to keep entries digestible for the small early levels."""
    if "en" not in langs:
        return
    try:
        from datasets import load_dataset
        ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
        for ex in ds:
            instr = (ex.get("instruction") or "").strip()
            inp   = (ex.get("input") or "").strip()
            out   = (ex.get("output") or "").strip()
            if not instr or not out or len(out) > 600:      # keep short, simple Q&A
                continue
            user = f"{instr}\n{inp}" if inp else instr
            text = f"User: {user}\nAssistant: {out}"
            if _hash01(instr) < system_frac:                # condition on a system prompt
                text = _prepend_system(text, _persona_for(instr))
            yield {"text": text, "lang": "en"}
    except Exception as e:
        print(f"    [instruct] {e}")


def stream_simple_wikipedia(limit_mb: Optional[int] = None) -> Iterator[dict]:
    """Simple English Wikipedia — short, plain-language articles. Level 2."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikimedia/wikipedia", "20231101.simple",
                          split="train", streaming=True)
        for art in ds:
            t = art.get("text", "")
            if len(t) >= 100:
                yield {"text": t, "lang": "en"}
    except Exception as e:
        print(f"    [simple_wikipedia] {e}")


# ──────────────────────────── dispatcher ────────────────────────────────────
def stream_source(key: str, *, langs: List[str], n_tokens: int,
                  arithmetic_level: int = 1, limit_mb: Optional[int] = None,
                  extra_streamers: Optional[Dict[str, Callable]] = None
                  ) -> Optional[Iterator[dict]]:
    """Return an iterator of {'text','lang'} records for a source key, or None
    if unknown. Synthetic generators are sized from the token budget (~6 tokens
    per short example). `extra_streamers` provides the full-corpus streamers
    (wikipedia/arc/gsm8k/math/ethics) defined in prepare_data.py."""
    approx_examples = max(n_tokens // 6, 1000)
    if key == "tinystories":
        return stream_tinystories(limit_mb)
    if key in ("instruct", "instructions"):     # instruction→response (answer directly)
        return stream_instruct(langs, limit_mb)
    if key in ("dialogue", "daily_dialog"):     # real human conversation corpora
        return stream_dialogue(langs, limit_mb)
    if key == "simple_wikipedia":
        return stream_simple_wikipedia(limit_mb)
    if key == "arithmetic":
        return stream_arithmetic(langs, arithmetic_level, limit_mb)
    if key == "analogies":                      # TEMPORARY synthetic (no real corpus yet)
        return gen_analogies(approx_examples)
    if key in ("memory", "memory_synth"):       # synthetic recall/use-of-memory (<mem>)
        return gen_memory(approx_examples)
    if key in ("causal_synth", "causal"):       # real cause→effect (EN) from e-CARE
        return stream_causal(langs, limit_mb)
    if key in ("agentic", "tools"):             # real tool-use loop (EN), Claude-style JSON
        return stream_agentic(langs, limit_mb)
    if key in ("reasoning", "cot"):             # real chain-of-thought (EN), <think>…</think>
        return stream_reasoning(langs, limit_mb)
    if key == "mcp":                            # real tool use over MCP / JSON-RPC (EN)
        return stream_mcp(langs, limit_mb)
    if key == "skills":                         # real skills (EN), Claude-style SKILL.md
        return stream_skills(langs, limit_mb)
    if extra_streamers and key in extra_streamers:
        return extra_streamers[key]()
    return None
