"""
Graded data for the LEVEL curriculum.

Each level teaches information of increasing complexity. This module provides:

  - `flesch_kincaid_grade` / `passes_filter` — a readability gate (US grade level)
    used to keep text at/below a level's reading difficulty (level 5 = no gate).
  - streamers over real HuggingFace corpora for the lower levels: TinyStories,
    conversation (`stream_dialogue`), arithmetic (`stream_arithmetic`), causal
    reasoning (`stream_causal`) and Simple-English Wikipedia.
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
    ('user'|'assistant'|'call'|'result', payload) — shared by the agentic (stage 6)
    and MCP (stage 7) serializers below."""
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


# ──────────────────────────── MCP protocol (real, EN) ───────────────────────
# Stage 7: the SAME real tool interactions, re-serialized into the Model Context
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


# ──────────────────────────── HF graded corpora ─────────────────────────────
def stream_tinystories(limit_mb: Optional[int] = None) -> Iterator[dict]:
    """TinyStories — short, simple children's stories (EN). Level 1 language."""
    try:
        from datasets import load_dataset
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        for ex in ds:
            t = ex.get("text", "")
            if t.strip():
                yield {"text": t, "lang": "en"}
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
    """Normalize (speaker, text) turns to a two-party "A:/B:" transcript.
    Returns None for empty or 3+ speaker conversations (kept strictly dyadic)."""
    label: Dict[str, str] = {}
    lines: List[str] = []
    for spk, txt in turns:
        txt = " ".join(str(txt).split())
        if not txt:
            continue
        if spk not in label:
            if len(label) >= 2:                    # third speaker → drop conversation
                return None
            label[spk] = "A" if not label else "B"
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
# Language-agnostic by design — add a language by listing corpora here; the
# multilingual OpenAssistant backbone below already covers many languages.
_DIALOGUE_CORPORA: Dict[str, List[tuple]] = {
    "en": [
        ("Estwld/empathetic_dialogues_llm",
         lambda ex: [(c.get("role"), c.get("content")) for c in (ex.get("conversations") or [])]),
        ("knkarthick/dialogsum",
         lambda ex: _parse_person_dialogue(ex.get("dialogue", ""))),
    ],
}

# Multilingual backbone: real human assistant conversations tagged by language.
# Covers any requested language present in the data (en, es, de, fr, ru, zh, …),
# so adding a language to a level's `model.languages` needs no code change.
_OASST_REPOS = ("OpenAssistant/oasst1", "OpenAssistant/oasst2")


def _stream_oasst(langs: set) -> Iterator[dict]:
    """Reconstruct OpenAssistant message trees into A:/B: transcripts, one per
    leaf path, keeping monolingual paths whose language is requested."""
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


def stream_dialogue(langs: List[str], limit_mb: Optional[int] = None) -> Iterator[dict]:
    """Stream real human conversations as A:/B: transcripts in the requested
    languages. Exact duplicates are dropped (e.g. OpenAssistant sibling answers
    sharing a prompt). No synthetic fallback — if the corpora are unreachable the
    source simply yields nothing (the readability filter then grades what remains)."""
    from datasets import load_dataset
    langs_set = set(langs)
    seen: set = set()

    def _fresh(rec: dict) -> bool:
        h = hash(rec["text"])
        if h in seen:
            return False
        seen.add(h)
        return True

    for lang in langs:                                      # language-specific corpora
        for name, extract in _DIALOGUE_CORPORA.get(lang, []):
            try:
                ds = load_dataset(name, split="train", streaming=True)
                for ex in ds:
                    text = _format_dialogue(extract(ex))
                    if text and _fresh({"text": text}):
                        yield {"text": text, "lang": lang}
            except Exception as e:
                print(f"    [dialogue/{name}] {e}")
    for rec in _stream_oasst(langs_set):                    # multilingual backbone
        if _fresh(rec):
            yield rec


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
    if key in ("dialogue", "daily_dialog"):     # real human conversation corpora
        return stream_dialogue(langs, limit_mb)
    if key == "simple_wikipedia":
        return stream_simple_wikipedia(limit_mb)
    if key == "arithmetic":
        return stream_arithmetic(langs, arithmetic_level, limit_mb)
    if key == "analogies":                      # TEMPORARY synthetic (no real corpus yet)
        return gen_analogies(approx_examples)
    if key in ("causal_synth", "causal"):       # real cause→effect (EN) from e-CARE
        return stream_causal(langs, limit_mb)
    if key in ("agentic", "tools"):             # real tool-use loop (EN), Claude-style JSON
        return stream_agentic(langs, limit_mb)
    if key == "mcp":                            # real tool use over MCP / JSON-RPC (EN)
        return stream_mcp(langs, limit_mb)
    if extra_streamers and key in extra_streamers:
        return extra_streamers[key]()
    return None
