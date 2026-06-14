"""Stage 1 data sources — language and communication.

Real conversation/instruction corpora (TinyStories, dialogue, Alpaca/Dolly/NoRobots
instructions, Simple Wikipedia) plus offline compositional generators (basic chat,
graded dictionary definitions, grammar rules). Everything is normalized to plain
User:/Assistant: transcripts so completion-only loss masking applies.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from src.models.cognition.stage01_language.offline import (
    _COMPARATIVE,  # noqa: F401  (re-exported for the graded.py shim)
    _PAST_IRREG,  # noqa: F401  (re-exported for the graded.py shim + stage test)
    _PLURAL_IRREG,  # noqa: F401  (re-exported for the graded.py shim + stage test)
    gen_basic_chat,
    gen_definitions,
    gen_grammar,
)
from src.models.sdk import (
    STORY_PROMPTS,
    emotion_to_mood,
    hash01,
    interleave,
    persona_for,
    prepend_system,
    stable_hash,
)


# ── TinyStories (EN) ─────────────────────────────────────────────────────────
def stream_tinystories(
    limit_mb: int | None = None, story_request_frac: float = 0.25
) -> Iterator[dict]:
    """TinyStories — short, simple children's stories (EN). Level 1 language.
    A fraction are reframed as a `User: <story prompt>` → `Assistant: <story>` turn
    (completion-masked at train time) so the model learns to TELL a story when
    asked, not only to continue prose."""
    try:
        from datasets import load_dataset

        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        for ex in ds:
            text = ex.get("text", "")
            if not text.strip():
                continue
            if hash01(text) < story_request_frac:  # reframe as a request
                # NO system prompt: telling a story should be a NATURAL response to
                # the request, not a behaviour that needs a persona to unlock.
                prompt = STORY_PROMPTS[
                    int(hash01("p" + text) * len(STORY_PROMPTS)) % len(STORY_PROMPTS)
                ]
                yield {"text": f"User: {prompt}\nAssistant: {text.strip()}", "lang": "en"}
            else:
                yield {"text": text, "lang": "en"}  # plain narrative (grammar)
    except Exception as e:
        print(f"    [tinystories] {e}")


# ── real conversation corpora ──────────────────────────────────────────────────
def _format_dialogue(turns: list[tuple]) -> str | None:
    """Normalize (speaker, text) turns to a "User:/Assistant:" transcript — the
    SAME turn convention the reasoning/agentic/MCP data and the chat runtime use,
    so a model trained on dialogue is primed to reply as the assistant. The first
    speaker is the User, the second the Assistant. Returns None for empty or 3+
    speaker conversations (kept strictly dyadic)."""
    label: dict[str, str] = {}
    lines: list[str] = []
    for speaker, text in turns:
        text = " ".join(str(text).split())
        if not text:
            continue
        if speaker not in label:
            if len(label) >= 2:  # third speaker → drop conversation
                return None
            label[speaker] = "User" if not label else "Assistant"
        lines.append(f"{label[speaker]}: {text}")
    return "\n".join(lines) if len(lines) >= 2 else None


def _parse_person_dialogue(raw: str) -> list[tuple]:
    """Parse DialogSum '#Person1#: ...' transcripts into (speaker, text) turns."""
    turns: list[tuple] = []
    for line in raw.splitlines():
        match = re.match(r"\s*#(Person\d+)#\s*:\s*(.*)", line)
        if match:
            turns.append((match.group(1), match.group(2)))
        elif turns and line.strip():  # wrapped continuation line
            turns[-1] = (turns[-1][0], turns[-1][1] + " " + line.strip())
    return turns


# Per-language everyday-conversation corpora: {lang: [(HF id, extractor → turns)]}.
_DIALOGUE_CORPORA: dict[str, list[tuple]] = {
    "en": [
        ("knkarthick/dialogsum", lambda ex: _parse_person_dialogue(ex.get("dialogue", ""))),
        (
            "allenai/soda",
            lambda ex: list(zip(ex.get("speakers") or [], ex.get("dialogue") or [], strict=False)),
        ),
        (
            "roskoN/dailydialog",
            lambda ex: [
                (i % 2, u) for i, u in enumerate(ex.get("utterances") or ex.get("dialog") or [])
            ],
        ),
    ],
}

_OASST_REPOS = ("OpenAssistant/oasst1", "OpenAssistant/oasst2")


def _stream_oasst(langs: set) -> Iterator[dict]:
    """Reconstruct OpenAssistant message trees into User:/Assistant: transcripts,
    one per leaf path, keeping monolingual paths whose language is requested."""
    from datasets import load_dataset

    for repo in _OASST_REPOS:
        try:
            ds = load_dataset(repo, split="train")  # small; full load to chain trees
        except Exception as e:
            print(f"    [dialogue/{repo}] {e}")
            continue
        by_id = {m["message_id"]: m for m in ds}
        parents = {m["parent_id"] for m in ds if m.get("parent_id")}
        for m in ds:  # start from leaves (no children)
            if m["message_id"] in parents:
                continue
            lang = m.get("lang")
            if lang not in langs:
                continue
            chain, cur = [], m
            while cur is not None:
                chain.append(cur)
                cur = by_id.get(cur.get("parent_id")) if cur.get("parent_id") else None
            if any(c.get("lang") != lang for c in chain):  # keep paths in a single language
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


def _stream_empathetic_balanced(per_emotion_cap: int = 250) -> Iterator[dict]:
    """Stream EmpatheticDialogues BALANCED across its 32 emotion labels (cap per
    emotion) so the model sees the full mood range evenly instead of over-fitting
    the support/apologetic subset (the "hi → I'm sorry…" failure)."""
    from collections import Counter

    from datasets import load_dataset

    try:
        ds = load_dataset("Estwld/empathetic_dialogues_llm", split="train", streaming=True)
    except Exception as e:
        print(f"    [dialogue/empathetic] {e}")
        return
    counts: Counter = Counter()
    for ex in ds:
        emotion = (ex.get("emotion") or "unknown").strip().lower()
        if counts[emotion] >= per_emotion_cap:  # mood balance: even per emotion
            continue
        turns = [(c.get("role"), c.get("content")) for c in (ex.get("conversations") or [])]
        text = _format_dialogue(turns)
        if text:
            counts[emotion] += 1
            mood = emotion_to_mood(emotion)
            yield {
                "text": prepend_system(text, persona_for(text), mood),
                "mood": mood,
                "lang": "en",
            }


def stream_dialogue(langs: list[str], limit_mb: int | None = None) -> Iterator[dict]:
    """Stream real human conversations as User:/Assistant: transcripts, MOOD-BALANCED
    and round-robin INTERLEAVED across sources (emotion-balanced EmpatheticDialogues +
    general SODA/DialogSum + the OASST assistant backbone) so the model's default
    register isn't dominated by any one tone. Exact duplicates are dropped."""
    langs_set = set(langs)
    seen: set = set()

    def _fresh(rec: dict) -> bool:
        h = stable_hash(rec["text"])
        if h in seen:
            return False
        seen.add(h)
        return True

    substreams: list[Iterator[dict]] = []
    if "en" in langs_set:  # emotion-labelled → balanced
        substreams.append(_stream_empathetic_balanced())
    for lang in langs:  # general conversation corpora
        for name, extract in _DIALOGUE_CORPORA.get(lang, []):
            substreams.append(_stream_corpus(name, extract, lang))
    substreams.append(_stream_oasst(langs_set))  # multilingual assistant backbone

    for rec in interleave(*substreams):  # mix moods throughout, no blocks
        if _fresh(rec):
            yield rec


def stream_instruct(
    langs: list[str], limit_mb: int | None = None, system_frac: float = 0.4
) -> Iterator[dict]:
    """Simple instruction→response pairs (Alpaca/Dolly/NoRobots, EN) framed as
    User:/Assistant: so the model learns to ANSWER a request directly. A fraction get
    a `System:` persona so the model also learns to CONDITION on a system prompt."""
    if "en" not in langs:
        return
    from datasets import load_dataset

    seen: set = set()

    def _emit(instr: str, inp: str, out: str):
        instr, inp, out = instr.strip(), (inp or "").strip(), (out or "").strip()
        if not instr or not out or len(out) > 600:  # keep short, simple Q&A
            return None
        user = f"{instr}\n{inp}" if inp else instr
        text = f"User: {user}\nAssistant: {out}"
        if hash01(instr) < system_frac:  # condition on a system prompt
            text = prepend_system(text, persona_for(instr))
        h = stable_hash(text)
        if h in seen:
            return None
        seen.add(h)
        return {"text": text, "lang": "en"}

    try:
        for ex in load_dataset("tatsu-lab/alpaca", split="train", streaming=True):
            rec = _emit(ex.get("instruction", ""), ex.get("input", ""), ex.get("output", ""))
            if rec:
                yield rec
    except Exception as e:
        print(f"    [instruct/alpaca] {e}")
    try:
        for ex in load_dataset("databricks/databricks-dolly-15k", split="train", streaming=True):
            rec = _emit(ex.get("instruction", ""), ex.get("context", ""), ex.get("response", ""))
            if rec:
                yield rec
    except Exception as e:
        print(f"    [instruct/dolly] {e}")
    try:
        for ex in load_dataset("HuggingFaceH4/no_robots", split="train", streaming=True):
            msgs = ex.get("messages") or []
            user = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
            asst = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
            rec = _emit(user, "", asst)
            if rec:
                yield rec
    except Exception as e:
        print(f"    [instruct/no_robots] {e}")


def stream_simple_wikipedia(limit_mb: int | None = None) -> Iterator[dict]:
    """Simple English Wikipedia — short, plain-language articles. Level 2."""
    try:
        from datasets import load_dataset

        ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train", streaming=True)
        for art in ds:
            text = art.get("text", "")
            if len(text) >= 100:
                yield {"text": text, "lang": "en"}
    except Exception as e:
        print(f"    [simple_wikipedia] {e}")


# ── source-key → builder (resolved by the stage registry) ──────────────────────
def _build_tinystories(*, limit_mb=None, **_):
    return stream_tinystories(limit_mb)


def _build_instruct(*, langs, limit_mb=None, **_):
    return stream_instruct(langs, limit_mb)


def _build_dialogue(*, langs, limit_mb=None, **_):
    return stream_dialogue(langs, limit_mb)


def _build_simple_wikipedia(*, limit_mb=None, **_):
    return stream_simple_wikipedia(limit_mb)


def _build_definitions(*, approx_examples, arithmetic_level=1, **_):
    return gen_definitions(approx_examples, level=arithmetic_level)


def _build_grammar(*, approx_examples, arithmetic_level=1, **_):
    return gen_grammar(approx_examples, level=arithmetic_level)


def _build_basic_chat(*, approx_examples, **_):
    return gen_basic_chat(approx_examples)


SOURCES = {
    "tinystories": _build_tinystories,
    "instruct": _build_instruct,
    "instructions": _build_instruct,
    "dialogue": _build_dialogue,
    "daily_dialog": _build_dialogue,
    "simple_wikipedia": _build_simple_wikipedia,
    "definitions": _build_definitions,
    "dictionary": _build_definitions,
    "grammar": _build_grammar,
    "basic_chat": _build_basic_chat,
    "smalltalk": _build_basic_chat,
}
