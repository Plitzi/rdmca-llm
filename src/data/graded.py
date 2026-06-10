"""
Graded data for the LEVEL curriculum.

Each level teaches information of increasing complexity. This module provides:

  - `flesch_kincaid_grade` / `passes_filter` — a readability gate (US grade level)
    used to keep text at/below a level's reading difficulty (level 5 = no gate).
  - synthetic generators (`gen_arithmetic`, `gen_analogies`, `gen_causal`) whose
    difficulty is parameterized by level — so basic arithmetic exists from
    level 1 and ramps up.
  - thin wrappers over simple/graded HuggingFace corpora (TinyStories, simple
    dialogue, Simple-English Wikipedia) for the lower levels.

`stream_source(key, ...)` dispatches a source name (as listed in a level's
`curriculum.stageN.data.sources`) to the right generator/streamer. The full
corpora (`wikipedia`, `arc_*`, `gsm8k`, `math`, `ethics`) live in
scripts/prepare_data.py and are passed in via `extra_streamers` to avoid a
circular import.
"""
from __future__ import annotations
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


# ──────────────────────────── synthetic: arithmetic ─────────────────────────
# arithmetic_level: 1 single-digit +/−, 2 two-digit +/−/×, 3 multi-digit +
# fractions + simple algebra, 4+ larger mixed problems. Templated in words too,
# so the model couples language with arithmetic (you need words to "ask" sums).
_ARITH_TEMPLATES = [
    "{a} {op} {b} = {r}",
    "What is {a} {opw} {b}? The answer is {r}.",
    "{a} {opw} {b} equals {r}.",
    "Q: {a} {op} {b} = ?  A: {r}",
]
_OPW = {"+": "plus", "-": "minus", "*": "times"}


def gen_arithmetic(level: int, n: int, seed: int = 0) -> Iterator[dict]:
    rng = random.Random(seed)
    for _ in range(n):
        if level <= 1:
            a, b = rng.randint(0, 9), rng.randint(0, 9)
            op = rng.choice(["+", "-"])
            if op == "-" and b > a:
                a, b = b, a               # keep results non-negative for kids
        elif level == 2:
            a, b = rng.randint(0, 99), rng.randint(0, 99)
            op = rng.choice(["+", "-", "*"])
            if op == "-" and b > a:
                a, b = b, a
        else:  # level >= 3 — multi-digit / algebra
            if level >= 3 and rng.random() < 0.3:
                # simple linear equation: x + b = c  →  x = c-b
                x = rng.randint(1, 50); b = rng.randint(1, 50)
                yield {"text": f"Solve for x: x + {b} = {x + b}. x = {x}.", "lang": "en"}
                continue
            a, b = rng.randint(0, 999), rng.randint(0, 999)
            op = rng.choice(["+", "-", "*"])
            if op == "-" and b > a:
                a, b = b, a
        r = {"+": a + b, "-": a - b, "*": a * b}[op]
        tmpl = rng.choice(_ARITH_TEMPLATES)
        yield {"text": tmpl.format(a=a, b=b, r=r, op=op, opw=_OPW[op]), "lang": "en"}


# ──────────────────────────── synthetic: analogies / sequences ──────────────
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


# ──────────────────────────── synthetic: simple dialogue ────────────────────
# Gives level 1 a basic conversational ability with a child-sized vocabulary,
# fully offline (no fragile external dialogue dataset). A few templates also
# fold basic arithmetic into conversation, coupling language with counting.
_NAMES = ["Sam", "Mia", "Ben", "Ana", "Leo", "Eva", "Tom", "Lucy"]
_THINGS = ["apples", "dogs", "books", "cats", "balls", "cars", "stars", "cookies"]
_NUMWORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def gen_dialogue(n: int, seed: int = 3) -> Iterator[dict]:
    rng = random.Random(seed)
    for _ in range(n):
        kind = rng.randint(0, 4)
        if kind == 0:
            yield {"text": "A: Hi! How are you?\nB: I am good, thank you. And you?\nA: I am happy today.", "lang": "en"}
        elif kind == 1:
            name = rng.choice(_NAMES)
            yield {"text": f"A: What is your name?\nB: My name is {name}.\nA: Nice to meet you, {name}!", "lang": "en"}
        elif kind == 2:
            a, b = rng.randint(1, 5), rng.randint(1, 4)
            yield {"text": f"A: What is {_NUMWORDS[a]} plus {_NUMWORDS[b]}?\nB: {_NUMWORDS[a]} plus {_NUMWORDS[b]} is {_NUMWORDS[a+b]}.", "lang": "en"}
        elif kind == 3:
            c = rng.randint(1, 5); thing = rng.choice(_THINGS)
            yield {"text": f"A: How many {thing} are there?\nB: There are {_NUMWORDS[c]} {thing}.", "lang": "en"}
        else:
            yield {"text": "A: What do you like?\nB: I like to play and read.\nA: Me too! Let's play.", "lang": "en"}


# ──────────────────────────── synthetic: causal ─────────────────────────────
_CAUSES = [
    ("it rained", "the ground got wet"), ("she studied hard", "she passed the exam"),
    ("the sun set", "it became dark"), ("he dropped the glass", "it broke"),
    ("they watered the plant", "it grew"), ("the fire spread", "the alarm rang"),
    ("we turned off the light", "the room went dark"), ("the ice melted", "the water rose"),
]


def gen_causal(n: int, seed: int = 2) -> Iterator[dict]:
    rng = random.Random(seed)
    tmpl = ["Because {c}, {e}.", "{c}, so {e}.", "Since {c}, {e}.", "{e} because {c}."]
    for _ in range(n):
        c, e = rng.choice(_CAUSES)
        yield {"text": rng.choice(tmpl).format(c=c, e=e), "lang": "en"}


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


# NOTE: the public `daily_dialog` HF dataset is script-based and no longer loads
# on datasets>=3; L1 conversation is provided by the synthetic `gen_dialogue`
# generator instead (offline, child-sized vocabulary). See `dialogue` source.


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
    if key in ("dialogue", "daily_dialog"):     # synthetic simple conversation
        return gen_dialogue(approx_examples)
    if key == "simple_wikipedia":
        return stream_simple_wikipedia(limit_mb)
    if key == "arithmetic":
        return gen_arithmetic(arithmetic_level, approx_examples)
    if key == "analogies":
        return gen_analogies(approx_examples)
    if key == "causal_synth":
        return gen_causal(approx_examples)
    if extra_streamers and key in extra_streamers:
        return extra_streamers[key]()
    return None
