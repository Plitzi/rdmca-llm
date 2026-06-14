"""Stage 3 data sources — abstraction and symbolic composition (arithmetic).

Real symbolic arithmetic (atlas-math-sets) blended with unlimited synthetic graded
fill: atomic single-digit primitives, WORKED column add/subtract (carry/borrow), and
varied surface forms — so the model learns the ALGORITHM and generalizes, not a
single-digit lookup table.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterator

from src.core.data.blend import blend

# ── real arithmetic (atlas-math-sets) ──────────────────────────────────────────
_ARITH_RE = re.compile(r"\s*(\d+)\s*([+\-x×*/])\s*(\d+)\s*=")
# atlas-math has ~17.8M rows; the graded subset is small/finite, so a bounded prefix
# covers it many times without churning the whole dataset every run.
_ARITH_SCAN_CAP = 400_000


def _arith_difficulty(a: int, b: int, op: str) -> int:
    """Coarse difficulty from operand magnitude + operation type."""
    largest = max(a, b)
    if op in "+-":
        return 1 if largest < 10 else (2 if largest < 100 else 3)
    return 2 if largest < 100 else 3  # ×/÷ never count as level 1


def stream_arithmetic(langs: list[str], level: int, limit_mb: int | None = None) -> Iterator[dict]:
    """Stream real arithmetic equations graded to `level`. Symbolic content, so
    tagged with the primary configured language (it is language-agnostic). Scans a
    bounded prefix of the dataset (the graded space is small and finite)."""
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
        match = _ARITH_RE.match(out)
        if not match:
            continue  # skip roots/powers/word problems
        a, op, b = int(match.group(1)), match.group(2), int(match.group(3))
        if _arith_difficulty(a, b, op) <= level:
            yield {"text": out, "lang": lang}


# ── synthetic graded fill ──────────────────────────────────────────────────────
_OP_WORD = {"+": "plus", "-": "minus", "x": "times", "/": "divided by"}


def _grade_arith(rng: random.Random, level: int):
    """(a, b, op, result) graded so level 1 = single-digit +/− (never negative),
    level 2 = two-digit +/−/×, level 3+ = larger + exact division."""
    if level <= 1:
        op = rng.choice(["+", "-"])
        a, b = rng.randint(0, 9), rng.randint(0, 9)
    elif level == 2:
        op = rng.choice(["+", "-", "x"])
        a, b = rng.randint(0, 99), rng.randint(0, 99)
    else:
        op = rng.choice(["+", "-", "x", "/"])
        a, b = rng.randint(0, 999), rng.randint(1, 99)
    if op == "-" and b > a:  # keep subtraction non-negative
        a, b = b, a
    if op == "/":  # make division exact
        b = max(b, 1)
        a -= a % b
    result = {"+": a + b, "-": a - b, "x": a * b, "/": (a // b if b else 0)}[op]
    return a, b, op, result


def _worked_operands(rng: random.Random, level: int):
    """Operands for a WORKED (step-by-step) +/- example. Wider than the bare single-digit
    facts so the carry/borrow ALGORITHM is exercised (and can generalize), growing with
    level. +/- only — column steps teach the algorithm; ×/÷ stay as facts."""
    digits = {1: (1, 3), 2: (2, 4)}.get(level, (3, 6))  # L1 up to 3-digit → carries appear
    d = rng.randint(*digits)
    hi = 10**d - 1
    a, b = rng.randint(0, hi), rng.randint(0, hi)
    op = rng.choice(["+", "-"])
    # Bias ADDITION toward a units CARRY (the measured weak spot). Multi-digit only.
    if op == "+" and d >= 2 and rng.random() < 0.6:
        units_a, units_b = a % 10, b % 10
        if units_a + units_b < 10:
            b = min(b + (10 - units_a - units_b), hi)  # push units to overflow → carry
    if op == "-" and b > a:
        a, b = b, a
    return a, b, op


def _add_worked(a: int, b: int) -> str:
    """Compact column-addition steps (right→left, with carry). Stated result is a+b."""
    width = max(len(str(a)), len(str(b)))
    digits_a, digits_b = str(a).zfill(width), str(b).zfill(width)
    carry, parts = 0, []
    for i in range(width - 1, -1, -1):
        x, y = int(digits_a[i]), int(digits_b[i])
        column = x + y + carry
        head = f"{x} + {y}" + (f" + {carry}" if carry else "")
        parts.append(
            f"{head} = {column}, write {column % 10}" + (", carry 1" if column >= 10 else "")
        )
        carry = 1 if column >= 10 else 0
    if carry:
        parts.append("write the carried 1")
    return "; ".join(parts) + f". So {a} + {b} = {a + b}."


def _sub_worked(a: int, b: int) -> str:
    """Compact column-subtraction steps (right→left, with borrow). Requires a≥b."""
    width = max(len(str(a)), len(str(b)))
    digits_a, digits_b = str(a).zfill(width), str(b).zfill(width)
    borrow, parts = 0, []
    for i in range(width - 1, -1, -1):
        x, y = int(digits_a[i]) - borrow, int(digits_b[i])
        if x < y:
            parts.append(f"{x + 10} - {y} = {x + 10 - y} (borrow 1)")
            borrow = 1
        else:
            parts.append(f"{x} - {y} = {x - y}")
            borrow = 0
    return "; ".join(parts) + f". So {a} - {b} = {a - b}."


# Varied ways a user ASKS for a calculation — so the worked compute is triggered by the
# PRESENCE of an arithmetic question, not one rigid template.
_ARITH_Q_TEMPLATES = (
    "What is {a} {op} {b}?",
    "what's {a} {op} {b}?",
    "How much is {a} {op} {b}?",
    "Can you calculate {a} {op} {b}?",
    "Please solve {a} {op} {b}.",
    "I need the answer for {a} {op} {b}.",
    "Compute {a} {op} {b}.",
    "{a} {op} {b} = ?",
    "Could you work out {a} {op} {b}?",
)
_ARITH_PREFIXES = (
    "",
    "",
    "",
    "",
    "Hi, ",
    "Hello! ",
    "Hey, ",
    "I have a question: ",
    "Quick one — ",
    "Can you help me? ",
)


def _worked_question(rng: random.Random, a: int, b: int, op: str) -> str:
    """A user turn asking for `a op b`, with varied phrasing + optional conversational
    lead-in, so the trigger to COMPUTE generalizes beyond a single template."""
    question = rng.choice(_ARITH_Q_TEMPLATES).format(a=a, op=op, b=b)
    return rng.choice(_ARITH_PREFIXES) + question


def _atomic_fact(rng: random.Random) -> str:
    """An ATOMIC single-digit operation — the primitives the column algorithm invokes
    step by step (especially the three-term carry form `a + b + 1`, the measured weak
    spot). Surfaces match the worked-step text so the grounding transfers in-place."""
    roll = rng.random()
    if roll < 0.60:  # three-term carry add (the weak spot)
        a, b = rng.randint(0, 9), rng.randint(0, 9)
        return f"{a} + {b} + 1 = {a + b + 1}"
    if roll < 0.82:  # two-term single-digit add
        a, b = rng.randint(0, 9), rng.randint(0, 9)
        return f"{a} + {b} = {a + b}"
    if roll < 0.90:  # two-term single-digit sub
        a, b = rng.randint(0, 9), rng.randint(0, 9)
        if b > a:
            a, b = b, a
        return f"{a} - {b} = {a - b}"
    # borrow primitive: (units+10) - b, as it appears in the worked subtraction steps
    a, b = rng.randint(0, 9), rng.randint(0, 9)
    return f"{a + 10} - {b} = {a + 10 - b}"


def gen_arithmetic(n: int, level: int = 1, seed: int = 1) -> Iterator[dict]:
    """Synthetic graded arithmetic with VARIED surface forms — symbolic, worded, Q&A,
    counting, comparisons, and WORKED step-by-step solutions. The worked form teaches
    the ALGORITHM so the model COMPUTES and generalizes to unseen numbers. Worked steps
    are INLINE (not <think> — that scaffolding is the reasoning stage's)."""
    rng = random.Random(seed)
    for _ in range(n):
        roll = rng.random()
        if roll < 0.08:  # counting sequence
            start, step = rng.randint(0, 5), rng.choice([1, 1, 1, 2])
            seq = [start + step * i for i in range(rng.randint(4, 6))]
            yield {"text": "Counting: " + " ".join(map(str, seq)), "lang": "en"}
            continue
        if roll < 0.34:  # ATOMIC single-digit primitives (incl. carry)
            yield {"text": _atomic_fact(rng), "lang": "en"}
            continue
        if roll < 0.68:  # WORKED step-by-step (+/-) → generalizes
            a, b, op = _worked_operands(rng, level)
            steps = _add_worked(a, b) if op == "+" else _sub_worked(a, b)
            question = _worked_question(rng, a, b, op)
            yield {"text": f"User: {question}\nAssistant: {steps}", "lang": "en"}
            continue
        a, b, op, result = _grade_arith(rng, level)
        if roll < 0.70 and level <= 1:  # comparison
            sign = ">" if a > b else ("<" if a < b else "=")
            yield {"text": f"{a} {sign} {b}", "lang": "en"}
            continue
        form = rng.random()
        # The "What is a OP b?" QUESTION prompt is reserved for the WORKED solution
        # above — for +/- it must ALWAYS map to step-by-step, never a bare number.
        if op in ("+", "-"):
            if form < 0.5:  # symbolic equation
                yield {"text": f"{a} {op} {b} = {result}", "lang": "en"}
            else:  # worded statement
                yield {"text": f"{a} {_OP_WORD[op]} {b} equals {result}.", "lang": "en"}
        elif form < 0.40:  # symbolic equation
            yield {"text": f"{a} {op} {b} = {result}", "lang": "en"}
        elif form < 0.70:  # worded statement
            yield {"text": f"{a} {_OP_WORD[op]} {b} equals {result}.", "lang": "en"}
        else:  # Q&A (×/÷ only → answer-masked)
            yield {"text": f"User: What is {a} {op} {b}?\nAssistant: {result}", "lang": "en"}


def _build_arithmetic(*, langs, approx_examples, arithmetic_level=1, limit_mb=None, **_):
    return blend(
        stream_arithmetic(langs, arithmetic_level, limit_mb),
        gen_arithmetic(approx_examples, arithmetic_level),
        approx_examples,
    )


SOURCES = {"arithmetic": _build_arithmetic}
