"""
Worked (step-by-step) arithmetic generation — teaches the column add/subtract ALGORITHM
so the model COMPUTES and generalizes to unseen multi-digit numbers, instead of
memorizing a single-digit lookup table (the observed 12+7→"12" / 1234+5555→"8" failure).

Guards: the worked steps are arithmetically correct (incl. carry/borrow), the stated
result always equals the true value, multi-digit examples appear even at level 1, and a
meaningful fraction of generated examples are worked.
"""

import itertools
import re

from src.plugins.cognition.stage03_abstraction.sources import (
    _add_worked,
    _sub_worked,
    gen_arithmetic,
)


def test_add_worked_correct_with_and_without_carry():
    assert _add_worked(1234, 5555).endswith("So 1234 + 5555 = 6789.")  # no carry
    s = _add_worked(47, 38)
    assert s.endswith("So 47 + 38 = 85.") and "carry 1" in s  # carry
    # the final "So a + b = R" is always the true sum
    for a, b in [(0, 0), (9, 9), (99, 1), (4567, 8999), (1000, 1)]:
        assert _add_worked(a, b).endswith(f"So {a} + {b} = {a + b}.")


def test_sub_worked_correct_with_and_without_borrow():
    assert _sub_worked(68, 45).endswith("So 68 - 45 = 23.")  # no borrow
    s = _sub_worked(52, 37)
    assert s.endswith("So 52 - 37 = 15.") and "borrow 1" in s  # borrow
    for a, b in [(9, 9), (100, 1), (5000, 1234), (80, 7)]:
        assert _sub_worked(a, b).endswith(f"So {a} - {b} = {a - b}.")


def test_gen_arithmetic_emits_correct_worked_examples_at_level1():
    """Even at level 1, worked multi-digit examples appear and every stated result is
    the true value (so the model is trained on CORRECT computation, not noise)."""
    rows = list(gen_arithmetic(800, level=1, seed=3))
    worked = [
        r["text"]
        for r in rows
        if "write" in r["text"]
        or "borrow" in r["text"]
        or re.search(r"So \d+ [+-] \d+ = \d+\.$", r["text"])
    ]
    assert len(worked) > 50, "expected a meaningful fraction of worked examples"
    # multi-digit operands must appear (the whole point — generalization beyond 1 digit)
    assert any(re.search(r"So \d{2,} [+-]", w) for w in worked)
    # every worked 'So a op b = r' is arithmetically true
    checked = 0
    for w in worked:
        m = re.search(r"So (\d+) ([+-]) (\d+) = (\d+)\.$", w)
        if not m:
            continue
        a, op, b, r = int(m[1]), m[2], int(m[3]), int(m[4])
        assert r == (a + b if op == "+" else a - b), f"wrong worked result: {w}"
        checked += 1
    assert checked > 50


def test_question_prompt_never_maps_to_a_bare_pm_result():
    """REGRESSION: the 'User: What is a OP b?' QUESTION prompt must, for + and -, ALWAYS
    resolve to the WORKED step-by-step answer — never a bare number. Otherwise the same
    prompt has two targets (steps vs bare) and greedy decoding collapses to the memorized
    bare result that fails OOD (the observed 12+7→'14'). +/- bare facts stay declarative
    ('5 + 9 = 14'); only ×/÷ (no worked form) may use the Q&A surface."""
    rows = [r["text"] for r in gen_arithmetic(20000, level=1, seed=11)]

    def worked(t):
        return "Assistant:" in t and ". So " in t

    bad = [t for t in rows if re.search(r"What is \d+ [+-] \d+\?", t) and not worked(t)]
    assert not bad, f"+/- question prompt mapped to a bare answer (collision): {bad[:3]}"
    # and worked is a major share (the stage's core faculty gets a lot of gradient)
    n_worked = sum(1 for t in rows if worked(t))
    assert n_worked > len(rows) * 0.30, f"worked share too low: {n_worked}/{len(rows)}"


def test_gen_arithmetic_level1_single_digit_nonneg():
    """Level-1 graded equations stay single-digit and never negative (the borrow
    primitive '(d+10) - d = …' is the one intentional 10–18 minuend exception)."""
    eq = re.compile(r"^(\d+) ([+\-]) (\d+) = (-?\d+)$")
    seen_eq = 0
    for rec in itertools.islice(gen_arithmetic(2000, level=1, seed=3), 2000):
        t = rec["text"]
        m = eq.match(t)
        if not m:  # counting / comparison / worded / Q&A
            continue
        a, op, b, c = int(m[1]), m[2], int(m[3]), int(m[4])
        assert c >= 0  # never negative
        assert (a + b if op == "+" else a - b) == c  # arithmetic is correct
        if op == "-" and 10 <= a <= 19 and b < 10:
            continue  # atomic borrow primitive — intentional 10–18 minuend
        seen_eq += 1
        assert a < 10 and b < 10  # single-digit operands
    assert seen_eq > 0


def test_atomic_carry_primitives_are_present_and_correct():
    """The column algorithm invokes single-digit sums step by step; the THREE-TERM carry
    form `a + b + 1` was the measured weak spot (e.g. '2 + 1 + 1 = 3'). The generator must
    ground these primitives (a meaningful share) and every stated fact must be correct."""
    rows = [r["text"] for r in gen_arithmetic(20000, level=1, seed=5)]
    carry = [t for t in rows if re.fullmatch(r"\d \+ \d \+ 1 = \d+", t)]
    assert len(carry) > len(rows) * 0.03, f"too few carry primitives: {len(carry)}"
    for t in carry:
        a, b, _, r = re.findall(r"\d+", t)
        assert int(r) == int(a) + int(b) + 1, f"wrong carry fact: {t}"
