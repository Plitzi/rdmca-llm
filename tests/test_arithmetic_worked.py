"""
Worked (step-by-step) arithmetic generation — teaches the column add/subtract ALGORITHM
so the model COMPUTES and generalizes to unseen multi-digit numbers, instead of
memorizing a single-digit lookup table (the observed 12+7→"12" / 1234+5555→"8" failure).

Guards: the worked steps are arithmetically correct (incl. carry/borrow), the stated
result always equals the true value, multi-digit examples appear even at level 1, and a
meaningful fraction of generated examples are worked.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.graded import _add_worked, _sub_worked, gen_arithmetic


def test_add_worked_correct_with_and_without_carry():
    assert _add_worked(1234, 5555).endswith("So 1234 + 5555 = 6789.")   # no carry
    s = _add_worked(47, 38)
    assert s.endswith("So 47 + 38 = 85.") and "carry 1" in s            # carry
    # the final "So a + b = R" is always the true sum
    for a, b in [(0, 0), (9, 9), (99, 1), (4567, 8999), (1000, 1)]:
        assert _add_worked(a, b).endswith(f"So {a} + {b} = {a + b}.")


def test_sub_worked_correct_with_and_without_borrow():
    assert _sub_worked(68, 45).endswith("So 68 - 45 = 23.")             # no borrow
    s = _sub_worked(52, 37)
    assert s.endswith("So 52 - 37 = 15.") and "borrow 1" in s           # borrow
    for a, b in [(9, 9), (100, 1), (5000, 1234), (80, 7)]:
        assert _sub_worked(a, b).endswith(f"So {a} - {b} = {a - b}.")


def test_gen_arithmetic_emits_correct_worked_examples_at_level1():
    """Even at level 1, worked multi-digit examples appear and every stated result is
    the true value (so the model is trained on CORRECT computation, not noise)."""
    rows = list(gen_arithmetic(800, level=1, seed=3))
    worked = [r["text"] for r in rows if "write" in r["text"] or "borrow" in r["text"]
              or re.search(r"So \d+ [+-] \d+ = \d+\.$", r["text"])]
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
