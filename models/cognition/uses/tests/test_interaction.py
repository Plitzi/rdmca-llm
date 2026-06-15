"""
Tests for the interactive-session ergonomics (models/cognition/uses/common/interaction.py) and the
agent's interrupt / mid-run steering hooks (src/agent.run_agent):

  - InterruptGuard exposes a live stop flag and restores the SIGINT handler;
  - SessionInput queues typed-ahead lines and drains them in order (EOF-safe);
  - run_agent aborts immediately when should_stop() is set (no generation);
  - run_agent injects queued steering messages as the latest User turn so the
    next step answers the correction.
"""

import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.cognition.uses.common.agent import Tool, run_agent
from models.cognition.uses.common.interaction import InterruptGuard, SessionInput


# ── InterruptGuard ───────────────────────────────────────────────────────────
def test_interrupt_guard_flag_and_handler_restore():
    import signal

    before = signal.getsignal(signal.SIGINT)
    with InterruptGuard() as g:
        assert g.stopped() is False and g.was_interrupted is False
        g._on_sigint()  # simulate Ctrl-C
        assert g.stopped() is True and g.was_interrupted is True
    assert signal.getsignal(signal.SIGINT) is before  # handler restored on exit


# ── SessionInput queue ───────────────────────────────────────────────────────
def _drain_when_ready(si: SessionInput, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        out = si.drain_pending()
        if out:
            return out
        time.sleep(0.02)
    return si.drain_pending()


def test_session_input_queues_lines_in_order(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("first\nsecond\nthird\n"))
    si = SessionInput()
    got = _drain_when_ready(si)
    assert got == ["first", "second", "third"]


def test_session_input_next_message_then_eof(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("hello\n"))
    si = SessionInput()
    assert si.next_message() == "hello"
    assert si.next_message() is None  # EOF → None (leave session)


# ── run_agent: interrupt ─────────────────────────────────────────────────────
def test_run_agent_aborts_on_should_stop():
    calls = {"n": 0}

    def gen(_prompt):
        calls["n"] += 1
        return 'Action: {"name": "noop", "input": {}}'

    res = run_agent(gen, [], "do something", should_stop=lambda: True)
    assert res.get("note") == "interrupted"
    assert calls["n"] == 0  # never generated


# ── run_agent: mid-run steering injection ────────────────────────────────────
def test_run_agent_injects_steering_correction():
    prompts: list = []
    calls = {"n": 0}

    def steering():  # user "types" during step 1,
        calls["n"] += 1  # so it arrives for step 2
        return ["actually use the metric system"] if calls["n"] == 2 else []

    def gen(prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            return 'Action: {"name": "noop", "input": {"x": 1}}'  # step 1 → tool
        return "Final answer."  # step 2 → done

    noop = Tool(name="noop", description="no-op", input_schema={}, run=lambda _inp: {"ok": True})
    res = run_agent(gen, [noop], "convert 5 miles", max_steps=4, get_steering=steering)
    assert res["final"] == "Final answer."
    # the SECOND prompt carries the correction as the latest User turn + fresh cue
    assert "User: actually use the metric system" in prompts[1]
    assert prompts[1].rstrip().endswith("Assistant:")
    assert "actually use the metric system" not in prompts[0]  # not before it was typed
