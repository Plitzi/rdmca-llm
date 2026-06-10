#!/usr/bin/env python3
"""Agent usage — drive the model through a Claude Code-style tool loop.

The model emits `Action: {"name","input"}`, this runner executes the matching
tool, feeds back an `Observation`, and repeats until the model answers. One
example tool (calculator) and one example skill (arithmetic-helper SKILL.md) are
wired in; add more under tools/ and skills/.

Usage:
  python uses/agent/run_agent.py --level 1 --stage 9 --query "What time is it?"
  python uses/agent/run_agent.py --dummy --query "hello"     # plumbing only (random weights)

Reasoning (default medium) and live streaming (default on) are inherited from
the chat runner; the loop runs several think→act→observe rounds until it answers.

Note: tool-call quality depends on model scale/training. Small levels may not
emit valid Actions yet; the loop degrades gracefully to a direct answer.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from uses.chat import run_chat as chat        # reuse model loading + generation
from src import agent
from src.config import resolve_config_path
from uses.agent.tools.current_time import TOOL as CURRENT_TIME

# Tools available to the agent (add your own here). Deliberately NOT a calculator
# — arithmetic is a learned skill (stage 3) the model should do itself; a tool for
# it would mask whether the model actually learned arithmetic.
TOOLS = [CURRENT_TIME]

# Skills available to the agent (Claude-style SKILL.md files).
SKILLS_DIR = Path(__file__).resolve().parent / "skills"


def load_skill(name: str) -> str | None:
    p = SKILLS_DIR / name / "SKILL.md"
    return p.read_text(encoding="utf-8") if p.exists() else None


def make_generate_fn(model, mcfg, tokenizer, *, temperature: float, top_p: float,
                     max_new_tokens: int, think: str = "off", stream: bool = False,
                     max_seconds: float | None = None):
    """Wrap the model as generate_fn(prompt_text) -> response_text.

    Returns the model's full turn (a <think>…</think> scratchpad, if any, plus
    the answer/action) — `run_agent` splits the scratchpad out for display and
    ignores it when parsing the action. When `think` is on the model first writes
    a budget-capped scratchpad; when `stream` is on each step is printed live."""
    budget    = agent.think_budget(think, max_new_tokens) if tokenizer.ready else 0
    stream_on = stream and tokenizer.ready

    def _gen(prompt_text: str) -> str:
        if tokenizer.ready:
            ids = tokenizer.encode(prompt_text, lang="en", add_bos=True, add_eos=False)
        else:
            ids = [2] + [ord(c) % mcfg.vocab_size for c in prompt_text] + [10]
        if budget > 0:
            think_text, gen_ids, _ = chat.generate_thinking(
                model, list(ids), tokenizer=tokenizer, lang="en",
                max_new_tokens=max_new_tokens, think_budget=budget,
                temperature=temperature, top_p=top_p, vocab_size=mcfg.vocab_size,
                context_len=mcfg.context_len, stream=stream_on, max_seconds=max_seconds,
                think_prefix="\n  💭 ", answer_prefix="\n  ▸ ")
        else:
            if stream_on:
                sys.stdout.write("\n  ▸ "); sys.stdout.flush()
            gen_ids, _ = chat.generate(
                model, list(ids), max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p, vocab_size=mcfg.vocab_size,
                context_len=mcfg.context_len, stream=stream_on, max_seconds=max_seconds,
                decode_fn=(tokenizer.decode if stream_on else None))
            think_text = ""
        if not (tokenizer.ready and gen_ids):
            return ""
        # Re-attach the scratchpad so run_agent can surface it; it is ignored by
        # the action parser. (No-op when think is off.)
        answer = tokenizer.decode(gen_ids)
        return f"{agent.THINK_OPEN} {think_text} {agent.THINK_CLOSE}\n{answer}" if think_text else answer
    return _gen


def main() -> None:
    ap = argparse.ArgumentParser(description="RDMCA agent (tool loop)")
    ap.add_argument("--level", type=int, default=1, help="Educational level (default: 1)")
    ap.add_argument("--stage", type=int, default=9, help="Checkpoint stage (default: 9 = Skills, the most complete)")
    ap.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint .npz")
    ap.add_argument("--dummy", action="store_true", help="Random weights (plumbing test)")
    ap.add_argument("--query", required=True, help="The user message")
    ap.add_argument("--skill", default="time-helper", help="Skill to inject (dir name)")
    ap.add_argument("--max-steps", type=int, default=6,
                    help="Max think→act→observe rounds before giving up (default: 6)")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--topp", type=float, default=0.9)
    ap.add_argument("--maxtok", type=int, default=64, help="Max new tokens per step")
    ap.add_argument("--think", choices=agent.THINKING_LEVELS, default="medium",
                    help="Reasoning effort per step: off, low, medium (default), high")
    ap.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True,
                    help="Stream each step live (default: on; --no-stream to disable)")
    ap.add_argument("--quant", choices=("none", "int8", "int4"), default="none",
                    help="Weight quantization for limited hardware: none, int8, int4")
    ap.add_argument("--max-seconds", type=float, default=chat.GEN_DEADLINE_S,
                    help="Per-step wall-clock cap, anti-loop guard (0 = unlimited)")
    args = ap.parse_args()

    cfg_path = resolve_config_path(None, args.level)
    load_args = SimpleNamespace(config=cfg_path, dummy=args.dummy,
                                checkpoint=args.checkpoint, stage=args.stage,
                                level=args.level, force=True, quant=args.quant)
    print("Loading model…")
    model, mcfg = chat.load_model(load_args)
    from src.modalities.text import TextTokenizer
    tokenizer = TextTokenizer()

    generate_fn = make_generate_fn(model, mcfg, tokenizer, temperature=args.temp,
                                   top_p=args.topp, max_new_tokens=args.maxtok,
                                   think=args.think, stream=args.stream,
                                   max_seconds=(args.max_seconds or None))
    skill_md = load_skill(args.skill)
    print(f"  Tools: {[t.name for t in TOOLS]} | Skill: {args.skill if skill_md else '—'}"
          f" | Thinking: {args.think} | Stream: {'on' if args.stream else 'off'}"
          f" | Quant: {args.quant}\n")
    print(f"User: {args.query}")

    result = agent.run_agent(generate_fn, TOOLS, args.query,
                             skill_md=skill_md, max_steps=args.max_steps,
                             think=args.think)

    # Structured recap of the rounds. Thinking is only re-printed here when it
    # was NOT already streamed live (avoids duplicating the scratchpad).
    print("\n── trace ──")
    for i, step in enumerate(result["steps"], 1):
        if step.get("thinking") and not args.stream:
            print(f"  [step {i}] 💭 {step['thinking']}")
        print(f"  [step {i}] Action: {step['action']}")
        print(f"            Observation: {step['observation']}")
    if result.get("final"):
        if result.get("thinking") and not args.stream:
            print(f"\n💭 {result['thinking']}")
        print(f"\nAgent: {result['final']}")
    else:
        print(f"\nAgent: (no final answer — {result.get('note', 'stopped')})")


if __name__ == "__main__":
    main()
