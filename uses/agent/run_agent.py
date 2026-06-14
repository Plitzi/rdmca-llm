#!/usr/bin/env python3
"""Agent usage — drive the model through a Claude Code-style tool loop.

The model emits `Action: {"name","input"}`, this runner executes the matching
tool, feeds back an `Observation`, and repeats until the model answers. One
example tool (calculator) and one example skill (arithmetic-helper SKILL.md) are
wired in; add more under tools/ and skills/.

Usage:
  python uses/agent/run_agent.py --level 1 --stage 10 --query "What time is it?"
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

from src import agent
from src.config import resolve_config_path
from uses.agent.tools.current_time import TOOL as CURRENT_TIME
from uses.agent.tools.todo import TOOL as TODO
from uses.chat import run_chat as chat  # reuse model loading + generation
from uses.common.interaction import InterruptGuard, SessionInput

# Tools available to the agent (add your own here). Deliberately NOT a calculator
# — arithmetic is a learned skill (stage 3) the model should do itself; a tool for
# it would mask whether the model actually learned arithmetic.
TOOLS = [CURRENT_TIME, TODO]  # TODO = planning aid the model uses when available

# Skills available to the agent (Claude-style SKILL.md files).
SKILLS_DIR = Path(__file__).resolve().parent / "skills"


def load_skill(name: str) -> str | None:
    p = SKILLS_DIR / name / "SKILL.md"
    return p.read_text(encoding="utf-8") if p.exists() else None


def make_generate_fn(
    model,
    mcfg,
    tokenizer,
    *,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    think: str = "off",
    stream: bool = False,
    max_seconds: float | None = None,
    should_stop=None,
):
    """Wrap the model as generate_fn(prompt_text) -> response_text.

    Returns the model's full turn (a <think>…</think> scratchpad, if any, plus
    the answer/action) — `run_agent` splits the scratchpad out for display and
    ignores it when parsing the action. When `think` is on the model first writes
    a budget-capped scratchpad; when `stream` is on each step is printed live."""
    budget = agent.think_budget(think, max_new_tokens) if tokenizer.ready else 0
    stream_on = stream and tokenizer.ready

    def _gen(prompt_text: str) -> str:
        if tokenizer.ready:
            ids = tokenizer.encode(prompt_text, lang="en", add_bos=True, add_eos=False)
        else:
            ids = [2] + [ord(c) % mcfg.vocab_size for c in prompt_text] + [10]
        if budget > 0:
            think_text, gen_ids, _ = chat.generate_thinking(
                model,
                list(ids),
                tokenizer=tokenizer,
                lang="en",
                max_new_tokens=max_new_tokens,
                think_budget=budget,
                temperature=temperature,
                top_p=top_p,
                vocab_size=mcfg.vocab_size,
                context_len=mcfg.context_len,
                stream=stream_on,
                max_seconds=max_seconds,
                think_prefix="\n  💭 ",
                answer_prefix="\n  ▸ ",
                should_stop=should_stop,
            )
        else:
            if stream_on:
                sys.stdout.write("\n  ▸ ")
                sys.stdout.flush()
            gen_ids, _ = chat.generate(
                model,
                list(ids),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                vocab_size=mcfg.vocab_size,
                context_len=mcfg.context_len,
                stream=stream_on,
                max_seconds=max_seconds,
                decode_fn=(tokenizer.decode if stream_on else None),
                should_stop=should_stop,
            )
            think_text = ""
        if not (tokenizer.ready and gen_ids):
            return ""
        # Re-attach the scratchpad so run_agent can surface it; it is ignored by
        # the action parser. (No-op when think is off.)
        answer = tokenizer.decode(gen_ids)
        return (
            f"{agent.THINK_OPEN} {think_text} {agent.THINK_CLOSE}\n{answer}"
            if think_text
            else answer
        )

    return _gen


def main() -> None:
    ap = argparse.ArgumentParser(description="RDMCA agent (tool loop)")
    ap.add_argument("--level", type=int, default=1, help="Educational level (default: 1)")
    ap.add_argument(
        "--stage",
        type=int,
        default=10,
        help="Checkpoint stage (default: 10 = Skills, the most complete)",
    )
    ap.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint .npz")
    ap.add_argument("--dummy", action="store_true", help="Random weights (plumbing test)")
    ap.add_argument("--query", required=True, help="The user message")
    ap.add_argument("--skill", default="time-helper", help="Skill to inject (dir name)")
    ap.add_argument(
        "--system",
        default=None,
        help="System prompt persona prepended to the tool-use instructions",
    )
    ap.add_argument(
        "--mood",
        default=None,
        help="Pin the mood, or omit to read it from the query (neutral default)",
    )
    ap.add_argument(
        "--no-mood",
        dest="no_mood",
        action="store_true",
        help="Disable moods entirely: always neutral, focused on the task",
    )
    ap.add_argument(
        "--max-steps",
        type=int,
        default=6,
        help="Max think→act→observe rounds before giving up (default: 6)",
    )
    ap.add_argument(
        "--context-slots",
        dest="context_slots",
        action="store_true",
        help="Use STR per-sector context slots (§12) for the step tail: "
        "route each Action/Observation block to its sector slot, evict "
        "overflow to memory, and assemble the tail from the slots "
        "(experimental; best with trained sectors). Off by default "
        "(char-windowed recent steps). Header stays pinned either way.",
    )
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--topp", type=float, default=0.9)
    ap.add_argument("--maxtok", type=int, default=64, help="Max new tokens per step")
    ap.add_argument(
        "--think",
        choices=agent.THINKING_LEVELS,
        default="medium",
        help="Reasoning effort per step: off, low, medium (default), high",
    )
    ap.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream each step live (default: on; --no-stream to disable)",
    )
    ap.add_argument(
        "--quant",
        type=chat.parse_quant,
        default=None,
        metavar="none|N",
        help="Weight quantization bit-width: none (default) or 2-8 bits "
        "(e.g. 8, int4). 4-bit (≈⅛ size) is the limited-hardware tier",
    )
    ap.add_argument(
        "--max-seconds",
        type=float,
        default=chat.GEN_DEADLINE_S,
        help="Per-step wall-clock cap, anti-loop guard (0 = unlimited)",
    )
    args = ap.parse_args()

    cfg_path = resolve_config_path(None, args.level)
    load_args = SimpleNamespace(
        config=cfg_path,
        dummy=args.dummy,
        checkpoint=args.checkpoint,
        stage=args.stage,
        level=args.level,
        force=True,
        quant=args.quant,
    )
    print("Loading model…")
    model, mcfg = chat.load_model(load_args)
    from src.modalities.text import TextTokenizer

    tokenizer = TextTokenizer()

    # Ctrl-C aborts the run; typing while it works queues a correction that steers
    # the next step (Claude Code-style). The reader is harmless when non-interactive.
    guard = InterruptGuard()
    session = SessionInput()
    generate_fn = make_generate_fn(
        model,
        mcfg,
        tokenizer,
        temperature=args.temp,
        top_p=args.topp,
        max_new_tokens=args.maxtok,
        think=args.think,
        stream=args.stream,
        max_seconds=(args.max_seconds or None),
        should_stop=guard.stopped,
    )
    skill_md = load_skill(args.skill)
    print(
        f"  Tools: {[t.name for t in TOOLS]} | Skill: {args.skill if skill_md else '—'}"
        f" | Thinking: {args.think} | Stream: {'on' if args.stream else 'off'}"
        f" | Quant: {f'{args.quant}-bit' if args.quant else 'none'}\n"
    )
    print(f"User: {args.query}")

    # Mood applies to EVERY surface, not just the chat: read the query's mood from
    # the shared mood head (neutral by default) and fold it into the system line.
    from src.modalities.moods import MOODS, mood_system_phrase
    from src.model.mood import classify_mood, load_mood_head

    mood = "neutral"
    if not args.no_mood:
        if args.mood in MOODS:
            mood = args.mood
        else:
            head = load_mood_head(
                mcfg.d_model, level=args.level, stage=args.stage, checkpoint=args.checkpoint
            )
            if head is not None:
                mood, _ = classify_mood(model, tokenizer, head, args.query)
    sys_persona = " ".join(p for p in (args.system, mood_system_phrase(mood)) if p) or None
    if mood != "neutral":
        print(f"  (mood: {mood})")

    # Memory recall applies to the agent too: embed the query, pull the most
    # relevant consolidated + recent memories, and lead the agent prompt with them
    # (same <mem> block as the chat). Lazy/optional — empty stores ⇒ no injection.
    # Raw memory body (one '- item' per line); build_agent_prompt wraps it in the
    # <mem>…</mem> block, so we must NOT pre-wrap here.
    memory = ""
    if tokenizer.ready:
        try:
            from src.memory.recall import MemoryRecall

            mems = MemoryRecall(model, tokenizer).recall(args.query)
            memory = "\n".join(f"- {m.text.strip()}" for m in mems)
            if memory:
                print(f"  (memory: {len(mems)} recalled)")
        except Exception as e:
            print(f"  (memory recall off: {e})")

    # Optional STR sector context-slots (§12) for the step tail — the SAME manager
    # the chat wires for its history body, so the agent windows/forgets its tool-loop
    # transcript by sector relevance instead of a flat char cut. OPT-IN
    # (--context-slots), additive: off ⇒ the char-windowed tail, base path unchanged.
    context_mgr = enc = dec = None
    if getattr(args, "context_slots", False) and tokenizer.ready:
        try:
            from src.routing.context_manager import build_context_manager

            context_mgr = build_context_manager(model, tokenizer, context_len=mcfg.context_len)

            def enc(s):
                return tokenizer.encode(s, lang="en", add_bos=False, add_eos=False)

            dec = tokenizer.decode
            gate_on = getattr(model, "gate", None) is not None
            print(
                f"  Context slots: on (§12 STR; routing via "
                f"{'trained MoE gate' if gate_on else 'classifier/single-slot'})."
            )
        except Exception as e:
            print(f"  Context slots: off ({e}).")

    with guard:  # Ctrl-C → abort the run
        result = agent.run_agent(
            generate_fn,
            TOOLS,
            args.query,
            skill_md=skill_md,
            max_steps=args.max_steps,
            think=args.think,
            system=sys_persona,
            memory=memory,
            context_mgr=context_mgr,
            encode=enc,
            decode=dec,
            should_stop=guard.stopped,
            get_steering=session.drain_pending,
        )
    if guard.was_interrupted:
        print("\n  [stopped]")

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

    # ── Context & token accounting (same report every surface uses) ──────────
    from src.memory.experience_log import load_experiences
    from src.observability import ContextReport, count_tokens

    header = agent.build_agent_prompt(TOOLS, args.query, skill_md, args.think, system=sys_persona)
    reasoning = " ".join(s.get("thinking", "") for s in result["steps"]) + (
        result.get("thinking") or ""
    )
    try:
        mem_files = len(load_experiences())
    except Exception:
        mem_files = 0
    sys_tok = count_tokens(
        tokenizer, "System: " + (sys_persona + " " if sys_persona else "") + agent.AGENT_SYSTEM
    )
    tools_tok = count_tokens(tokenizer, agent.tools_spec(TOOLS))
    skills_tok = count_tokens(tokenizer, skill_md or "")
    header_tok = count_tokens(tokenizer, header)
    report = ContextReport(
        surface="agent",
        context_len=mcfg.context_len,
        system_tokens=sys_tok,
        tools_tokens=tools_tok,
        skills_tokens=skills_tok,
        history_tokens=max(0, header_tok - sys_tok - tools_tok - skills_tok),  # query/framing
        tokens_in=header_tok,
        tokens_out=count_tokens(tokenizer, result.get("final") or ""),
        tokens_reasoning=count_tokens(tokenizer, reasoning),
        mood=mood,
        memory_files=mem_files,
        tools_available=len(TOOLS),
        skills_available=1 if skill_md else 0,
        params={
            "temp": args.temp,
            "top_p": args.topp,
            "think": args.think,
            "steps": len(result["steps"]),
        },
    )
    print("\n" + report.render())


if __name__ == "__main__":
    main()
