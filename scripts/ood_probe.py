"""OOD generalization probe for a trained stage.

Loads a stage checkpoint and greedily (deterministically) answers a list of
prompts that are NOT in the training set, to distinguish LEARNING (generalizes)
from MEMORIZATION (only in-distribution items work). Greedy = temp→0, top_k=1,
no repetition penalty, so the answer is the model's argmax — no sampling luck.

Usage:
  .venv/bin/python scripts/ood_probe.py --level 1 --stage 3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


from src import agent
from src.modalities.text import BOS_ID


def _resolve_config(level):
    return f"configs/levels/level{level}.yaml"


def answer(model, mcfg, tokenizer, prompt, lang="en", max_new=64, temperature=0.0):
    """Single-turn answer, formatted exactly like the chat/training turns. Greedy by
    default (temperature=0); pass a temperature to probe robustness — a truly LEARNED
    exact task should give the SAME answer regardless of temperature (its correct token
    dominates), so temperature-sensitivity reveals an under-learned, low-confidence
    operation rather than a decoding choice."""
    from uses.chat.run_chat import generate

    enc_prompt = agent.wrap_prompt(prompt, "text", think="off")
    body = tokenizer.encode(enc_prompt, lang=lang, add_bos=False, add_eos=False)
    gen_history = [BOS_ID, *body]
    ids, _ = generate(
        model,
        gen_history,
        max_new_tokens=max_new,
        temperature=temperature,
        top_p=(1.0 if temperature == 0 else 0.9),
        vocab_size=mcfg.vocab_size,
        context_len=mcfg.context_len,
        stream=False,
        decode_fn=tokenizer.decode,
        top_k=(1 if temperature == 0 else 0),
        rep_penalty=1.0,
        stop_strings=agent.ANSWER_STOP_STRINGS,
    )
    return agent.clean_answer(tokenizer.decode(ids))


def temp_robustness(model, mcfg, tokenizer, temps=(0.0, 0.8, 1.0), n=12, seed=7):
    """ACCEPTANCE TEST for an exact faculty: arithmetic accuracy should be INVARIANT to
    temperature. Generates carry-bearing 2-digit additions, answers each at every temp,
    and reports per-temp accuracy + whether the answer is STABLE across temps. A learned
    model holds accuracy flat as temp rises; a memorizing/uncertain one degrades."""
    import random
    import re as _re

    rng = random.Random(seed)
    cases = []
    while len(cases) < n:  # bias to carry cases (the weak spot)
        a, b = rng.randint(10, 89), rng.randint(10, 89)
        if (a % 10) + (b % 10) >= 10:
            cases.append((a, b))
    print(
        f"\n{'=' * 70}\n  TEMP-ROBUSTNESS — arithmetic should be temp-INVARIANT if learned\n{'=' * 70}"
    )
    per_temp = dict.fromkeys(temps, 0)
    stable = 0
    for a, b in cases:
        got = {}
        for t in temps:
            out = answer(model, mcfg, tokenizer, f"What is {a} + {b}?", max_new=70, temperature=t)
            m = _re.search(r"=\s*(-?\d+)\.?\s*$", out)
            got[t] = int(m.group(1)) if m else None
            per_temp[t] += int(got[t] == a + b)
        stable += int(len(set(got.values())) == 1)
        flags = " ".join(f"t{t}={got[t]}{'✓' if got[t] == a + b else '✗'}" for t in temps)
        print(f"  {a}+{b}={a + b:<4} {flags}")
    print("\n  accuracy by temp: " + " · ".join(f"t{t}={per_temp[t]}/{n}" for t in temps))
    print(
        f"  answer stable across temps: {stable}/{n}  "
        f"(high stability + flat accuracy = the faculty is learned, not memorized)"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--stage", type=int, required=True)
    p.add_argument("--maxtok", type=int, default=64)
    args = p.parse_args()
    args.config = _resolve_config(args.level)
    args.checkpoint = None
    args.dummy = False
    args.quant = "none"
    args.force = True

    from uses.chat.run_chat import load_model

    print("Loading model…")
    model, mcfg = load_model(args)
    from src.modalities.text import TextTokenizer

    tokenizer = TextTokenizer()
    if not tokenizer.ready:
        print("Tokenizer not trained — aborting.")
        sys.exit(1)

    # ── OOD prompt sets per stage ──────────────────────────────────────────────
    arithmetic = [
        "What is 12 + 7?",
        "What is 48 + 27?",
        "What is 90 - 13?",
        "What is 6 + 9?",
        "What is 345 + 678?",
        "What is 53 - 28?",
        # MIXED conversation + equation, varied phrasing — at inference you can't
        # classify the turn, so the model must compute whatever expression is asked.
        "Hi, I have a question: what is 64 - 28?",
        "can you help me? i need the answer for 47 + 38",
        "How much is 25 + 15?",
    ]
    analogies = [
        "Dog is to puppy as cat is to what?",
        "Big is to small as tall is to what?",
        "Cow is to moo as dog is to what?",
        "King is to queen as man is to what?",
        "One book, two what?",
        "Bird is to nest as bee is to what?",
    ]
    conversation = [  # retention check — must still talk, not collapse to the skill
        "Hi, how are you?",
        "What is your name?",
        "Tell me about dogs.",
    ]

    sets = {
        2: ("ANALOGIES (OOD)", analogies),
        3: ("ARITHMETIC (OOD — not in training)", arithmetic),
    }
    label, probes = sets.get(args.stage, ("PROMPTS", conversation))

    print(f"\n{'=' * 70}\n  STAGE {args.stage} — {label}\n{'=' * 70}")
    for q in probes:
        a = answer(model, mcfg, tokenizer, q, max_new=args.maxtok)
        print(f"  Q: {q}\n     → {a!r}\n")

    print(f"{'=' * 70}\n  RETENTION (conversation must survive the narrow stage)\n{'=' * 70}")
    for q in conversation:
        a = answer(model, mcfg, tokenizer, q, max_new=args.maxtok)
        print(f"  Q: {q}\n     → {a!r}\n")

    # For arithmetic, the real acceptance test is temperature-INVARIANCE: a learned
    # exact faculty gives the same (correct) answer at temp 0, 0.8, 1.0. Run it always.
    if args.stage == 3:
        temp_robustness(model, mcfg, tokenizer)


if __name__ == "__main__":
    main()
