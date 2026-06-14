"""Stage 4 data sources — causal and procedural reasoning.

Real cause→effect statements (e-CARE, EN) blended with synthetic preschool-level
cause→effect statements and 'what happened next?' Q&A.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

from src.core.data.blend import blend


# ── real causal (e-CARE) ────────────────────────────────────────────────────────
def _causal_statement(cause: str, effect: str) -> str:
    cause = cause.strip().rstrip(".")
    effect = effect.strip()
    if effect:
        effect = effect[0].lower() + effect[1:]
    return f"{cause}, so {effect}"


def stream_causal(langs: list[str], limit_mb: int | None = None) -> Iterator[dict]:
    """Stream real cause→effect statements (EN) reconstructed from e-CARE."""
    if "en" not in {lang.lower() for lang in langs}:
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
        if ex.get("question") == "cause":  # premise is the effect
            cause, effect = correct, premise
        else:  # premise is the cause
            cause, effect = premise, correct
        yield {"text": _causal_statement(cause, effect), "lang": "en"}


# ── synthetic causal fill ────────────────────────────────────────────────────────
_CAUSE_EFFECT = [
    ("it rained", "the ground got wet"),
    ("she dropped the glass", "it broke"),
    ("he forgot his umbrella", "he got wet"),
    ("the sun came out", "the snow melted"),
    ("they ran very fast", "they got tired"),
    ("she watered the plant", "it grew"),
    ("he ate too much candy", "his stomach hurt"),
    ("the fire was hot", "the ice melted"),
    ("nobody fed the cat", "it got hungry"),
    ("the wind blew hard", "the leaves fell"),
    ("she studied a lot", "she passed the test"),
    ("he touched the hot stove", "he got burned"),
    ("it got dark", "they turned on the light"),
    ("the baby was tired", "it fell asleep"),
    ("they planted seeds", "flowers grew"),
    ("he did not sleep", "he felt sleepy"),
    ("the cup had a hole", "the water leaked out"),
    ("she told a funny joke", "everyone laughed"),
    ("the ice cream sat in the sun", "it melted"),
    ("he opened the window", "the room got cold"),
    ("the dog heard a noise", "it barked"),
    ("she pushed the swing", "it went up"),
    ("they shared the cake", "everyone was happy"),
    ("he tripped on a rock", "he fell down"),
    ("the balloon got too big", "it popped"),
    ("she turned the key", "the door opened"),
]


def gen_causal(n: int, seed: int = 1) -> Iterator[dict]:
    """Synthetic simple cause→effect (statements + 'what happened next?' Q&A) at a
    preschool reading level — augments the finite e-CARE corpus for stage 4."""
    rng = random.Random(seed)

    def cap(text: str) -> str:
        return text[0].upper() + text[1:]

    for _ in range(n):
        cause, effect = rng.choice(_CAUSE_EFFECT)
        roll = rng.random()
        if roll < 0.4:
            yield {"text": f"{cap(cause)}, so {effect}.", "lang": "en"}
        elif roll < 0.7:
            yield {"text": f"Because {cause}, {effect}.", "lang": "en"}
        else:  # cause → effect as a Q&A
            yield {
                "text": f"User: {cap(cause)}. What happened next?\nAssistant: {cap(effect)}.",
                "lang": "en",
            }


def _build_causal(*, langs, approx_examples, limit_mb=None, **_):
    return blend(stream_causal(langs, limit_mb), gen_causal(approx_examples), approx_examples)


SOURCES = {"causal": _build_causal, "causal_synth": _build_causal}
