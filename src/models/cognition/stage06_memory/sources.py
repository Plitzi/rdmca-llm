"""Stage 6 data sources — memory management.

Synthetic recall-and-use examples: each leads with a `<mem>…</mem>` block (the SAME
framing src/agent.py injects at inference) holding the relevant fact among distractors,
then a User question and an Assistant answer that USES the fact. ~20% are negatives
where the answer is NOT in memory, so the model learns to recall, not hallucinate.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

_MEM_NAMES = [
    "Maria",
    "Tom",
    "Aisha",
    "Kenji",
    "Lucia",
    "Omar",
    "Sven",
    "Priya",
    "Diego",
    "Lena",
    "Nora",
    "Hugo",
]
_MEM_FACTS = [
    ("favorite color", ["blue", "green", "red", "purple", "orange", "teal", "yellow"]),
    ("pet", ["a cat", "a dog", "a parrot", "a rabbit", "a turtle", "a hamster"]),
    ("home city", ["Lima", "Cairo", "Oslo", "Kyoto", "Madrid", "Accra", "Quito"]),
    ("job", ["a teacher", "a nurse", "a baker", "an engineer", "a pilot", "a chef"]),
    ("favorite food", ["pasta", "mango", "sushi", "tacos", "lentils", "ramen"]),
    ("birthday month", ["March", "July", "October", "January", "May", "September"]),
]


def _mem_fact_line(name: str, attr: str, val: str) -> str:
    return f"{name}'s {attr} is {val}."


def gen_memory(n: int, seed: int = 1) -> Iterator[dict]:
    """Synthetic recall-and-use examples: a <mem> block of facts + distractors, a
    question, and an answer that uses (or correctly disclaims) the memory."""
    rng = random.Random(seed)
    for _ in range(n):
        k = rng.randint(2, 4)  # facts in the <mem> block
        names = rng.sample(_MEM_NAMES, k)
        attrs = [rng.choice(_MEM_FACTS) for _ in range(k)]
        facts = [(names[i], attrs[i][0], rng.choice(attrs[i][1])) for i in range(k)]
        lines = [f"- {_mem_fact_line(*f)}" for f in facts]
        rng.shuffle(lines)
        block = "<mem>\n" + "\n".join(lines) + "\n</mem>"
        if rng.random() < 0.8:  # positive: answer lives in memory
            target = rng.choice(facts)
            question, answer = f"What is {target[0]}'s {target[1]}?", _mem_fact_line(*target)
        else:  # negative: not in memory
            outsider = rng.choice([name for name in _MEM_NAMES if name not in names])
            attr = rng.choice(_MEM_FACTS)[0]
            question, answer = f"What is {outsider}'s {attr}?", "I don't have that in my memory."
        yield {"text": f"{block}\nUser: {question}\nAssistant: {answer}", "lang": "en"}


def _build_memory(*, approx_examples, **_):
    return gen_memory(approx_examples)


SOURCES = {"memory": _build_memory, "memory_synth": _build_memory}
