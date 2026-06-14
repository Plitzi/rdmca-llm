"""Stage 2 data sources — perception and pattern recognition.

Compositional analogies BY RELATION (not fixed tuples) + numeric sequence patterns,
so the model learns the relation/pattern and generalizes, not memorizes instances.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

# COMPOSITIONAL by RELATION: each relation holds MANY instance pairs; an analogy is
# built by sampling TWO different instances of the SAME relation ("a:b :: c:d"), so
# the model learns the RELATION (baby-animal, antonym, …), not specific tuples.
# TODO: fold in a real graded analogy corpus (BATS/Google) when one loads on datasets>=3.
_ANALOGY_RELATIONS = {
    "baby_animal": [
        ("dog", "puppy"),
        ("cat", "kitten"),
        ("cow", "calf"),
        ("horse", "foal"),
        ("sheep", "lamb"),
        ("bear", "cub"),
        ("frog", "tadpole"),
        ("hen", "chick"),
        ("kangaroo", "joey"),
        ("goat", "kid"),
        ("lion", "cub"),
        ("deer", "fawn"),
    ],
    "antonym": [
        ("hot", "cold"),
        ("big", "small"),
        ("up", "down"),
        ("fast", "slow"),
        ("happy", "sad"),
        ("day", "night"),
        ("open", "closed"),
        ("light", "dark"),
        ("high", "low"),
        ("full", "empty"),
        ("wet", "dry"),
        ("hard", "soft"),
        ("near", "far"),
        ("old", "new"),
        ("loud", "quiet"),
        ("rich", "poor"),
    ],
    "gender": [
        ("king", "queen"),
        ("man", "woman"),
        ("boy", "girl"),
        ("prince", "princess"),
        ("actor", "actress"),
        ("uncle", "aunt"),
        ("father", "mother"),
        ("son", "daughter"),
        ("brother", "sister"),
        ("husband", "wife"),
        ("nephew", "niece"),
        ("rooster", "hen"),
    ],
    "animal_sound": [
        ("dog", "bark"),
        ("cat", "meow"),
        ("cow", "moo"),
        ("duck", "quack"),
        ("lion", "roar"),
        ("bird", "chirp"),
        ("horse", "neigh"),
        ("sheep", "baa"),
        ("frog", "croak"),
        ("bee", "buzz"),
        ("snake", "hiss"),
        ("wolf", "howl"),
    ],
    "animal_home": [
        ("dog", "kennel"),
        ("bird", "nest"),
        ("bee", "hive"),
        ("fish", "water"),
        ("cow", "barn"),
        ("horse", "stable"),
        ("spider", "web"),
        ("lion", "den"),
        ("rabbit", "burrow"),
        ("ant", "anthill"),
        ("pig", "pen"),
        ("bat", "cave"),
    ],
    "plural": [
        ("child", "children"),
        ("foot", "feet"),
        ("mouse", "mice"),
        ("man", "men"),
        ("tooth", "teeth"),
        ("person", "people"),
        ("goose", "geese"),
        ("woman", "women"),
        ("leaf", "leaves"),
        ("knife", "knives"),
    ],
    "worker_tool": [
        ("painter", "brush"),
        ("writer", "pen"),
        ("farmer", "plow"),
        ("chef", "knife"),
        ("teacher", "chalk"),
        ("carpenter", "hammer"),
        ("gardener", "spade"),
        ("doctor", "stethoscope"),
        ("tailor", "needle"),
        ("barber", "scissors"),
    ],
    "whole_part": [
        ("car", "wheel"),
        ("tree", "leaf"),
        ("book", "page"),
        ("hand", "finger"),
        ("house", "door"),
        ("clock", "hand"),
        ("bike", "pedal"),
        ("body", "arm"),
        ("flower", "petal"),
        ("shirt", "sleeve"),
    ],
}


def gen_analogies(n: int, seed: int = 1) -> Iterator[dict]:
    """Compositional analogies + numeric patterns for stage 2 (pattern recognition).
    An analogy samples TWO instances of one relation so the model learns the relation,
    not fixed tuples. Three forms: a STATEMENT, a COMPLETION Q&A ('a is to b as c is to
    ___?' → d, the analogy skill), and a numeric pattern."""
    rng = random.Random(seed)
    relations = list(_ANALOGY_RELATIONS.values())
    for _ in range(n):
        roll = rng.random()
        if roll < 0.75:  # word analogy (relation-based)
            pairs = rng.choice(relations)
            (a, b), (c, d) = rng.sample(pairs, 2)
            if rng.random() < 0.5:  # completion Q&A (answer-masked skill)
                yield {
                    "text": f"User: {a} is to {b} as {c} is to what?\nAssistant: {d}.",
                    "lang": "en",
                }
            else:  # statement
                yield {"text": f"{a} is to {b} as {c} is to {d}.", "lang": "en"}
        else:  # numeric pattern (sequence completion)
            start = rng.randint(1, 9)
            step = rng.randint(1, 5)
            seq = [start + step * i for i in range(4)]
            yield {
                "text": f"Pattern: {seq[0]} {seq[1]} {seq[2]} {seq[3]} -> {seq[3] + step}",
                "lang": "en",
            }


def _build_analogies(*, approx_examples, **_):
    return gen_analogies(approx_examples)


SOURCES = {"analogies": _build_analogies}
