"""Stage 1 offline compositional generators — basic chat, dictionary definitions,
and grammar rules.

These need no network: they synthesize clean, low-entropy fluency anchors and
rule-application examples (parts of speech, a/an, plurals, tense, comparatives,
synonyms/antonyms) from curated banks, scaling vocabulary with `level` via the tiered
dictionary. Split out of sources.py, which keeps the real (HF) corpus streamers.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

from src.plugins.cognition.stage01_language.dictionary import DICT_TIERS
from src.plugins.sdk import cycle_records

# ── basic everyday conversation (clean, coherent, low-entropy fluency anchor) ──
_BC_GREET_IN = [
    "Hi",
    "Hello",
    "Hey",
    "Hi there",
    "Hello there",
    "Good morning",
    "Good afternoon",
    "Good evening",
]
_BC_GREET_OUT = [
    "Hi!",
    "Hello!",
    "Hey!",
    "Hi there!",
    "Hello! Nice to meet you.",
    "Hi! How can I help you?",
    "Hello! How are you?",
]
_BC_HOW_IN = ["How are you?", "How are you doing?", "How's it going?", "How do you feel today?"]
_BC_HOW_OUT = [
    "I'm doing well, thank you! How are you?",
    "I'm good, thanks for asking! How about you?",
    "I'm fine, thank you. How are you today?",
    "I'm great, thanks! How can I help?",
]
_BC_FACTS = [
    ("What color is the sky?", "The sky is blue."),
    ("What color is grass?", "Grass is green."),
    ("What color is the sun?", "The sun is yellow."),
    ("What sound does a dog make?", "A dog says woof."),
    ("What sound does a cat make?", "A cat says meow."),
    ("How many legs does a dog have?", "A dog has four legs."),
    ("How many days are in a week?", "There are seven days in a week."),
    ("What do bees make?", "Bees make honey."),
    ("Where do fish live?", "Fish live in water."),
    ("What do we use to see?", "We use our eyes to see."),
    ("What is the opposite of hot?", "The opposite of hot is cold."),
    ("What is the opposite of big?", "The opposite of big is small."),
    ("What do you drink when you are thirsty?", "You drink water when you are thirsty."),
    ("What comes after the number two?", "The number three comes after two."),
]
_BC_IDENTITY = [
    ("What is your name?", "I'm RDMCA, your helpful assistant."),
    ("Who are you?", "I'm RDMCA, a friendly assistant here to help you."),
    ("What can you do?", "I can chat with you, answer simple questions, and help you learn."),
    ("Can you help me?", "Yes, I'd be happy to help. What do you need?"),
    ("Are you a robot?", "I'm a computer assistant. I'm here to help you."),
]
_BC_POLITE = [
    ("Thank you!", "You're welcome!"),
    ("Thanks a lot.", "You're welcome! Happy to help."),
    ("Sorry.", "That's okay, no problem."),
    ("Goodbye!", "Goodbye! Have a nice day."),
    ("Bye!", "Bye! Take care."),
    ("See you later.", "See you later! Take care."),
    ("Please help me.", "Of course, I'm happy to help."),
]


def gen_basic_chat(n: int, seed: int = 1) -> Iterator[dict]:
    """Clean, short, COHERENT everyday exchanges (greet / how-are-you / identity /
    simple facts / politeness) — the high-frequency conversation a base must get
    right to feel fluent. A bounded, de-duplicated set; oversampled by the loader."""
    pairs: list[tuple] = []
    for greeting in _BC_GREET_IN:
        for reply in _BC_GREET_OUT:
            pairs.append((greeting, reply))
    for question in _BC_HOW_IN:
        for answer in _BC_HOW_OUT:
            pairs.append((question, answer))
    pairs += _BC_FACTS + _BC_IDENTITY + _BC_POLITE
    records = [{"text": f"User: {q}\nAssistant: {a}", "lang": "en"} for q, a in pairs]
    yield from cycle_records(records, n, seed)


# ── dictionary / word meanings ─────────────────────────────────────────────────
def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _definition_surfaces(word: str, pos: str, definition: str) -> list[str]:
    """All distinct surface forms for one dictionary entry — a definition statement
    plus the 'what does X mean?' Q&A phrasings the chat/agent actually use."""
    cap = word.capitalize()
    art = _article(word)
    if pos == "n":
        stmt = f"{art.capitalize()} {word} is {definition}."
        return [
            stmt,
            f"User: What is {art} {word}?\nAssistant: {stmt}",
            f"User: What does {word!r} mean?\nAssistant: {stmt}",
        ]
    if pos == "v":
        stmt = f"To {word} means to {definition}."
        return [
            stmt,
            f"User: What does it mean to {word}?\nAssistant: {stmt}",
            f"User: What does {word!r} mean?\nAssistant: {stmt}",
        ]
    stmt = f"{cap} means {definition}."
    return [
        stmt,
        f"User: What does {word!r} mean?\nAssistant: {stmt}",
        f"User: What is the meaning of {word!r}?\nAssistant: {stmt}",
    ]


def gen_definitions(n: int, level: int = 1, seed: int = 1) -> Iterator[dict]:
    """Graded dictionary entries — statements + 'what does X mean?' Q&A — so the
    model learns word MEANINGS, not just word sequences. Includes every tier up to
    `level`, so vocabulary grows per level. Yields each unique (word, surface) record
    ONCE (deterministically shuffled); the loader's oversampling provides repetition."""
    bank: dict[str, tuple] = {}
    for tier in DICT_TIERS[: max(level, 1)]:
        bank.update(tier)
    records = [
        {"text": surface, "lang": "en"}
        for word, (pos, definition) in bank.items()
        for surface in _definition_surfaces(word, pos, definition)
    ]
    yield from cycle_records(records, n, seed)


# ── grammar (compositional) ────────────────────────────────────────────────────
_G_NOUNS_CONS = [
    "car",
    "dog",
    "cat",
    "tree",
    "book",
    "house",
    "friend",
    "school",
    "table",
    "garden",
    "river",
    "ball",
    "door",
    "cup",
]
_G_NOUNS_VOWEL = [
    "apple",
    "egg",
    "orange",
    "umbrella",
    "elephant",
    "island",
    "apron",
    "insect",
    "owl",
    "envelope",
]
_G_VERBS = ["run", "eat", "sleep", "play", "read", "walk", "help", "jump", "give", "learn"]
_G_ADJS = ["big", "small", "happy", "fast", "kind", "brave", "cold", "quiet", "heavy", "red"]
_PLURAL_IRREG = {
    "child": "children",
    "foot": "feet",
    "mouse": "mice",
    "man": "men",
    "tooth": "teeth",
    "person": "people",
    "woman": "women",
    "goose": "geese",
}
_PAST_REG = {
    "walk": "walked",
    "play": "played",
    "help": "helped",
    "jump": "jumped",
    "learn": "learned",
    "open": "opened",
    "clean": "cleaned",
    "call": "called",
    "look": "looked",
    "want": "wanted",
}
_PAST_IRREG = {
    "go": "went",
    "run": "ran",
    "eat": "ate",
    "sleep": "slept",
    "give": "gave",
    "see": "saw",
    "make": "made",
    "come": "came",
    "take": "took",
    "find": "found",
}
_COMPARATIVE = {
    "big": ("bigger", "biggest"),
    "small": ("smaller", "smallest"),
    "fast": ("faster", "fastest"),
    "slow": ("slower", "slowest"),
    "hot": ("hotter", "hottest"),
    "cold": ("colder", "coldest"),
    "kind": ("kinder", "kindest"),
    "happy": ("happier", "happiest"),
    "sad": ("sadder", "saddest"),
    "tall": ("taller", "tallest"),
}
_SYNONYMS = {
    "happy": "glad",
    "big": "large",
    "small": "little",
    "fast": "quick",
    "sad": "unhappy",
    "scared": "afraid",
    "cold": "chilly",
    "kind": "nice",
}
_G_ANIMATE = ["dog", "cat", "friend", "bird", "boy", "girl", "child", "man", "woman", "teacher"]
_G_ANIMATE_REG = ["dog", "cat", "friend", "bird", "boy", "girl", "teacher"]
_ANTONYMS = {
    "happy": "sad",
    "big": "small",
    "hot": "cold",
    "fast": "slow",
    "up": "down",
    "open": "closed",
    "day": "night",
    "full": "empty",
    "new": "old",
    "near": "far",
}
_TRANSITIVE = {
    "eat": ["food", "an apple", "lunch"],
    "read": ["a book", "a story"],
    "help": ["a friend", "her mom"],
    "give": ["a gift", "the ball"],
    "build": ["a house", "a tower"],
    "see": ["a bird", "the moon"],
}


def gen_grammar(n: int, level: int = 1, seed: int = 1) -> Iterator[dict]:
    """Compositional grammar + word-usage for stage 1 (language). Teaches RULES applied
    across many words (parts of speech, a/an, plurals, verb tense, comparatives, adjective
    placement, sentence composition, subject-verb agreement) and word MEANING-IN-USE
    (synonyms/antonyms + example sentences) — as statements and completion Q&A so the
    model learns to APPLY the rule, not memorize sentences. Vocabulary SCALES with `level`
    via the tiered dictionary bank; morphology stays curated (correct forms)."""
    rng = random.Random(seed)
    bank: dict[str, tuple] = {}
    for tier in DICT_TIERS[: max(level, 1)]:
        bank.update(tier)
    pos_nouns = list(
        {*_G_NOUNS_CONS, *_G_NOUNS_VOWEL, *[w for w, (p, _) in bank.items() if p == "n"]}
    )
    pos_verbs = list({*_G_VERBS, *[w for w, (p, _) in bank.items() if p == "v"]})
    pos_adjs = list({*_G_ADJS, *[w for w, (p, _) in bank.items() if p == "a"]})
    for _ in range(n):
        k = rng.randint(0, 9)
        if k == 0:  # parts of speech (vocab scales with level)
            pos, word = rng.choice(
                [
                    ("noun", rng.choice(pos_nouns)),
                    ("verb", rng.choice(pos_verbs)),
                    ("adjective", rng.choice(pos_adjs)),
                ]
            )
            desc = {
                "noun": "names a person, place, or thing",
                "verb": "is an action you can do",
                "adjective": "describes a noun",
            }[pos]
            art = "an" if pos[0] in "aeiou" else "a"  # an adjective / a noun / a verb
            yield {
                "text": f"{art.capitalize()} {pos} {desc}. '{word}' is {art} {pos}.",
                "lang": "en",
            }
        elif k == 1:  # a / an (article + vowel rule)
            if rng.random() < 0.5:
                word = rng.choice(_G_NOUNS_VOWEL)
                art = "an"
            else:
                word = rng.choice(_G_NOUNS_CONS)
                art = "a"
            if rng.random() < 0.5:  # never shows the wrong form, only the rule
                yield {
                    "text": f"User: Which article goes before '{word}', 'a' or 'an'?\n"
                    f"Assistant: {art} — use 'an' before a vowel sound. {art} {word}.",
                    "lang": "en",
                }
            else:
                yield {"text": f"We say '{art} {word}'.", "lang": "en"}
        elif k == 2:  # plurals (regular + irregular)
            if rng.random() < 0.7:
                word = rng.choice(_G_NOUNS_CONS)
                yield {
                    "text": (
                        f"User: What is the plural of '{word}'?\nAssistant: {word}s. Add 's' to make a plural."
                    )
                    if rng.random() < 0.5
                    else f"One {word}, two {word}s.",
                    "lang": "en",
                }
            else:
                singular, plural = rng.choice(list(_PLURAL_IRREG.items()))
                yield {"text": f"Some plurals change: one {singular}, two {plural}.", "lang": "en"}
        elif k == 3:  # verb tense (regular + irregular)
            if rng.random() < 0.6:
                verb, past = rng.choice(list(_PAST_REG.items()))
                rule = " Add 'ed' for the past tense."
            else:
                verb, past = rng.choice(list(_PAST_IRREG.items()))
                rule = " Some verbs change in the past tense."
            yield {
                "text": (f"User: What is the past tense of '{verb}'?\nAssistant: {past}.{rule}")
                if rng.random() < 0.5
                else f"Today I {verb}. Yesterday I {past}.",
                "lang": "en",
            }
        elif k == 4:  # comparatives
            adj, (comp, sup) = rng.choice(list(_COMPARATIVE.items()))
            yield {
                "text": f"{adj}, {comp}, {sup}. Use '{comp}' to compare two things.",
                "lang": "en",
            }
        elif k == 5:  # adjective placement (vocab scales w/ level)
            adj, noun = rng.choice(pos_adjs), rng.choice(_G_NOUNS_CONS)
            yield {
                "text": f"The {adj} {noun}. An adjective comes before the noun it describes.",
                "lang": "en",
            }
        elif k == 6:  # sentence composition (SV, labeled)
            noun, verb = rng.choice(_G_ANIMATE), rng.choice(_G_VERBS)
            yield {
                "text": f"The {noun} {verb}s. A sentence needs a subject and a verb — "
                f"here the subject is '{noun}' and the verb is '{verb}s'.",
                "lang": "en",
            }
        elif k == 7:  # subject-verb agreement (regular plurals)
            noun, verb = rng.choice(_G_ANIMATE_REG), rng.choice(_G_VERBS)
            yield {"text": f"One {noun} {verb}s. Two {noun}s {verb}.", "lang": "en"}
        elif k == 8:  # SVO sentence (transitive only → grammatical)
            verb, objs = rng.choice(list(_TRANSITIVE.items()))
            noun, obj = rng.choice(_G_ANIMATE), rng.choice(objs)
            yield {"text": f"The {noun} {verb}s {obj}.", "lang": "en"}
        else:  # word meaning in use (synonym/antonym)
            if rng.random() < 0.5 and _SYNONYMS:
                word, synonym = rng.choice(list(_SYNONYMS.items()))
                noun = rng.choice(_G_ANIMATE)
                yield {
                    "text": f"'{word}' means about the same as '{synonym}'. A {noun} can be {word} or {synonym}.",
                    "lang": "en",
                }
            else:
                word, antonym = rng.choice(list(_ANTONYMS.items()))
                yield {"text": f"The opposite of '{word}' is '{antonym}'.", "lang": "en"}
