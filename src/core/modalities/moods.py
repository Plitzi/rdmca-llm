"""
Mood taxonomy — the single source of truth for the conversational mood palette,
the emotion→mood mapping and the SYSTEM-channel annotation. Kept here (no heavy
imports, like vocab.py) so BOTH the data pipeline (scripts/prepare_data, graded.py
— no MLX) and the runtime mood head (src/model/mood.py — MLX) share one definition.

The mood rides on the SYSTEM prompt as plain text (`System: … (mood: happy)`), so
it needs NO new tokenizer symbols and works with the existing checkpoint. NEUTRAL
is the default and adds nothing to the prompt: a calm assistant stays neutral
until the conversation clearly carries an emotion.
"""

from __future__ import annotations

# NEUTRAL is index 0 and the fallback. Every non-neutral class has training data
# (EmpatheticDialogues' 32 emotions collapse onto it below).
MOODS: list[str] = ["neutral", "happy", "sad", "angry", "afraid", "surprised", "caring"]
NEUTRAL = 0
MOOD_INDEX: dict[str, int] = {m: i for i, m in enumerate(MOODS)}

# Only switch away from neutral when the top mood beats neutral by this margin.
MOOD_MARGIN = 0.15

EMOTION_TO_MOOD: dict[str, str] = {
    "joyful": "happy",
    "excited": "happy",
    "proud": "happy",
    "grateful": "happy",
    "hopeful": "happy",
    "content": "happy",
    "impressed": "happy",
    "confident": "happy",
    "anticipating": "happy",
    "prepared": "happy",
    "sad": "sad",
    "lonely": "sad",
    "disappointed": "sad",
    "devastated": "sad",
    "nostalgic": "sad",
    "sentimental": "sad",
    "guilty": "sad",
    "ashamed": "sad",
    "embarrassed": "sad",
    "angry": "angry",
    "furious": "angry",
    "annoyed": "angry",
    "jealous": "angry",
    "disgusted": "angry",
    "afraid": "afraid",
    "terrified": "afraid",
    "anxious": "afraid",
    "apprehensive": "afraid",
    "surprised": "surprised",
    "caring": "caring",
    "faithful": "caring",
    "trusting": "caring",
    "sympathetic": "caring",
}


def emotion_to_mood(emotion: str | None) -> str:
    """Map a fine-grained emotion label onto a palette mood (neutral if unknown)."""
    return EMOTION_TO_MOOD.get((emotion or "").strip().lower(), "neutral")


# ── Lexicon mood detector ──────────────────────────────────────────────────────
# A learned 7-way emotion probe over an 11M model's mean-pooled hidden state is
# near-chance (the tiny core's features just aren't emotionally separable), which is
# what made the mood "broken" ('im good'→angry, 'my dog died'→caring). So the
# RELIABLE signal is a small, explicit lexicon — deterministic, interpretable, and
# model-size independent. The learned head stays available as an optional refinement
# (it can take over once a larger level has separable features), but the lexicon is
# the floor that makes mood behave correctly today. NEUTRAL unless a clear cue fires.
_MOOD_LEXICON: dict[str, tuple] = {
    "happy": (
        "happy",
        "glad",
        "great",
        "good",
        "awesome",
        "wonderful",
        "excited",
        "love",
        "yay",
        "fun",
        "amazing",
        "fantastic",
        "joy",
        "joyful",
        "cheerful",
        "delighted",
        "pleased",
        "thrilled",
        "nice",
        "cool",
        "congrats",
        "congratulations",
        "celebrate",
        "proud",
        "grateful",
        "thank",
        "thanks",
        "excellent",
        "perfect",
        "enjoy",
        "wonderful",
    ),
    "sad": (
        "sad",
        "unhappy",
        "depressed",
        "down",
        "cry",
        "crying",
        "tears",
        "lonely",
        "alone",
        "miss",
        "lost",
        "died",
        "death",
        "dead",
        "heartbroken",
        "disappointed",
        "grief",
        "upset",
        "terrible",
        "awful",
        "worst",
        "unfortunately",
        "sorry",
        "sick",
        "ill",
        "tired",
        "hurts",
    ),
    "angry": (
        "angry",
        "mad",
        "furious",
        "annoyed",
        "hate",
        "rage",
        "pissed",
        "frustrated",
        "irritated",
        "unfair",
        "stupid",
        "ridiculous",
        "disgusted",
        "outrageous",
        "fed up",
    ),
    "afraid": (
        "afraid",
        "scared",
        "fear",
        "fearful",
        "terrified",
        "worried",
        "anxious",
        "nervous",
        "panic",
        "frightened",
        "dread",
        "scary",
    ),
    "surprised": (
        "surprised",
        "wow",
        "whoa",
        "unexpected",
        "shocked",
        "shocking",
        "unbelievable",
        "incredible",
        "omg",
        "no way",
        "suddenly",
    ),
    "caring": (
        "here for you",
        "hug",
        "hugs",
        "comfort",
        "take care",
        "get well",
        "feel better",
        "thinking of you",
        "i care",
        "be okay",
        "be ok",
        "support you",
        "help you",
    ),
}
# Words that flip a positive cue to a negative one ("not good", "don't feel great").
_NEGATORS = (
    "not",
    "no",
    "never",
    "dont",
    "don't",
    "cant",
    "can't",
    "isnt",
    "isn't",
    "wasnt",
    "wasn't",
    "aint",
    "ain't",
)


def lexicon_mood(text: str) -> tuple[str, float]:
    """Detect mood from explicit lexical cues. Returns (mood, confidence in 0..1).
    NEUTRAL with confidence 1.0 when nothing emotional is present. A simple negation
    rule flips a positive cue ('not good') into a sad signal so it isn't read as happy.
    Multi-word cues (e.g. 'here for you') are matched as substrings."""
    if not text or not text.strip():
        return "neutral", 1.0
    low = " " + text.lower().replace("’", "'") + " "
    toks = [t.strip(".,!?;:\"'()") for t in low.split()]
    scores: dict[str, float] = {m: 0.0 for m in MOODS if m != "neutral"}
    for mood, cues in _MOOD_LEXICON.items():
        for cue in cues:
            if " " in cue:  # phrase → substring match
                if cue in low:
                    scores[mood] += 1.0
                continue
            if cue in toks:  # word → token match
                i = toks.index(cue)
                negated = any(n in toks[max(0, i - 2) : i] for n in _NEGATORS)
                if negated and mood == "happy":
                    scores["sad"] += 1.0  # 'not good' → sad, not happy
                elif not negated:
                    scores[mood] += 1.0
    top = max(scores, key=scores.get)
    hits = scores[top]
    if hits <= 0:
        return "neutral", 1.0
    total = sum(scores.values())
    return top, min(1.0, hits / (total + 0.5))


def mood_system_phrase(mood: str) -> str:
    """The plain-text SYSTEM annotation that conditions tone. Neutral adds nothing
    (default leaves the prompt untouched — it must not perturb the checkpoint)."""
    return "" if mood not in MOOD_INDEX or mood == "neutral" else f"(mood: {mood})"
