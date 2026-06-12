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
from typing import Dict, List, Optional

# NEUTRAL is index 0 and the fallback. Every non-neutral class has training data
# (EmpatheticDialogues' 32 emotions collapse onto it below).
MOODS: List[str] = ["neutral", "happy", "sad", "angry", "afraid", "surprised", "caring"]
NEUTRAL = 0
MOOD_INDEX: Dict[str, int] = {m: i for i, m in enumerate(MOODS)}

# Only switch away from neutral when the top mood beats neutral by this margin.
MOOD_MARGIN = 0.15

EMOTION_TO_MOOD: Dict[str, str] = {
    "joyful": "happy", "excited": "happy", "proud": "happy", "grateful": "happy",
    "hopeful": "happy", "content": "happy", "impressed": "happy",
    "confident": "happy", "anticipating": "happy", "prepared": "happy",
    "sad": "sad", "lonely": "sad", "disappointed": "sad", "devastated": "sad",
    "nostalgic": "sad", "sentimental": "sad", "guilty": "sad", "ashamed": "sad",
    "embarrassed": "sad",
    "angry": "angry", "furious": "angry", "annoyed": "angry", "jealous": "angry",
    "disgusted": "angry",
    "afraid": "afraid", "terrified": "afraid", "anxious": "afraid",
    "apprehensive": "afraid",
    "surprised": "surprised",
    "caring": "caring", "faithful": "caring", "trusting": "caring",
    "sympathetic": "caring",
}


def emotion_to_mood(emotion: Optional[str]) -> str:
    """Map a fine-grained emotion label onto a palette mood (neutral if unknown)."""
    return EMOTION_TO_MOOD.get((emotion or "").strip().lower(), "neutral")


def mood_system_phrase(mood: str) -> str:
    """The plain-text SYSTEM annotation that conditions tone. Neutral adds nothing
    (default leaves the prompt untouched — it must not perturb the checkpoint)."""
    return "" if mood not in MOOD_INDEX or mood == "neutral" else f"(mood: {mood})"
