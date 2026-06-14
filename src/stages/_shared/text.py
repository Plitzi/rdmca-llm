"""Readability gate + a reproducible content hash, shared by several stages."""

from __future__ import annotations

import hashlib
import re


def stable_hash(text: str) -> str:
    """Deterministic content hash for dedup. Python's built-in `hash()` is salted
    per process (PYTHONHASHSEED), so the SAME corpus deduped in two runs of
    prepare_data could keep DIFFERENT examples — making gates non-comparable across
    runs. A content hash makes the prepared data reproducible."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


_VOWELS = "aeiouy"


def syllable_count(word: str) -> int:
    """Approximate syllable count: number of vowel groups (min 1)."""
    word = word.lower()
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)
    if word.endswith("e") and count > 1:  # silent final 'e'
        count -= 1
    return max(count, 1)


def flesch_kincaid_grade(text: str) -> float:
    """Flesch-Kincaid US grade level. Higher = harder to read.
    grade = 0.39·(words/sentence) + 11.8·(syllables/word) − 15.59."""
    words = re.findall(r"[A-Za-zÀ-ÿ']+", text)
    if not words:
        return 0.0
    sentences = max(len(re.findall(r"[.!?]+", text)), 1)
    syllables = sum(syllable_count(w) for w in words)
    words_per_sentence = len(words) / sentences
    syllables_per_word = syllables / len(words)
    return 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59


def passes_filter(text: str, spec: dict | None) -> bool:
    """True if `text` is simple enough for the filter spec. `spec` is None at
    level 5 (everything passes). Keys: `max_grade`, `max_word_len`."""
    if not spec:
        return True
    if "max_word_len" in spec and any(len(w) > spec["max_word_len"] for w in text.split()):
        return False
    return not ("max_grade" in spec and flesch_kincaid_grade(text) > spec["max_grade"])


# Backwards-compatible private aliases (old graded.py names, still imported by tests).
_stable_hash = stable_hash
_syllables = syllable_count
