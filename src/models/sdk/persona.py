"""System personas + mood annotation, shared by the conversational sources.

These shape REGISTER, not facts. A fraction of the conversational/instruction data
is given a `System:` persona so the model learns to CONDITION on a system prompt;
emotional dialogues carry a `(mood: …)` tag on that SAME channel so tone is driven
by an explicit, neutral-by-default mood. All plain ASCII — no new tokenizer symbols.
"""

from __future__ import annotations

import hashlib

from src.core.modalities.moods import mood_system_phrase

SYSTEM_PERSONAS: list[str] = [
    "You are a helpful, friendly assistant. Answer simply and directly.",
    "You are a kind assistant who talks to young children. Keep words simple.",
    "You are a cheerful helper. Be warm and encouraging.",
    "You are a calm, patient assistant. Explain things gently.",
    "You are a concise assistant. Give short, clear answers.",
    "You are a curious, playful assistant who loves to chat.",
    "You are a storyteller who tells short, simple stories.",
    "You are a thoughtful assistant. Be honest and clear.",
]

STORY_PROMPTS: list[str] = [
    "Tell me a story.",
    "Can you tell me a short story?",
    "Tell me a little story please.",
    "I want to hear a story.",
    "Tell me a bedtime story.",
]


def hash01(key: str) -> float:
    """Deterministic value in [0,1) keyed on text — stable selection across runs."""
    return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) / 0x100000000


def persona_for(key: str) -> str:
    return SYSTEM_PERSONAS[int(hash01(key) * len(SYSTEM_PERSONAS)) % len(SYSTEM_PERSONAS)]


def prepend_system(text: str, persona: str, mood: str = "neutral") -> str:
    """Add a `System:` line (with an optional non-neutral `(mood: …)` tag) above a
    User:/Assistant: transcript so the model learns to condition on it."""
    tag = mood_system_phrase(mood)
    system_line = f"System: {persona}" + (f" {tag}" if tag else "")
    return f"{system_line}\n{text}"
