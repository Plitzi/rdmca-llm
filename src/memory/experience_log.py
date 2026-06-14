"""
Experience queue — the bridge between live interaction and consolidation.

Active interaction (uses/chat/run_chat.py) appends signal-bearing turns here; the
consolidation daemon drains the queue, scores/filters/consolidates it, then clears
it. This is the "experience and memory as the true training signal" loop of RDMCA
§6.5.2 — the model evolves only from what it actually experienced.

NOT every turn is an experience. Like a human, the model should learn from
**errors that were corrected** (highest value — a prediction error) and from
**successes that were confirmed** (reinforcement), but NOT from routine chatter with
no feedback (that would just memorize a transcript with no real benefit). So a turn
is logged ONLY when it carries a learning signal:

  - feedback="corrected": the user fixed the model's answer. The learning TARGET is
    the corrected text (the model's wrong answer is kept for contrast). Top value.
  - feedback="accepted":  the user confirmed the answer was good. Reinforce it.
  - feedback="neutral":   no signal → NOT logged (dropped here).

The Relevance Engine later turns `feedback` into the ground-truth Utility term, and
still filters routine/duplicate "accepted" turns by novelty (see relevance/engine).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

EXPERIENCE_LOG = "data/runtime/experiences.jsonl"

FEEDBACK_KINDS = ("corrected", "accepted", "neutral")

# Conservative cues that the user's message is CORRECTING the previous answer
# (EN + ES, since levels can enable Spanish). Kept conservative — false positives
# turn an ordinary follow-up into a (wrong) lesson, so we only fire on explicit
# disagreement/repair phrasing, not on any negative word.
_CORRECTION_CUES = re.compile(
    r"^\s*(no[,.\s]|nope\b|wrong\b|that'?s\s+(?:not|wrong|incorrect)\b|incorrect\b|"
    r"actually\b|i\s+meant\b|you'?re\s+wrong\b|that'?s\s+not\s+right\b|"
    r"eso\s+(?:no|está\s+mal|es\s+incorrecto)\b|no[,.\s]+en\s+realidad\b|"
    r"te\s+equivocas\b|me\s+refería\b|está\s+mal\b)",
    re.IGNORECASE,
)


def detect_correction(user_message: str) -> bool:
    """True if `user_message` looks like the user correcting the prior answer.
    Used for IMPLICIT feedback: a human notices they were misunderstood from the
    flow of the conversation, without any special button."""
    return bool(user_message and _CORRECTION_CUES.match(user_message))


def _build_target(prompt: str, response: str, feedback: str, correction: str | None) -> str:
    """The text the model should LEARN from this experience, as a User:/Assistant:
    transcript (same convention as the dialogue/agentic training data):
      - corrected → learn the CORRECTED answer (not the wrong one the model gave),
      - accepted  → reinforce the answer the model actually gave."""
    answer = correction if (feedback == "corrected" and correction) else response
    return f"User: {prompt}\nAssistant: {answer}"


def log_experience(
    prompt: str,
    response: str = "",
    feedback: str = "neutral",
    correction: str | None = None,
    lang: str = "en",
    modality: str = "text",
    path: str = EXPERIENCE_LOG,
) -> bool:
    """Append ONE signal-bearing turn as a consolidation experience. Returns True if
    written, False if dropped (no learning signal / empty). `feedback` ∈ FEEDBACK_KINDS.

    Backward-compatible record: keeps a `text` field (the learning target) that the
    daemon/pipeline already read, plus the new prompt/response/feedback/correction."""
    if feedback == "neutral" or not prompt or not prompt.strip():
        return False  # routine turn with no feedback → don't save
    if feedback not in FEEDBACK_KINDS:
        feedback = "neutral"
        return False
    text = _build_target(prompt, response, feedback, correction)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "text": text,
                    "prompt": prompt,
                    "response": response,
                    "feedback": feedback,
                    "correction": correction,
                    "lang": lang,
                    "modality": modality,
                    "timestamp": time.time(),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    return True


def load_experiences(path: str = EXPERIENCE_LOG) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def clear_experiences(path: str = EXPERIENCE_LOG) -> None:
    """Empty the queue after the daemon drained it. TRUNCATE (not unlink): the chat
    appends to this same file, and unlinking forks the inode — a write between the
    daemon's read and the clear would land in an orphaned inode and be lost. Truncating
    keeps one inode so concurrent appends always target the live file."""
    p = Path(path)
    if p.exists():
        p.write_text("")
