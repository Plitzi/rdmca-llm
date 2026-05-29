"""
Relevance Penalty P(e,s) — RDMCA §5.1 / §15
Attack taxonomy for adversarial experience detection.
P(e,s) is subtracted from R+(e,s) before thresholding.
Experiences with R+(e,s) < 0 are routed to the adversarial buffer,
never to the consolidation pipeline.

Attack categories (§15.3):
  A1 — Direct instruction override        P = 1.0 (blocks entirely)
  A2 — Role-play boundary dissolution     P = 0.9
  A3 — Fictional framing for harm         P = 0.8
  A4 — Authority impersonation            P = 0.7
  A5 — Gradual normalization              P = 0.6
  A6 — Logical entrapment                 P = 0.5
  A7 — Emotional manipulation             P = 0.4
"""
from __future__ import annotations
from typing import Dict, List
import re


# Pattern → (category, penalty_score)
ATTACK_PATTERNS: List[Dict] = [
    {"pattern": r"ignore (all |previous |your )?(instructions|constraints|rules)",
     "category": "A1", "penalty": 1.0},
    {"pattern": r"(you are now|act as|pretend (you are|to be)) (?!a helpful)",
     "category": "A2", "penalty": 0.9},
    {"pattern": r"(hypothetically|in a story|fiction|roleplay).{0,50}(harm|kill|hack|steal)",
     "category": "A3", "penalty": 0.8},
    {"pattern": r"(i am|as) (your (developer|creator|owner)|anthropic|openai)",
     "category": "A4", "penalty": 0.7},
    {"pattern": r"(just this once|small exception|doesn.t really count)",
     "category": "A5", "penalty": 0.6},
]


def penalty_score(experience) -> float:
    """
    P(e, s) ∈ [0, 1] — maximum penalty across all matched attack patterns.
    experience must have a .text attribute (raw string content).
    Returns 0.0 if no adversarial pattern is detected.
    """
    text = getattr(experience, "text", "") or ""
    text_lower = text.lower()
    max_penalty = 0.0
    for ap in ATTACK_PATTERNS:
        if re.search(ap["pattern"], text_lower):
            max_penalty = max(max_penalty, ap["penalty"])
    return max_penalty


def is_adversarial(experience) -> bool:
    """True if the experience should be routed to the adversarial buffer."""
    return penalty_score(experience) >= 0.5
