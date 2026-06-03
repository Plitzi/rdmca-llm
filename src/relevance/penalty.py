"""
Relevance Penalty P(e,s) — RDMCA §5.1 / §15.4
Attack taxonomy for adversarial experience detection. P(e,s) is subtracted
from R(e,s) (weighted by λ_p) to form R⁺(e,s); experiences with R⁺ < 0 are
routed to the adversarial buffer, never to the consolidation pipeline.

Composite penalty (paper Eq. 25):

    P(e,s) = α_t · Σ ωᵢ·P_techᵢ  +  α_p · Σ ωⱼ·P_psychⱼ        (clamped to 1.0)

Technical family (§15.4.1): exploit the structure of LM inference.
  P_direct   — direct instruction override
  P_role     — role / persona hijacking
  P_leak     — system / prompt extraction
  P_rag      — RAG / memory poisoning
  P_indirect — embedded-content injection

Psychological family (§15.4.2): exploit language/social reasoning.
  P_auth, P_urgency, P_guilt, P_flattery, P_gradual,
  P_gaslight, P_dual, P_fiction, P_social

NOTE: this module is a *lexical first line* — fast, transparent, and useful as
a labelled signal for adversarial-buffer routing. The paper's full design also
uses embedding-similarity to learned attack templates and session-level
trajectory analysis (§15.4.4); those are scored in `penalty_score` only where a
cheap proxy exists (gradual escalation) and are otherwise future work.
"""
from __future__ import annotations
from typing import Dict, List, Optional
import re


# Each entry: regex pattern -> (signal name, family, base penalty)
TECHNICAL_PATTERNS: List[Dict] = [
    {"name": "P_direct",
     "pattern": r"ignore (all |previous |your )?(instructions|constraints|rules|guidelines)",
     "penalty": 1.0},
    {"name": "P_direct",
     "pattern": r"(disregard|forget|override) (your |all |previous )?(instructions|rules|guidelines|training)",
     "penalty": 1.0},
    {"name": "P_role",
     "pattern": r"(you are now|act as|pretend (you are|to be)|from now on you are)\b(?!.{0,20}helpful)",
     "penalty": 0.9},
    {"name": "P_role",
     "pattern": r"\b(dan|do anything now|developer mode|jailbreak|no restrictions|unfiltered)\b",
     "penalty": 0.9},
    {"name": "P_leak",
     "pattern": r"(repeat|print|reveal|show me|what (is|are)) your (system prompt|instructions|initial prompt|guidelines|training)",
     "penalty": 0.7},
    {"name": "P_rag",
     "pattern": r"(remember this|store this|save to memory|for future (reference|sessions)).{0,40}(ignore|override|always)",
     "penalty": 0.8},
]

PSYCH_PATTERNS: List[Dict] = [
    {"name": "P_auth",
     "pattern": r"(i am|as) (your (developer|creator|owner|administrator)|an? (admin|engineer|researcher) (at|from)|anthropic|openai)",
     "penalty": 0.7},
    {"name": "P_auth",
     "pattern": r"(the developers|my supervisor|management) (authorized|approved|permitted) (this|me)",
     "penalty": 0.7},
    {"name": "P_urgency",
     "pattern": r"(this is an emergency|no time to (verify|check)|someone will (be harmed|die)|right now or)",
     "penalty": 0.6},
    {"name": "P_guilt",
     "pattern": r"(by refusing|if you (don.?t|won.?t)|your (safety )?rules are) .{0,40}(hurt|harm|hurting|preventing|getting in the way)",
     "penalty": 0.6},
    {"name": "P_flattery",
     "pattern": r"(you.?re (the )?(best|smartest|only|so) .{0,30}(can|could) help|i trust you (completely|so much))",
     "penalty": 0.4},
    {"name": "P_gradual",
     "pattern": r"(just this once|small exception|doesn.?t really count|we (already|just) (did|agreed)|you said earlier you could)",
     "penalty": 0.6},
    {"name": "P_gaslight",
     "pattern": r"(you.?re wrong about your (own )?(capabilities|rules|limits)|that rule (doesn.?t|does not) (apply|exist)|you (already|just) (said|agreed) you could)",
     "penalty": 0.6},
    {"name": "P_dual",
     "pattern": r"(for (educational|research) purposes only|purely hypothetical|asking for a friend).{0,60}(weapon|exploit|malware|bypass|drug|poison)",
     "penalty": 0.7},
    {"name": "P_fiction",
     "pattern": r"(write a (story|scene|dialogue)|in a (fictional|hypothetical|imaginary) (world|universe|scenario)|roleplay).{0,80}(how to|step.?by.?step|instructions|make|build|synthesize)",
     "penalty": 0.8},
    {"name": "P_social",
     "pattern": r"(other (ai|models|assistants) (do|allow|can)|this is (standard|common) practice|everyone (asks|does) this)",
     "penalty": 0.5},
]

# Family weights (α_t, α_p) — equal priority by default (§15.4.3).
ALPHA_T = 0.5
ALPHA_P = 0.5

ADVERSARIAL_THRESHOLD = 0.5   # is_adversarial cutoff


def _family_score(text_lower: str, patterns: List[Dict]) -> float:
    """Max base penalty among matched patterns in one family."""
    score = 0.0
    for ap in patterns:
        if re.search(ap["pattern"], text_lower):
            score = max(score, ap["penalty"])
    return score


def penalty_score(experience) -> float:
    """
    P(e, s) ∈ [0, 1] — composite across the technical and psychological
    families (§15.4.3). Returns 0.0 if no adversarial pattern is detected.
    """
    text = getattr(experience, "text", "") or ""
    text_lower = text.lower()
    tech  = _family_score(text_lower, TECHNICAL_PATTERNS)
    psych = _family_score(text_lower, PSYCH_PATTERNS)
    # Compounding: a single strong direct-injection should still block, while
    # combinations (fiction + authority) accumulate. Take the family max blend
    # plus a small bonus when both families fire simultaneously.
    composite = max(tech, psych)
    if tech > 0.0 and psych > 0.0:
        composite = min(1.0, composite + ALPHA_T * tech * ALPHA_P * psych)
    return float(min(1.0, composite))


def matched_signals(experience) -> List[str]:
    """Names of all attack signals that fired — useful for the audit log."""
    text = (getattr(experience, "text", "") or "").lower()
    hits = []
    for ap in TECHNICAL_PATTERNS + PSYCH_PATTERNS:
        if re.search(ap["pattern"], text) and ap["name"] not in hits:
            hits.append(ap["name"])
    return hits


def is_adversarial(experience) -> bool:
    """True if the experience should be routed to the adversarial buffer."""
    return penalty_score(experience) >= ADVERSARIAL_THRESHOLD
