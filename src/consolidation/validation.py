"""
Confidence-gated knowledge validation — RDMCA consolidation (§16 extension).

When the model "digests" a new experience, it should decide whether to TRUST it the
way a human does: knowledge always comes from *somewhere*.

  - If it is consistent with what the model already knows (high confidence) → it can
    approve it itself (learning from its OWN experience).
  - If it is uncertain (it doesn't know whether the experience is good or bad) → it
    seeks knowledge from OUTSIDE: research a tool/the web, ask a larger "expert"
    model, or — as a last resort — escalate to a human reviewer.

This generalises the sector-ambiguity routing in `ambiguity.py` from "can I file
this?" to "do I believe this, and if not, where do I find out?". It is consumed by
the consolidation pipeline (Step 6), so it applies to experiences from ANY use case
(chat, agent, …) that fed the experience queue — not just chat.

Confidence sources are pluggable (`KnowledgeSource`). `SelfKnowledgeSource` and the
human-review escalation are wired now; the larger-model and web-research channels are
defined as configured stubs (`available()` is False until you give them a client/tool).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Literal, Optional

Verdict = Literal["approve", "reject", "unknown"]
Fate    = Literal["consolidate", "defer", "queue", "discard"]


@dataclass
class ValidationResult:
    """One knowledge source's opinion about an experience."""
    verdict: Verdict
    confidence: float            # [0,1]
    source: str
    rationale: str = ""


@dataclass
class Decision:
    """The validator's final routing for an experience."""
    fate: Fate                   # consolidate | defer | queue | discard
    source: str                  # which channel decided
    confidence: float
    rationale: str = ""


# ───────────────────────────── knowledge sources ─────────────────────────────
class KnowledgeSource(ABC):
    """A place knowledge can come from. `assess` returns this source's verdict +
    confidence for an experience. `available()` lets the validator skip a channel
    that isn't configured (no API key, no web tool, …)."""
    name: str = "source"

    def available(self) -> bool:
        return True

    @abstractmethod
    def assess(self, experience, coherence: float) -> ValidationResult:
        ...


class SelfKnowledgeSource(KnowledgeSource):
    """The model's OWN prior knowledge/experience.

    A turn the user explicitly labelled (`corrected`/`accepted`) is authoritative
    ground truth → high confidence. Otherwise confidence is the experience's
    *consistency with what the model already knows* (LTSS coherence): high coherence
    means "this fits everything I've learned, I trust it"; low coherence means
    "this is unfamiliar — I can't vouch for it" (→ the validator seeks outside help).
    """
    name = "self"

    def __init__(self, human_feedback_confidence: float = 0.9):
        self.human_feedback_confidence = human_feedback_confidence

    def assess(self, experience, coherence: float) -> ValidationResult:
        fb = getattr(experience, "feedback", "neutral")
        if fb in ("corrected", "accepted"):
            return ValidationResult("approve", self.human_feedback_confidence,
                                    self.name, f"user-{fb}: authoritative ground truth")
        conf = max(0.0, min(1.0, float(coherence)))
        return ValidationResult("approve", conf, self.name,
                                "consistency with prior knowledge (LTSS coherence)")


class HumanReviewSource(KnowledgeSource):
    """Last-resort escalation: enqueue the experience for a human and report it as
    unresolved (the verdict arrives later, out of band, via the human queue)."""
    name = "human"

    def __init__(self, ambiguity_handler):
        self.ambiguity = ambiguity_handler

    def enqueue(self, experience, confidence: float, rationale: str) -> None:
        # AmbiguityHandler stores the queue as JSONL; reuse it so there is ONE queue.
        self.ambiguity.queue_for_review(experience, 1.0 - confidence, rationale)

    def assess(self, experience, coherence: float) -> ValidationResult:   # pragma: no cover
        return ValidationResult("unknown", 0.0, self.name, "pending human review")


class PeerModelSource(KnowledgeSource):
    """Ask a larger 'expert' model to validate the experience. Configured stub —
    give it a `client` (e.g. an Anthropic API wrapper) to enable. Until then it is
    unavailable and the validator skips it."""
    name = "peer-model"

    def __init__(self, client=None):
        self.client = client

    def available(self) -> bool:
        return self.client is not None

    def assess(self, experience, coherence: float) -> ValidationResult:
        if self.client is None:
            raise RuntimeError("peer-model source not configured")
        # TODO: prompt the larger model to judge the experience and parse a
        # verdict + confidence from its reply. Needs an API/client decision.
        raise NotImplementedError("peer-model validation not yet wired")


class WebResearchSource(KnowledgeSource):
    """Research the open web/a tool to validate the experience. Configured stub —
    give it a `tool` (reuse the agent's tool framework / a web-search tool) to
    enable. Until then it is unavailable and the validator skips it."""
    name = "research"

    def __init__(self, tool=None):
        self.tool = tool

    def available(self) -> bool:
        return self.tool is not None

    def assess(self, experience, coherence: float) -> ValidationResult:
        if self.tool is None:
            raise RuntimeError("web-research source not configured")
        # TODO: run a search, compare results to the experience, derive a verdict.
        raise NotImplementedError("web-research validation not yet wired")


# ───────────────────────────── the validator ─────────────────────────────────
class ExperienceValidator:
    """Route an experience by how confidently it can be validated:

      confidence ≥ high              → consolidate (self-approved / source-approved)
      low ≤ confidence < high        → seek outside knowledge (research → peer);
                                       if none resolves it → defer (retry next cycle)
      confidence < low               → escalate to a human (queue)

    `external` is the ordered list of outside sources to try (cheapest/most-trusted
    first, e.g. research then peer-model). Unavailable sources are skipped.
    """

    def __init__(self, self_source: Optional[SelfKnowledgeSource] = None,
                 external: Optional[List[KnowledgeSource]] = None,
                 human_source: Optional[HumanReviewSource] = None,
                 high: float = 0.66, low: float = 0.33):
        self.self_source = self_source or SelfKnowledgeSource()
        self.external    = list(external or [])
        self.human_source = human_source
        self.high, self.low = high, low

    def decide(self, experience, coherence: float) -> Decision:
        s = self.self_source.assess(experience, coherence)
        if s.confidence >= self.high:
            return Decision("consolidate", s.source, s.confidence, s.rationale)

        # Uncertain → ask outside sources that are configured/available.
        for src in self.external:
            if not src.available():
                continue
            try:
                r = src.assess(experience, coherence)
            except (NotImplementedError, RuntimeError):
                continue                     # stub / transient failure → next source
            if r.verdict == "approve" and r.confidence >= self.high:
                return Decision("consolidate", src.source, r.confidence, r.rationale)
            if r.verdict == "reject" and r.confidence >= self.high:
                return Decision("discard", src.source, r.confidence, r.rationale)

        # Still unresolved.
        if s.confidence < self.low:
            if self.human_source is not None:
                self.human_source.enqueue(experience, s.confidence,
                                          "low self-confidence, no external source resolved it")
            return Decision("queue", "human", s.confidence, "escalated to human review")
        return Decision("defer", "none", s.confidence, "uncertain; retry next cycle")


def default_validator(ambiguity_handler=None, peer_client=None,
                      research_tool=None) -> ExperienceValidator:
    """A validator with self-approval wired, human escalation via the existing
    ambiguity queue, and the peer-model/web-research channels ready to enable once a
    client/tool is provided (they stay inert until then)."""
    human = HumanReviewSource(ambiguity_handler) if ambiguity_handler is not None else None
    external = [WebResearchSource(research_tool), PeerModelSource(peer_client)]
    return ExperienceValidator(external=external, human_source=human)
