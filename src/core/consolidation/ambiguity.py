"""
Ambiguity Deferral & Human Review Queue — RDMCA §16 / Implementation Guide §1.7
Experiences that cannot be confidently assigned a fate are deferred or
escalated to a human review queue.

Ambiguity score A(e,s) ∈ [0,1]:
  A(e,s) = 1 - max(sector_affinities)  (low confidence in sector assignment)
  Also elevated by: KL > ε, BCF near threshold, cross-sector affinity conflict.

Thresholds:
  A < 0.3         → clear:  consolidate normally
  0.3 ≤ A < 0.7  → defer:  retry next cycle (max 3 retries)
  A ≥ 0.7         → queue:  human review required

Human queue actions: consolidate | discard | adversarial | policy_defer
Experiences unreviewed for 7 days are auto-expired.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Literal

from src.core.memory.episodic_buffer import Experience

AMBIGUITY_DEFER = 0.3
AMBIGUITY_QUEUE = 0.7
MAX_DEFER_CYCLES = 3
QUEUE_EXPIRY_DAYS = 7

HumanAction = Literal["consolidate", "discard", "adversarial", "policy_defer"]


@dataclass
class QueueEntry:
    experience_uid: str
    text_preview: str
    ambiguity_score: float
    added_at: float
    defer_count: int = 0
    reviewed: bool = False
    action: HumanAction | None = None
    rationale: str = ""


class AmbiguityHandler:
    """
    Tracks deferred experiences and the human review queue.
    Persists queue to disk as JSONL.
    """

    def __init__(self, queue_path: str = "logs/human_queue.jsonl"):
        self.queue_path = queue_path
        self._deferred: list[tuple] = []  # (experience, defer_count)
        self._queue: list[QueueEntry] = []
        os.makedirs(os.path.dirname(queue_path) or ".", exist_ok=True)

    def ambiguity_score(self, affinities: list) -> float:
        """1 - max sector affinity.  affinities: [(sector_id, score), ...]"""
        if not affinities:
            return 1.0
        return 1.0 - max(sc for _, sc in affinities)

    def handle(
        self, experience: Experience, affinities: list, cycle_id: str
    ) -> Literal["clear", "defer", "queue"]:
        """
        Evaluate ambiguity and decide fate:
          'clear'  → experience passes to consolidation pipeline
          'defer'  → re-evaluate next cycle
          'queue'  → escalate to human review
        """
        score = self.ambiguity_score(affinities)

        if score < AMBIGUITY_DEFER:
            return "clear"

        defer_count = self._get_defer_count(experience.uid)

        if score < AMBIGUITY_QUEUE and defer_count < MAX_DEFER_CYCLES:
            self._increment_defer(experience)
            return "defer"

        # Escalate to human queue
        entry = QueueEntry(
            experience_uid=experience.uid,
            text_preview=experience.text[:200],
            ambiguity_score=score,
            added_at=time.time(),
            defer_count=defer_count,
        )
        self._queue.append(entry)
        self._persist_entry(entry)
        return "queue"

    def queue_for_review(self, experience: Experience, score: float, rationale: str = "") -> None:
        """Escalate an experience to the human queue directly (used by the
        confidence-gated validator, not just by sector-ambiguity). `score` is the
        ambiguity/uncertainty in [0,1]."""
        entry = QueueEntry(
            experience_uid=experience.uid,
            text_preview=experience.text[:200],
            ambiguity_score=float(score),
            added_at=time.time(),
            defer_count=self._get_defer_count(experience.uid),
            rationale=rationale,
        )
        self._queue.append(entry)
        self._persist_entry(entry)

    def expire_old_entries(self) -> int:
        """Remove queue entries older than QUEUE_EXPIRY_DAYS. Returns count."""
        cutoff = time.time() - QUEUE_EXPIRY_DAYS * 86400
        before = len(self._queue)
        self._queue = [e for e in self._queue if e.added_at > cutoff or e.reviewed]
        return before - len(self._queue)

    def _get_defer_count(self, uid: str) -> int:
        for exp, count in self._deferred:
            if exp.uid == uid:
                return count
        return 0

    def _increment_defer(self, experience: Experience) -> None:
        for i, (exp, count) in enumerate(self._deferred):
            if exp.uid == experience.uid:
                self._deferred[i] = (exp, count + 1)
                return
        self._deferred.append((experience, 1))

    def _persist_entry(self, entry: QueueEntry) -> None:
        with open(self.queue_path, "a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    def pending_count(self) -> int:
        return sum(1 for e in self._queue if not e.reviewed)
