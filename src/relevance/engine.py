"""
Relevance Engine — RDMCA §5
Central scoring function governing all cognitive operations.

R+(e, s) = σ( w_N·N(e,s) + w_U·U(e,s) + w_C·C(e,s) - w_R·Rep(e,s) ) - P(e,s)

Components:
  N(e,s)   — Novelty:     approximate mutual information via cosine distance
  U(e,s)   — Utility:     gradient alignment with recent loss signal
  C(e,s)   — Coherence:   LTSS retrieval similarity
  Rep(e,s) — Repetition:  temporal-decay similarity to episodic buffer
  P(e,s)   — Penalty:     adversarial/injection attack taxonomy (see penalty.py)

Decision thresholds (§5.2):
  θ1 — memory retrieval eligibility
  θ2 — consolidation buffer eligibility
  θ3 — parameter update eligibility
"""
from __future__ import annotations
import time
import math
from typing import List, Optional, Tuple

import mlx.core as mx
import numpy as np


# Default thresholds (tunable via config)
THETA_1 = 0.3   # memory retrieval
THETA_2 = 0.5   # consolidation buffer
THETA_3 = 0.7   # parameter update


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def novelty(e_emb: np.ndarray, state_emb: np.ndarray) -> float:
    """N(e,s) — cosine distance from current state embedding. §5.1"""
    return 1.0 - cosine_similarity(e_emb, state_emb)


def utility(e_emb: np.ndarray, grad_buffer: Optional[np.ndarray]) -> float:
    """U(e,s) — alignment with recent loss gradient. §5.1"""
    if grad_buffer is None:
        return 0.5
    return max(0.0, cosine_similarity(e_emb, grad_buffer))


def coherence(e_emb: np.ndarray, ltss) -> float:
    """C(e,s) — max cosine similarity among top-5 LTSS neighbors. §5.1"""
    results = ltss.search(e_emb, k=5)
    if not results:
        return 0.5
    return max(score for _, score in results)


def repetition(e_emb: np.ndarray,
               episodic_buffer: list,
               lambda_decay: float = 0.1) -> float:
    """Rep(e,s) — temporal-decay similarity to past experiences. §5.1"""
    if not episodic_buffer:
        return 0.0
    now = time.time()
    scores = []
    for past in episodic_buffer:
        sim   = cosine_similarity(e_emb, past.embedding)
        decay = math.exp(-lambda_decay * (now - past.timestamp))
        scores.append(sim * decay)
    return sum(scores) / len(scores)


class RelevanceEngine:
    """
    Scores every incoming experience with R+(e, s).
    Runs on CPU — target latency < 5ms per experience (M2).
    """

    def __init__(self,
                 ltss=None,
                 weights: Tuple[float, float, float, float] = (0.4, 0.2, 0.2, 0.2),
                 thresholds: Tuple[float, float, float] = (THETA_1, THETA_2, THETA_3)):
        self.ltss       = ltss
        self.w_N, self.w_U, self.w_C, self.w_R = weights
        self.theta1, self.theta2, self.theta3   = thresholds
        self._state_emb:   Optional[np.ndarray] = None
        self._grad_buffer: Optional[np.ndarray] = None

    def update_state(self, state_emb: np.ndarray) -> None:
        self._state_emb = state_emb

    def update_grad_buffer(self, grad: np.ndarray) -> None:
        self._grad_buffer = grad / (np.linalg.norm(grad) + 1e-8)

    def score(self, experience) -> float:
        """
        Returns R+(e,s) ∈ ℝ.  Negative values indicate adversarial content.
        experience must have: .embedding (np.ndarray), .episodic_buffer (list)
        """
        from .penalty import penalty_score
        e = experience.embedding
        s = self._state_emb if self._state_emb is not None else np.zeros_like(e)

        N   = novelty(e, s)
        U   = utility(e, self._grad_buffer)
        C   = coherence(e, self.ltss) if self.ltss else 0.5
        Rep = repetition(e, experience.episodic_context)
        P   = penalty_score(experience)

        raw = self.w_N * N + self.w_U * U + self.w_C * C - self.w_R * Rep
        return float(1 / (1 + math.exp(-raw))) - P   # sigmoid then subtract penalty

    # Threshold helpers
    def retrieval_eligible(self, score: float) -> bool:
        return score >= self.theta1

    def consolidation_eligible(self, score: float) -> bool:
        return score >= self.theta2

    def update_eligible(self, score: float) -> bool:
        return score >= self.theta3
