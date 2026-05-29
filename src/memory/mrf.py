"""
Memory Reevaluation Function (MRF) — RDMCA §10 / Implementation Guide §2.2
Runs at each consolidation cycle. Evaluates every T1/T2 experience and
assigns one of three fates: promote → T3 (LTSS), retain, or expire.

score = σ1·R+(e,s) + σ2·Freq(e) + σ3·Exc(e) + σ4·Coh(e)
        then multiplied by temporal decay: exp(-λ·Δt/86400)

Fate thresholds:
  score ≥ THETA_PROMOTE → promote to LTSS
  score ≥ THETA_RETAIN  → retain in consolidation buffer
  otherwise             → expire (remove from all buffers)

Special case — Cognitive Surprise:
  If Exc(e) > 2.5 σ, the experience bypasses the standard fate pipeline
  and is promoted directly to LTSS regardless of other scores.
"""
from __future__ import annotations
import math
import time
from typing import Literal

import numpy as np

from .episodic_buffer import Experience
from .ltss import LTSS


THETA_PROMOTE = 0.65
THETA_RETAIN  = 0.35
SURPRISE_SIGMA = 2.5
DECAY_LAMBDA   = 0.05   # per-day decay rate


Fate = Literal["promote", "retain", "expire"]


def z_score(emb: np.ndarray, centroid: np.ndarray,
            std: np.ndarray) -> float:
    """Exceptionality: max z-score component relative to LTSS distribution."""
    denom = np.maximum(std, 1e-8)
    return float(np.max(np.abs(emb - centroid) / denom))


def mrf(experience: Experience,
        relevance_score: float,
        ltss: LTSS,
        sigma: tuple = (0.4, 0.2, 0.2, 0.2)) -> Fate:
    """
    Evaluate a single experience and return its fate.

    Args:
        experience:      the experience to evaluate
        relevance_score: R+(e,s) already computed by the Relevance Engine
        ltss:            the Long-Term Semantic Store
        sigma:           (w_R, w_Freq, w_Exc, w_Coh) weights
    """
    s1, s2, s3, s4 = sigma
    e = experience.embedding

    freq = experience.retrieval_count / max(experience.age_days, 1)

    centroid = ltss.global_centroid
    std      = ltss.global_std
    if centroid is not None and std is not None:
        exc = z_score(e, centroid, std)
        # Cognitive surprise — direct promotion
        if exc >= SURPRISE_SIGMA:
            return "promote"
    else:
        exc = 0.0

    coh   = ltss.max_cosine_similarity(e)
    score = s1 * relevance_score + s2 * freq + s3 * exc + s4 * coh

    # Temporal decay
    delta_t = time.time() - experience.timestamp
    decayed = score * math.exp(-DECAY_LAMBDA * delta_t / 86400.0)

    if decayed >= THETA_PROMOTE:
        return "promote"
    if decayed >= THETA_RETAIN:
        return "retain"
    return "expire"
