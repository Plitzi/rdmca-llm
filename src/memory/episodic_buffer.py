"""
Episodic Buffer (T1) — RDMCA §10 / §8 Memory Graph System
In-memory short-term buffer. Holds raw experiences before RE scoring.
Experiences that pass θ2 move to the consolidation buffer (T2).
Experiences promoted by MRF move to LTSS (T3) permanently.

Memory tier hierarchy:
  T1 — Episodic buffer (this file): in-memory, current session
  T2 — Consolidation buffer:        priority queue, pending sector update
  T3 — LTSS:                        long-term persistent (ltss.py)
"""
from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Experience:
    text: str
    embedding: np.ndarray
    modality: str = "text"          # text | image | audio | mixed
    feedback: str = "neutral"       # neutral | accepted | corrected — the user's
                                    # reaction; drives the Relevance Engine's Utility
                                    # (a corrected error is the highest-value signal).
    timestamp: float = field(default_factory=time.time)
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    relevance_score: float = 0.0
    sector_assignment: Optional[int] = None   # s* after STR routing
    retrieval_count: int = 0
    age_days: float = 0.0
    episodic_context: List["Experience"] = field(default_factory=list)

    def update_age(self) -> None:
        self.age_days = (time.time() - self.timestamp) / 86400.0


class EpisodicBuffer:
    """
    Ring buffer for T1 experiences.
    Max capacity = max_size; oldest experiences are evicted on overflow.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size  = max_size
        self._buffer:  List[Experience] = []

    def add(self, exp: Experience) -> None:
        if len(self._buffer) >= self.max_size:
            self._buffer.pop(0)
        self._buffer.append(exp)

    def all(self) -> List[Experience]:
        return list(self._buffer)

    def recent(self, n: int = 50) -> List[Experience]:
        return self._buffer[-n:]

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)
