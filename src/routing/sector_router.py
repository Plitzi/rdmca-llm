"""
Sector Router — RDMCA §9 / §12
Translates STR affinity scores into a single sector assignment s*
for memory consolidation and parameter update purposes.

During inference: multiple sectors can activate simultaneously (weighted).
During consolidation: one sector s* per experience (highest affinity wins,
                      ties broken by recency of last sector update).
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

from src.model.lora import SECTORS


class SectorRouter:
    """Assigns each experience to its primary sector s* for consolidation."""

    def __init__(self):
        # Track last update time per sector for tie-breaking
        self._last_update: Dict[int, float] = {s: 0.0 for s in SECTORS}

    def assign(self, affinities: List[Tuple[int, float]]) -> Optional[int]:
        """
        Return the primary sector id s* for a given affinity vector.
        Returns None if no sector exceeds the minimum threshold.
        affinities: [(sector_id, score), ...] sorted descending.
        """
        if not affinities:
            return None
        # Highest affinity wins; tie-break by stalest update
        top_score = affinities[0][1]
        candidates = [(sid, sc) for sid, sc in affinities
                      if abs(sc - top_score) < 0.01]
        if len(candidates) == 1:
            return candidates[0][0]
        # Pick the sector that was updated least recently
        return min(candidates, key=lambda x: self._last_update[x[0]])[0]

    def record_update(self, sector_id: int, timestamp: float) -> None:
        self._last_update[sector_id] = timestamp

    def sector_name(self, sector_id: int) -> str:
        return SECTORS.get(sector_id, {}).get("name", f"S{sector_id}")
