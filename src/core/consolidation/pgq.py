"""
Parametric Growth Quantifier (PGQ) — RDMCA §14 / Implementation Guide §4.2
Monitors each consolidation cycle for signs that current parametric capacity
is insufficient. Triggers sector expansion or new sector creation.

Growth Necessity Score:
  GNS(t) = w_s·Sat(t) + w_e·Exc_rate(t) + w_err·Err(t) + w_c·Cluster_novel(t)

  Sat(t)          — sector saturation (gradient norms near capacity)
  Exc_rate(t)     — rate of cognitive-surprise experiences
  Err(t)          — prediction error on recent experiences
  Cluster_novel(t)— fraction of experiences in novel embedding clusters

GNS thresholds → action:
  < THETA_STABLE   → no action
  < THETA_EXPAND   → monitor (3 cycles)
  < THETA_SECTOR   → expand: increase LoRA rank by delta_r (max 5%/cycle)
  ≥ THETA_SECTOR   → create: instantiate S_{n+1} at rank 4
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Literal

THETA_STABLE = 0.3
THETA_EXPAND = 0.5
THETA_SECTOR = 0.75
DELTA_RANK = 2  # LoRA rank increment when expanding a sector


PGQDecision = Literal["stable", "monitor", "expand", "new_sector"]


@dataclass
class PGQResult:
    cycle_id: str
    gns: float
    decision: PGQDecision
    sector_id: int | None = None
    action_detail: str = ""
    timestamp: float = 0.0

    def __post_init__(self):
        self.timestamp = self.timestamp or time.time()


class PGQ:
    """
    Parametric Growth Quantifier.
    Call .evaluate() after every consolidation cycle.
    """

    def __init__(self, weights: tuple = (0.3, 0.25, 0.25, 0.2)):
        self.w_s, self.w_e, self.w_err, self.w_c = weights
        self._monitor_counts: dict[int, int] = {}
        self._history: list[PGQResult] = []
        self._next_sector_id: int = 8  # S8, S9, ... on creation

    def gns(
        self, saturation: float, exc_rate: float, pred_error: float, cluster_novel: float
    ) -> float:
        """Compute Growth Necessity Score ∈ [0, 1]."""
        return (
            self.w_s * saturation
            + self.w_e * exc_rate
            + self.w_err * pred_error
            + self.w_c * cluster_novel
        )

    def evaluate(
        self,
        cycle_id: str,
        saturation: float,
        exc_rate: float,
        pred_error: float,
        cluster_novel: float,
        busiest_sector_id: int,
        sectors: dict,
        model=None,
    ) -> PGQResult:
        """
        Evaluate growth necessity and apply the resulting structural change.
        sectors: {sector_id: SectorAdapter} (rank growth applied in place).
        model:   RDMCAFoundational — required to instantiate a new sector.
        """
        score = self.gns(saturation, exc_rate, pred_error, cluster_novel)

        if score < THETA_STABLE:
            result = PGQResult(cycle_id, score, "stable")

        elif score < THETA_EXPAND:
            cnt = self._monitor_counts.get(busiest_sector_id, 0) + 1
            self._monitor_counts[busiest_sector_id] = cnt
            result = PGQResult(
                cycle_id,
                score,
                "monitor",
                sector_id=busiest_sector_id,
                action_detail=f"observation cycle {cnt}/3",
            )

        elif score < THETA_SECTOR:
            # Expand: grow LoRA rank of the highest-load sector in place.
            adapter = sectors.get(busiest_sector_id)
            if adapter is not None:
                old_rank = adapter.rank
                new_rank = adapter.grow_rank(DELTA_RANK)
                logging.info(f"PGQ: expand S{busiest_sector_id} rank {old_rank} → {new_rank}")
                result = PGQResult(
                    cycle_id,
                    score,
                    "expand",
                    sector_id=busiest_sector_id,
                    action_detail=f"rank {old_rank} → {new_rank}",
                )
                self._monitor_counts[busiest_sector_id] = 0
            else:
                result = PGQResult(cycle_id, score, "monitor", action_detail="adapter not found")

        else:
            # Create a new adaptive sector at minimum viable rank (§10.7.4).
            new_sid = self._next_sector_id
            self._next_sector_id += 1
            if model is not None and hasattr(model, "add_sector"):
                model.add_sector(new_sid, rank=4)
                sectors[new_sid] = model.sectors[new_sid]
                detail = f"S{new_sid} created at rank 4"
            else:
                detail = f"S{new_sid} requested (no model to instantiate)"
            logging.info(f"PGQ: {detail}")
            result = PGQResult(
                cycle_id, score, "new_sector", sector_id=new_sid, action_detail=detail
            )

        self._history.append(result)
        return result
