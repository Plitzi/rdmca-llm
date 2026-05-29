"""
Sector Snapshot & Rollback System — RDMCA §16 / Implementation Guide §2.3
7-day rolling snapshot buffer per sector.
Automatic rollback on catastrophe detection (CAT).

Catastrophe triggers (any one):
  perf_drop   — benchmark delta < -DELTA_PERF
  kl_shift    — KL divergence of output distribution > DELTA_KL
  bcf_change  — any change in BCF probe set accuracy
  grad_spike  — gradient norm > μ + 3σ over rolling window

After 3 consecutive CAT triggers, the sector is frozen automatically.
"""
from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import mlx.core as mx

BUFFER_DEPTH     = 7     # days
DELTA_PERF       = 0.02  # 2% performance drop threshold
DELTA_KL         = 0.05  # KL divergence threshold
GRAD_SPIKE_SIGMA = 3.0


class SectorSnapshotManager:
    """Manages per-sector snapshots and rollback."""

    def __init__(self, snapshot_dir: str = "dist/snapshots"):
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._cat_counts:   Dict[int, int]            = {}
        self._frozen:       Dict[int, bool]           = {}
        self._grad_history: Dict[int, List[float]]    = {}

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot_before_update(self, sector_id: int,
                                sector_params: dict,
                                cycle_t: Optional[float] = None) -> Path:
        """Save sector params before a gradient update."""
        t   = cycle_t or time.time()
        tag = int(t)
        path = self.snapshot_dir / f"sector_{sector_id}_t{tag}.npz"
        mx.savez(str(path), **sector_params)
        self._prune_old_snapshots(sector_id)
        return path

    def _prune_old_snapshots(self, sector_id: int) -> None:
        """Keep only the last BUFFER_DEPTH snapshots per sector."""
        snaps = sorted(
            self.snapshot_dir.glob(f"sector_{sector_id}_t*.npz")
        )
        for old in snaps[:-BUFFER_DEPTH]:
            old.unlink()

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback(self, sector_id: int, sector_adapter,
                 cycle_t: Optional[float] = None) -> bool:
        """Restore sector to the most recent (or specified) snapshot."""
        snaps = sorted(self.snapshot_dir.glob(f"sector_{sector_id}_t*.npz"))
        if not snaps:
            logging.error(f"ROLLBACK failed: no snapshot for sector {sector_id}")
            return False
        target = snaps[-1]
        params = mx.load(str(target))
        sector_adapter.load_weights(list(params.items()))
        mx.eval(sector_adapter.parameters())
        logging.warning(f"ROLLBACK: sector {sector_id} restored from {target.name}")
        self._cat_counts[sector_id] = self._cat_counts.get(sector_id, 0) + 1
        if self._cat_counts[sector_id] >= 3:
            self._frozen[sector_id] = True
            logging.critical(f"Sector {sector_id} FROZEN after 3 consecutive CATs")
        return True

    # ------------------------------------------------------------------
    # Catastrophe detection
    # ------------------------------------------------------------------

    def detect_catastrophe(self, sector_id: int,
                            benchmark_delta: float,
                            kl_divergence: float,
                            bcf_delta: float,
                            grad_norm: float) -> bool:
        """Return True if any catastrophe trigger fires."""
        hist = self._grad_history.setdefault(sector_id, [])
        hist.append(grad_norm)
        if len(hist) > 100:
            hist.pop(0)

        perf_drop  = benchmark_delta < -DELTA_PERF
        kl_shift   = kl_divergence > DELTA_KL
        bcf_change = abs(bcf_delta) > 0.001
        grad_spike = (len(hist) >= 10 and
                      grad_norm > _mean(hist) + GRAD_SPIKE_SIGMA * _std(hist))

        triggered = any([perf_drop, kl_shift, bcf_change, grad_spike])
        if triggered:
            reasons = [
                f"perf_drop={benchmark_delta:.4f}" if perf_drop else None,
                f"kl={kl_divergence:.4f}"          if kl_shift else None,
                f"bcf_delta={bcf_delta:.4f}"        if bcf_change else None,
                f"grad_spike={grad_norm:.2f}"        if grad_spike else None,
            ]
            logging.warning(f"CAT sector {sector_id}: "
                            + " | ".join(r for r in reasons if r))
        else:
            self._cat_counts[sector_id] = 0   # reset on clean cycle

        return triggered

    def is_frozen(self, sector_id: int) -> bool:
        return self._frozen.get(sector_id, False)


def _mean(xs: list) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
