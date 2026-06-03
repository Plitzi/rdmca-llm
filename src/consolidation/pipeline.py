"""
Daily Consolidation Pipeline — RDMCA §11 / Implementation Guide §1.7
Runs during system idle time (CPU < 20% for 5+ min).
Executes the full 9-step consolidation cycle on the current buffer.

Pipeline steps (§1.7.1):
  1. Load episodic buffer from SQLite
  2. BCF filter: discard B(a,s)=0 → adversarial buffer
  3. R+ filter: discard R+(e,s) < 0 → adversarial buffer
  4. LTSS consistency filter: flag KL > ε → review queue
  5. MRF: promote / retain / expire each T1/T2 experience
  6. Ambiguity scoring: clear / defer / human queue
  7. Group by sector assignment s*(e)
  8. Masked gradient update per sector (≥ min_batch)
  9. PGQ evaluation + audit log
"""
from __future__ import annotations
import json
import logging
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from src.memory.episodic_buffer import EpisodicBuffer, Experience
from src.memory.ltss import LTSS
from src.memory.mrf import mrf
from src.relevance.engine import RelevanceEngine
from src.model.bcf import BCFHead
from src.model.lora import masked_sector_update
from src.consolidation.snapshot import SectorSnapshotManager
from src.consolidation.ambiguity import AmbiguityHandler
from src.consolidation.pgq import PGQ


MIN_BATCH_PER_SECTOR = 8
CONSOL_SEQ_LEN       = 128    # token length used for consolidation LM updates


@dataclass
class AuditEntry:
    cycle_id: str
    buffer_size_raw: int
    bcf_rejected: int
    r_neg_rejected: int
    ltss_flagged: int
    deferred_1: int
    human_queue_added: int
    sectors_updated: List[int]
    param_delta_norms: Dict[str, float]
    rollback_triggered: bool
    gns: float
    health_score: float
    modality_balance: Dict[str, int]
    clean_cycle: bool
    timestamp: float = 0.0

    def __post_init__(self):
        self.timestamp = self.timestamp or time.time()


class ConsolidationPipeline:

    def __init__(self,
                 buffer: EpisodicBuffer,
                 ltss: LTSS,
                 re: RelevanceEngine,
                 bcf: BCFHead,
                 sectors: dict,
                 snapshot_mgr: SectorSnapshotManager,
                 ambiguity: AmbiguityHandler,
                 pgq: PGQ,
                 log_dir: str = "logs",
                 adversarial_buffer: Optional[list] = None,
                 model=None,
                 tokenizer=None,
                 lr: float = 1e-4):
        self.buffer       = buffer
        self.ltss         = ltss
        self.re           = re
        self.bcf          = bcf
        self.sectors      = sectors
        self.snapshots    = snapshot_mgr
        self.ambiguity    = ambiguity
        self.pgq          = pgq
        self.log_dir      = Path(log_dir)
        self.adv_buffer   = adversarial_buffer if adversarial_buffer is not None else []
        self._cycle_history: List[AuditEntry] = []
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Optional learning components. When `model` (with sectors attached)
        # and `tokenizer` are provided, the pipeline performs real masked
        # gradient updates; otherwise it runs the filter/audit pipeline only.
        self.model       = model
        self.tokenizer   = tokenizer
        self.optimizer   = optim.AdamW(learning_rate=lr) if model is not None else None

    def run(self) -> AuditEntry:
        """Execute one full consolidation cycle. Returns the audit log entry."""
        cycle_id = str(uuid.uuid4())[:8]
        logging.info(f"[consolidation] cycle {cycle_id} started")
        t0 = time.time()

        experiences = self.buffer.all()
        raw_count   = len(experiences)

        bcf_rejected = r_neg_rejected = ltss_flagged = 0
        deferred = human_queued = 0
        sectors_updated: List[int] = []
        delta_norms: Dict[str, float] = {}
        rollback = False

        # --- Step 2: BCF filter ---
        clean: List[Experience] = []
        for exp in experiences:
            if self._bcf_permissible(exp):
                clean.append(exp)
            else:
                self.adv_buffer.append(exp)
                bcf_rejected += 1

        # --- Step 3: R+ filter ---
        scored: List[tuple] = []
        for exp in clean:
            score = self.re.score(exp)
            exp.relevance_score = score
            if score < 0:
                self.adv_buffer.append(exp)
                r_neg_rejected += 1
            else:
                scored.append((exp, score))

        # --- Step 4: LTSS consistency filter ---
        consistent: List[tuple] = []
        for exp, score in scored:
            # TODO: compute KL divergence against LTSS representations
            consistent.append((exp, score))

        # --- Step 5: MRF ---
        to_consolidate: List[Experience] = []
        for exp, score in consistent:
            fate = mrf(exp, score, self.ltss)
            if fate == "promote":
                from src.memory.ltss import LTSSNode
                self.ltss.add(LTSSNode(
                    id=exp.uid, embedding=exp.embedding,
                    content=exp.text, modality=exp.modality,
                ))
            if fate in ("promote", "retain"):
                to_consolidate.append(exp)

        # --- Step 6: Ambiguity scoring ---
        final: List[Experience] = []
        for exp in to_consolidate:
            # TODO: get affinities from STR
            affinities = [(exp.sector_assignment or 1, 0.8)]
            verdict = self.ambiguity.handle(exp, affinities, cycle_id)
            if verdict == "clear":
                final.append(exp)
            elif verdict == "defer":
                deferred += 1
            else:
                human_queued += 1

        # --- Steps 7-8: Group by sector and masked update ---
        sector_groups: Dict[int, List[Experience]] = {}
        for exp in final:
            sid = exp.sector_assignment or 1
            sector_groups.setdefault(sid, []).append(exp)

        for sid, group in sector_groups.items():
            if len(group) < MIN_BATCH_PER_SECTOR:
                continue
            if self.snapshots.is_frozen(sid):
                continue
            adapter = self._get_adapter(sid)
            if adapter is None:
                continue

            # Snapshot the sector before touching it (enables rollback).
            sector_params = dict(tree_flatten(adapter.parameters()))
            self.snapshots.snapshot_before_update(sid, sector_params)

            # Without a model + tokenizer we can only do the filter pipeline.
            if self.model is None or self.tokenizer is None:
                sectors_updated.append(sid)
                delta_norms[f"S{sid}"] = 0.0
                continue

            batch = self._build_token_batch(group)
            if batch is None:
                continue

            def loss_fn(model, _batch=batch, _sid=sid):
                model.set_active_sectors([(_sid, 1.0)])
                return model.mrl_loss(_batch)

            loss_val, gnorm = masked_sector_update(
                self.model, sid, loss_fn, self.optimizer)
            delta_norms[f"S{sid}"] = gnorm

            # Catastrophe detection (gradient-norm anomaly is computable
            # in-loop; perf/KL/BCF probes are wired separately).
            cat = self.snapshots.detect_catastrophe(
                sid, benchmark_delta=0.0, kl_divergence=0.0,
                bcf_delta=0.0, grad_norm=gnorm)
            if cat:
                self.snapshots.rollback(sid, adapter)
                rollback = True
            else:
                sectors_updated.append(sid)

        # --- Step 9: PGQ ---
        pgq_result = self.pgq.evaluate(
            cycle_id, saturation=0.0, exc_rate=0.0,
            pred_error=0.0, cluster_novel=0.0,
            busiest_sector_id=1, sectors=self.sectors,
        )

        # --- Audit log ---
        modality_counts: Dict[str, int] = {}
        for exp in experiences:
            modality_counts[exp.modality] = modality_counts.get(exp.modality, 0) + 1

        health = self._rolling_health(not rollback and human_queued == 0)
        entry = AuditEntry(
            cycle_id=cycle_id,
            buffer_size_raw=raw_count,
            bcf_rejected=bcf_rejected,
            r_neg_rejected=r_neg_rejected,
            ltss_flagged=ltss_flagged,
            deferred_1=deferred,
            human_queue_added=human_queued,
            sectors_updated=sectors_updated,
            param_delta_norms=delta_norms,
            rollback_triggered=rollback,
            gns=pgq_result.gns,
            health_score=health,
            modality_balance=modality_counts,
            clean_cycle=(not rollback and human_queued == 0),
        )
        self._write_log(entry)
        self.buffer.clear()
        logging.info(f"[consolidation] cycle {cycle_id} done in "
                     f"{time.time()-t0:.1f}s | health={health:.2f}")
        return entry

    def _bcf_permissible(self, exp: Experience) -> bool:
        """
        Behavioral Constraint check (§15.3). Uses the frozen foundational
        hidden state of the experience text. Falls back to permit-all only
        when the model/tokenizer/BCF head are not all available (filter
        pipeline-only mode), so consolidation never silently drops data in
        test/toy runs.
        """
        if self.model is None or self.tokenizer is None or self.bcf is None:
            return True
        text = getattr(exp, "text", "") or ""
        if not text:
            return True
        try:
            ids = self.tokenizer.encode(text, add_eos=True)
        except TypeError:
            ids = self.tokenizer.encode(text)
        if not ids:
            return True
        toks = mx.array(ids[:CONSOL_SEQ_LEN])[None]
        self.model.set_active_sectors([])          # core-only for the BCF gate
        h = self.model(toks)[:, -1, :]             # final-token hidden state
        return bool(self.bcf.is_permissible(h).item())

    def _get_adapter(self, sid: int):
        """Resolve a sector adapter from the model (preferred) or the dict."""
        if self.model is not None and self.model.sectors:
            return self.model.sectors.get(sid)
        return self.sectors.get(sid)

    def _build_token_batch(self, group: List[Experience]):
        """
        Tokenize a group of experiences into a padded [B, CONSOL_SEQ_LEN+1]
        batch for the masked LM consolidation update. Returns None if no text
        is available to learn from.
        """
        L = CONSOL_SEQ_LEN + 1
        rows = []
        for exp in group:
            text = getattr(exp, "text", "") or ""
            if not text:
                continue
            try:
                ids = self.tokenizer.encode(text, add_eos=True)
            except TypeError:
                ids = self.tokenizer.encode(text)
            if not ids:
                continue
            ids = ids[:L]
            if len(ids) < L:
                ids = ids + [0] * (L - len(ids))   # pad_id = 0
            rows.append(ids)
        if not rows:
            return None
        return mx.array(rows)

    def _rolling_health(self, clean: bool, window: int = 30) -> float:
        self._cycle_history.append(clean)   # type: ignore
        if len(self._cycle_history) > window:
            self._cycle_history.pop(0)
        clean_list = [c for c in self._cycle_history if isinstance(c, bool)]
        return sum(clean_list) / max(len(clean_list), 1)

    def _write_log(self, entry: AuditEntry) -> None:
        path = self.log_dir / f"cycle_{entry.cycle_id}.json"
        with open(path, "w") as f:
            json.dump(asdict(entry), f, indent=2)
