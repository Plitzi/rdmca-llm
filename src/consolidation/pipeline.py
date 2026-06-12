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

import src.backend as backend

from src.memory.episodic_buffer import EpisodicBuffer, Experience
from src.memory.ltss import LTSS
from src.memory.mrf import mrf
from src.relevance.engine import RelevanceEngine
from src.model.bcf import BCFHead
# (MoE joint update is done inline via the backend engine; the isolated
# masked_sector_update path is no longer used by the consolidation pipeline.)
from src.consolidation.snapshot import SectorSnapshotManager
from src.consolidation.ambiguity import AmbiguityHandler
from src.consolidation.pgq import PGQ
from src.routing.semantic_router import Chunk


MIN_BATCH_PER_SECTOR = 8
CONSOL_SEQ_LEN       = 128    # token length used for consolidation LM updates
LTSS_DUPLICATE_COH   = 0.95  # cosine sim above which an experience is flagged
DEFAULT_SECTOR       = 1     # fallback sector when routing is unavailable
AUX_LOSS_WEIGHT      = 0.01  # weight of the MoE load-balance auxiliary loss

# PGQ growth-signal calibration (§14). These turn raw cycle telemetry into the four
# [0,1] inputs of the Growth Necessity Score (replacing the old hardcoded 0.0s that
# kept PGQ permanently "stable" — issue C1).
PGQ_SAT_REF   = 5.0   # grad-norm scale at which the busiest sector reads as saturated
PGQ_NOVEL_COH = 0.30  # max-cosine to LTSS below which an experience is a NOVEL cluster
PGQ_EXC_THETA = 0.50  # relevance score ≥ this counts as a cognitive-surprise experience


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
                 semantic_router=None,
                 sector_router=None,
                 validator=None,
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
        self._cycle_history: List[bool] = []   # per-cycle clean/not-clean flags
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Optional routing components for sector assignment (step 6). When not
        # provided, the pipeline falls back to exp.sector_assignment.
        self.semantic_router = semantic_router
        self.sector_router   = sector_router

        # Optional confidence-gated knowledge validator (validation.py). When set, it
        # decides each experience's fate by how confidently it can be validated
        # (self / research / peer-model / human) — generalising the sector-ambiguity
        # routing. When None, the pipeline uses ambiguity-only routing (legacy).
        self.validator = validator

        # Optional learning components. When `model` (with sectors attached)
        # and `tokenizer` are provided, the pipeline performs real masked
        # gradient updates; otherwise it runs the filter/audit pipeline only.
        self.model       = model
        self.tokenizer   = tokenizer
        self.optimizer   = (backend.current().engine.make_optimizer(
            model, lr=lr, weight_decay=0.0) if model is not None else None)
        self._last_rollback = False
        self._last_loss: Optional[float] = None   # consolidation CE → PGQ pred_error

    def _growth_metrics(self, final, delta_norms, busiest) -> Dict[str, float]:
        """Turn this cycle's telemetry into PGQ's four [0,1] growth signals (§14).
        Each is grounded in a real measurement instead of the old hardcoded 0.0:

          pred_error   — consolidation cross-entropy as a fraction of the maximum
                         entropy log(vocab): 0 = perfectly modeled, 1 = at chance.
                         Scale-free (no magic constant); high ⇒ capacity can't fit.
          saturation   — how hard the busiest sector was pushed (its grad norm),
                         saturating toward 1: a proxy for capacity pressure.
          cluster_novel— fraction of consolidated experiences far from everything in
                         LTSS (a genuinely new region of concept space).
          exc_rate     — fraction of cognitively SURPRISING experiences (relevance
                         score, already assigned by the R+ filter, ≥ PGQ_EXC_THETA).
        """
        n = len(final)
        # pred_error
        pred_error = 0.0
        if self._last_loss is not None and self.model is not None:
            vocab = getattr(self.model.cfg, "vocab_size", 0) or 2
            pred_error = float(np.clip(self._last_loss / np.log(max(vocab, 2)), 0.0, 1.0))
        # saturation (busiest sector's grad norm → saturating map)
        gnorm = float(delta_norms.get(f"S{busiest}", 0.0))
        saturation = float(1.0 - np.exp(-gnorm / PGQ_SAT_REF))
        # cluster_novel
        novel = 0
        for exp in final:
            try:
                coh = self.ltss.max_cosine_similarity(exp.embedding)
            except Exception:
                coh = 1.0
            if coh < PGQ_NOVEL_COH:
                novel += 1
        cluster_novel = (novel / n) if n else 0.0
        # exc_rate (cognitive surprise = high relevance, already scored upstream)
        exc = sum(1 for e in final if getattr(e, "relevance_score", 0.0) >= PGQ_EXC_THETA)
        exc_rate = (exc / n) if n else 0.0
        return dict(saturation=saturation, exc_rate=exc_rate,
                    pred_error=pred_error, cluster_novel=cluster_novel)

    def _moe_update(self, experiences, sectors_updated, delta_norms) -> None:
        """Joint MoE update: the gate + expert sectors (S1..S6) train together on
        the consolidation LM loss + load-balance aux loss. Snapshots gate+experts,
        steps once, and rolls back on a gradient-norm catastrophe. S7 is excluded
        (frozen → isolated)."""
        eng = backend.current().engine
        m   = self.model
        self._last_rollback = False
        batch = self._build_token_batch(experiences)
        if batch is None:
            return
        expert_ids = list(m._expert_ids)

        # Snapshot gate (id 0) + each expert before the update (enables rollback).
        self.snapshots.snapshot_before_update(0, eng.state_dict(m.gate))
        for sid in expert_ids:
            self.snapshots.snapshot_before_update(sid, eng.state_dict(m.sectors[sid]))

        # Trainable = gate + experts only; base frozen, S7 frozen (isolated).
        eng.set_trainable(m, [m.gate] + [m.sectors[s] for s in expert_ids])
        m.use_moe()
        lam = AUX_LOSS_WEIGHT

        def loss_fn(model):
            return model.mrl_loss(batch) + lam * model.aux_loss()

        grad_fn = eng.value_and_grad(m, loss_fn)
        loss, grads = grad_fn(m)
        eng.eval(loss)
        self._last_loss = float(eng.item(loss))      # → PGQ pred_error signal
        gnorm = eng.grad_norm(m, grads)
        eng.optimizer_step(self.optimizer, m, grads)
        eng.freeze_all(m)

        cat = self.snapshots.detect_catastrophe(
            0, benchmark_delta=0.0, kl_divergence=0.0, bcf_delta=0.0, grad_norm=gnorm)
        if cat:
            self.snapshots.rollback(0, m.gate)
            for sid in expert_ids:
                self.snapshots.rollback(sid, m.sectors[sid])
            self._last_rollback = True
        else:
            sectors_updated.extend(expert_ids)
        for sid in expert_ids:
            delta_norms[f"S{sid}"] = gnorm

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
        self._last_loss = None          # reset; _moe_update sets it if it runs

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
        # Flag near-duplicate / potentially-conflicting experiences (very high
        # similarity to an existing LTSS node) for review; they still pass
        # through so the MRF can decide redundancy vs. reinforcement.
        consistent: List[tuple] = []
        for exp, score in scored:
            try:
                coh = self.ltss.max_cosine_similarity(exp.embedding)
            except Exception:
                coh = 0.0
            if coh >= LTSS_DUPLICATE_COH:
                ltss_flagged += 1
            exp.coherence = coh                 # kept for the confidence validator (step 6)
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
            affinities = self._sector_affinities(exp)
            if self.sector_router is not None:
                exp.sector_assignment = (self.sector_router.assign(affinities)
                                         or exp.sector_assignment)
            if self.validator is not None:
                # Confidence-gated validation: trust own knowledge, else seek
                # research / a peer model, else escalate to a human.
                decision = self.validator.decide(exp, getattr(exp, "coherence", 0.0))
                if decision.fate == "consolidate":
                    final.append(exp)
                elif decision.fate == "defer":
                    deferred += 1
                elif decision.fate == "discard":
                    self.adv_buffer.append(exp)
                else:                            # queue → human review
                    human_queued += 1
            else:
                verdict = self.ambiguity.handle(exp, affinities, cycle_id)
                if verdict == "clear":
                    final.append(exp)
                elif verdict == "defer":
                    deferred += 1
                else:
                    human_queued += 1

        # Group by routed sector — kept for stats/PGQ only (training is MoE-joint).
        sector_groups: Dict[int, List[Experience]] = {}
        for exp in final:
            sid = exp.sector_assignment or 1
            sector_groups.setdefault(sid, []).append(exp)

        # --- Steps 7-8: MoE joint update over the EXPERT sectors (S1..S6) ---
        # The gate routes each token to its top-k experts; the gate + all expert
        # sectors train jointly on the consolidation LM loss + a load-balance aux
        # loss. One experience can thus update several sectors (multi-sectorial:
        # new terminology → Linguistic, the method → Formal). The safety sector
        # S7 is NEVER in the trainable set here — it stays frozen/isolated and is
        # only shaped by the BCF probe training (train_bcf_head).
        if (final and self.model is not None and self.tokenizer is not None
                and getattr(self.model, "gate", None) is not None
                and len(final) >= MIN_BATCH_PER_SECTOR):
            self._moe_update(final, sectors_updated, delta_norms)
            rollback = rollback or self._last_rollback
        elif final and (self.model is None or self.tokenizer is None):
            # Filter-only mode (no model): report which experts would have updated.
            sectors_updated.extend(sorted({(e.sector_assignment or DEFAULT_SECTOR)
                                           for e in final}))

        # --- Step 9: PGQ — fed REAL growth signals from this cycle (was hardcoded
        # to 0.0, so PGQ was permanently "stable" and never grew capacity — C1). ---
        busiest = (max(sector_groups, key=lambda s: len(sector_groups[s]))
                   if sector_groups else DEFAULT_SECTOR)
        gm = self._growth_metrics(final, delta_norms, busiest)
        pgq_result = self.pgq.evaluate(
            cycle_id, busiest_sector_id=busiest, sectors=self.sectors,
            model=self.model, **gm,
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
        if (self.model is None or self.tokenizer is None or self.bcf is None
                or not getattr(self.tokenizer, "ready", False)):
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
        toks = backend.current().ops.array(np.asarray([ids[:CONSOL_SEQ_LEN]], dtype=np.int64))
        self.model.set_active_sectors([])          # core-only for the BCF gate
        h = self.model(toks)[:, -1, :]             # final-token hidden state
        return bool(self.bcf.is_permissible(h).item())

    def _get_adapter(self, sid: int):
        """Resolve a sector adapter from the model (preferred) or the dict."""
        if self.model is not None and self.model.sectors:
            return self.model.sectors.get(sid)
        return self.sectors.get(sid)

    def _sector_affinities(self, exp: Experience):
        """Sector affinities for an experience (STR §12). Uses the semantic
        router over the experience embedding when available; otherwise falls
        back to the experience's pre-assigned sector."""
        if self.semantic_router is not None and exp.embedding is not None:
            emb = backend.current().ops.array(np.asarray(exp.embedding, dtype=np.float32))
            chunk = Chunk(tokens=[], modality=exp.modality)
            routed = self.semantic_router.route(chunk, emb)
            if routed:
                return routed
        return [(exp.sector_assignment or DEFAULT_SECTOR, 1.0)]

    def _build_token_batch(self, group: List[Experience]):
        """
        Tokenize a group of experiences into a padded [B, CONSOL_SEQ_LEN+1]
        batch for the masked LM consolidation update. Returns None if no text
        is available to learn from.
        """
        if not getattr(self.tokenizer, "ready", False):
            return None
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
        return backend.current().ops.array(np.asarray(rows, dtype=np.int64))

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
