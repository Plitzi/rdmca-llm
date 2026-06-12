"""
Dynamic context manager — RDMCA §12 (Semantic Token Router context slots).

Replaces the single flat context window with a set of per-sector context slots
(Eq. 18: Context = ⋃_s {C_slot(s) : |C_slot(s)| ≤ W_s}). Incoming tokens are
segmented into chunks, each chunk is routed to its sector slot(s) by domain
affinity, and when a slot overflows its bound W_s the oldest chunk is EVICTED to
the episodic buffer (T1) — not discarded (§12.6). The model then attends over the
assembled union of the active slots.

This is the FRONT half of the memory pipeline whose back half (episodic → LTSS →
recall) already exists: slots are working memory; eviction feeds the consolidation
path; MemoryRecall reads it back. It is a CONTEXT-ASSEMBLY layer on top of the
attention — the model (KV cache + GQA + SDPA) is unchanged; the manager only
decides which tokens are in the window.

Routing is pluggable so it can use the cheapest available trained signal:
  • `route_fn(tokens) -> [(sector_id, affinity)]` — e.g. the trained MoE gate
    aggregated over the chunk (best, post-sector-attach); or
  • `embed_fn(tokens) -> np.ndarray` + the STR classifier; or
  • neither → a single slot (degrades to a flat window, zero regression).

Additive and OPT-IN: the default chat/agent path is unchanged unless a manager is
supplied, so an untuned router can never make the base worse than today.
"""
from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from src.routing.semantic_router import SemanticTokenRouter, Chunk, NUM_SECTORS
from src.memory.episodic_buffer import EpisodicBuffer, Experience


class ContextManager:
    def __init__(self, d_model: int, context_len: int = 2048,
                 slot_len: Optional[int] = None,
                 buffer: Optional[EpisodicBuffer] = None,
                 embed_fn: Optional[Callable[[List[int]], np.ndarray]] = None,
                 route_fn: Optional[Callable[[List[int]], List[Tuple[int, float]]]] = None,
                 decode_fn: Optional[Callable[[List[int]], str]] = None):
        self.context_len = context_len
        self.decode_fn = decode_fn
        # Per-sector slot bound W_s. Default splits the window so a few active
        # sectors fit together; a single dominant sector can still use up to W_s.
        self.slot_len = slot_len or max(256, context_len // 2)
        self.str = SemanticTokenRouter(d_model, context_len=self.slot_len)
        self.buffer = buffer if buffer is not None else EpisodicBuffer()
        self.embed_fn = embed_fn
        self.route_fn = route_fn
        # Insertion order across slots, so assembly can preserve conversational
        # recency instead of a fixed sector order (which would scramble a dialogue).
        self._order: List[Tuple[int, int]] = []      # (sector_id, chunk_seq)
        self._chunks: Dict[int, List[int]] = {}       # chunk_seq -> tokens
        self._seq = 0

    # ------------------------------------------------------------------
    def add(self, tokens: List[int]) -> None:
        """Segment `tokens`, route each chunk to its sector slot(s), evict overflow
        to the episodic buffer."""
        for chunk in self.str.segment(tokens):
            for sid, _ in self._route(chunk):
                self._add_to_slot(sid, chunk.tokens)

    def _route(self, chunk: Chunk) -> List[Tuple[int, float]]:
        if self.route_fn is not None:
            routed = self.route_fn(chunk.tokens)
            if routed:
                return routed
        if self.embed_fn is not None:
            import src.backend as backend
            emb = self.embed_fn(chunk.tokens)
            if emb is not None:
                routed = self.str.route(chunk, backend.current().ops.array(
                    np.asarray(emb, dtype=np.float32)))
                if routed:
                    return routed
        return [(1, 1.0)]                              # no router → single slot (flat-ish)

    def _add_to_slot(self, sid: int, tokens: List[int]) -> None:
        buf = self.str._contexts[sid]
        self._seq += 1
        self._order.append((sid, self._seq))
        self._chunks[self._seq] = list(tokens)
        buf.extend(tokens)
        overflow = len(buf) - self.slot_len
        if overflow > 0:
            evicted = buf[:overflow]
            del buf[:overflow]
            self._evict(sid, evicted)

    def _evict(self, sid: int, tokens: List[int]) -> None:
        """Overflowed chunk → episodic buffer (T1) for consolidation, not discarded.
        Decode the tokens back to TEXT when a decoder is available — an evicted chunk
        with `text=""` is dead weight in the consolidation pipeline (nothing to
        abstract into LTSS), which is exactly what gets re-read at recall time."""
        if not tokens:
            return
        emb = self.embed_fn(tokens) if self.embed_fn is not None else None
        if emb is None:
            emb = np.zeros((1,), dtype=np.float32)
        text = self.decode_fn(tokens) if self.decode_fn is not None else ""
        self.buffer.add(Experience(text=text, embedding=np.asarray(emb, dtype=np.float32),
                                   sector_assignment=sid))

    # ------------------------------------------------------------------
    def assemble(self, max_len: Optional[int] = None) -> List[int]:
        """The active context: chunks across all slots in insertion (recency) order,
        capped to `max_len` (keeping the most recent)."""
        cap = max_len or self.context_len
        seq: List[int] = []
        for _, cseq in self._order:
            seq.extend(self._chunks.get(cseq, []))
        return seq[-cap:]

    def active_sectors(self) -> List[int]:
        return [s for s in range(1, NUM_SECTORS + 1) if self.str.get_context(s)]

    def clear(self) -> None:
        self.str.clear_contexts()
        self._order.clear()
        self._chunks.clear()
        self._seq = 0


def build_context_manager(model, tokenizer=None, context_len: Optional[int] = None,
                          buffer: Optional[EpisodicBuffer] = None) -> ContextManager:
    """Wire a ContextManager to a loaded model. `embed_fn` is the model's last-token
    hidden state (same signal as recall/consolidation). `route_fn` uses the TRAINED
    MoE gate aggregated over the chunk when sectors are attached (the aligned routing
    source); otherwise it returns None and the manager falls back to the STR
    classifier / a single slot, so a base-only model degrades gracefully."""
    import src.backend as backend
    ops = backend.current().ops
    ctx = context_len or model.cfg.context_len

    def embed_fn(tokens: List[int]):
        if not tokens:
            return None
        toks = ops.array(np.asarray([tokens[-ctx:]], dtype=np.int64))
        h = model(toks)
        return np.asarray(ops.to_numpy(h))[0, -1, :].astype(np.float32)

    def route_fn(tokens: List[int]):
        # Aggregate the trained gate's expert affinity over the chunk → dominant
        # sector(s). Only available once sectors + gate are attached.
        gate = getattr(model, "gate", None)
        if gate is None or not getattr(model, "_expert_ids", None) or not tokens:
            return None
        toks = ops.array(np.asarray([tokens[-ctx:]], dtype=np.int64))
        h = model(toks)
        _, _, logits = gate(h)                              # [1, S, n_experts]
        aff = np.asarray(ops.to_numpy(ops.softmax(logits, axis=-1)))[0].mean(axis=0)
        from src.routing.semantic_router import MIN_AFFINITY
        routed = [(model._expert_ids[i], float(a))
                  for i, a in enumerate(aff) if a >= MIN_AFFINITY]
        return sorted(routed, key=lambda x: -x[1]) or None

    # Decode evicted chunks back to text so consolidation has real content (not "").
    decode_fn = getattr(tokenizer, "decode", None) if tokenizer is not None else None

    return ContextManager(model.cfg.d_model, context_len=ctx, buffer=buffer,
                          embed_fn=embed_fn, route_fn=route_fn, decode_fn=decode_fn)
