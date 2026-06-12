"""
Inference-time memory recall — the READ side of the memory system.

The base is no longer write-only: every surface (chat, agent, future API) embeds
the user's message, searches BOTH long-term stores, and injects the most relevant
memories into the prompt as a leading `<mem>…</mem>` block — the SAME framing the
Memory-management stage (stage 6) is trained on, so the frozen core consumes it
in-distribution.

Two stores, fused by cosine relevance (paper's working→episodic→LTSS view):
  • LTSS  — consolidated long-term semantic memory (promoted by the sleep cycle).
  • experience log — recent feedback turns (/ok, /fix) not yet consolidated.

Lazy and optional: with empty stores `recall()` returns [] and the preamble is
exactly as before — zero regression when there is nothing to remember.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

import src.backend as backend
from src.memory.ltss import LTSS
from src.memory.experience_log import load_experiences, EXPERIENCE_LOG


@dataclass
class Memory:
    text: str
    score: float
    source: str          # "ltss" | "experience"


class MemoryRecall:
    """Shared cross-surface memory retrieval. Construct once per session with the
    loaded model + tokenizer; call `recall(query)` each turn and feed
    `as_context(...)` into the prompt preamble."""

    def __init__(self, model, tokenizer, ltss: Optional[LTSS] = None,
                 experiences_path: str = EXPERIENCE_LOG,
                 emb_dim: Optional[int] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.emb_dim = emb_dim or model.cfg.d_model
        self.experiences_path = experiences_path
        self.ltss = ltss if ltss is not None else self._open_ltss()
        self._exp_cache: Optional[tuple] = None     # (mtime, [(text, emb)])

    def _open_ltss(self) -> Optional[LTSS]:
        try:
            return LTSS(emb_dim=self.emb_dim)
        except Exception:
            return None                              # no store yet → recall is a no-op

    # ------------------------------------------------------------------
    # Embedding — the model's last-token hidden state (same as consolidation,
    # see src/consolidation/pipeline.py: `self.model(toks)[:, -1, :]`).
    # ------------------------------------------------------------------
    def embed(self, text: str) -> Optional[np.ndarray]:
        if not text or not text.strip() or not getattr(self.tokenizer, "ready", False):
            return None
        ids = self.tokenizer.encode(text, add_bos=True, add_eos=False)
        if not ids:
            return None
        ids = ids[-self.model.cfg.context_len:]
        ops = backend.current().ops
        toks = ops.array(np.asarray([ids], dtype=np.int64))
        h = self.model(toks)                         # [1, S, d_model]
        emb = np.asarray(ops.to_numpy(h))[0, -1, :]  # last-token hidden state
        return emb.astype(np.float32)

    def _experience_embeddings(self) -> List[tuple]:
        """[(text, embedding)] for logged experiences, cached by file mtime so we
        embed each experience once, not every turn."""
        path = self.experiences_path
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return []
        if self._exp_cache and self._exp_cache[0] == mtime:
            return self._exp_cache[1]
        out = []
        for rec in load_experiences(path):
            text = (rec.get("text") or rec.get("response")
                    or rec.get("prompt") or "").strip()
            emb = self.embed(text)
            if emb is not None:
                out.append((text, emb))
        self._exp_cache = (mtime, out)
        return out

    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(a @ b / ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)))

    # ------------------------------------------------------------------
    def recall(self, query: str, k: int = 3, min_score: float = 0.25) -> List[Memory]:
        """Top-k most relevant memories for `query`, fused from LTSS + experiences,
        deduped by text and thresholded by cosine similarity."""
        q = self.embed(query)
        if q is None:
            return []
        cands: List[Memory] = []
        if self.ltss is not None and len(self.ltss):
            for node_id, sim in self.ltss.search(q, k=max(k * 2, k)):
                content = self.ltss.get_content(node_id)
                if content:
                    cands.append(Memory(content, float(sim), "ltss"))
        for text, emb in self._experience_embeddings():
            cands.append(Memory(text, self._cos(q, emb), "experience"))

        seen, fused = set(), []
        for m in sorted(cands, key=lambda x: -x.score):
            key = m.text.strip()
            if not key or key in seen or m.score < min_score:
                continue
            seen.add(key)
            fused.append(m)
            if len(fused) >= k:
                break
        return fused

    def as_context(self, mems: List[Memory]) -> str:
        """Format recalled memories as a `<mem>…</mem>` block for the prompt front
        (empty string when there is nothing to inject)."""
        if not mems:
            return ""
        from src import agent
        body = "\n".join(f"- {m.text.strip()}" for m in mems)
        return agent.memory_block(body)
