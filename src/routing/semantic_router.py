"""
Semantic Token Router (STR) — RDMCA §12
Eliminates fixed context window constraints by routing token chunks to
sector-specific context slots based on domain affinity.

Pipeline:
  1. Segment incoming token sequence into semantic chunks
  2. Compute sector affinity vector for each chunk via lightweight classifier
  3. Route chunk to sector context slot s* = argmax P(s|chunk)
  4. Multiple sectors can activate simultaneously for mixed-domain content

Chunk segmentation:
  Text:  sentence boundaries (NLTK) or fixed windows of 128 tokens
  Image: one chunk per image (196 tokens at 224×224)
  Audio: fixed windows of 50 tokens/sec

Context slot architecture:
  Each sector s maintains its own KV-cache context window (2048 tokens).
  The foundational backbone sees the union of all active sector contexts.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import src.backend as backend

B = backend.current()
nn = B.nn
ops = B.ops


NUM_SECTORS    = 7
CHUNK_SIZE     = 128    # tokens per text chunk
MIN_AFFINITY   = 0.15   # minimum affinity to activate a sector


@dataclass
class Chunk:
    tokens: List[int]
    modality: str = "text"
    embedding: Optional[np.ndarray] = None


class STRClassifier(nn.Module):
    """
    Lightweight sector affinity classifier.
    Input: mean-pooled chunk embedding (d_model dims).
    Output: softmax distribution over NUM_SECTORS.
    """

    def __init__(self, d_model: int, n_sectors: int = NUM_SECTORS):
        super().__init__()
        self.fc   = nn.Linear(d_model, n_sectors, bias=True)

    def __call__(self, emb):
        """emb: [..., d_model]  →  [..., n_sectors] probabilities"""
        return ops.softmax(self.fc(emb), axis=-1)


class SemanticTokenRouter:
    """
    Routes token chunks to sector context slots.
    Wraps a trained STRClassifier and maintains per-sector context queues.
    """

    def __init__(self, d_model: int, context_len: int = 2048):
        self.classifier  = STRClassifier(d_model)
        self.context_len = context_len
        # Per-sector token queues (ring buffers)
        self._contexts: Dict[int, List[int]] = {s: [] for s in range(1, NUM_SECTORS + 1)}

    def segment(self, tokens: List[int]) -> List[Chunk]:
        """Split token sequence into fixed-size chunks."""
        return [
            Chunk(tokens=tokens[i:i + CHUNK_SIZE])
            for i in range(0, len(tokens), CHUNK_SIZE)
        ]

    def route(self, chunk: Chunk, emb) -> List[Tuple[int, float]]:
        """
        Returns list of (sector_id, affinity) for all sectors with
        affinity ≥ MIN_AFFINITY, sorted descending.
        """
        probs = ops.to_numpy(self.classifier(emb)).reshape(-1).tolist()   # [n_sectors]
        routed = [
            (sid + 1, float(p))
            for sid, p in enumerate(probs)
            if p >= MIN_AFFINITY
        ]
        return sorted(routed, key=lambda x: -x[1])

    def add_to_context(self, sector_id: int, tokens: List[int]) -> None:
        """Append tokens to a sector's context slot (ring buffer)."""
        buf = self._contexts[sector_id]
        buf.extend(tokens)
        if len(buf) > self.context_len:
            del buf[:len(buf) - self.context_len]

    def get_context(self, sector_id: int) -> List[int]:
        return list(self._contexts[sector_id])

    def clear_contexts(self) -> None:
        for sid in self._contexts:
            self._contexts[sid].clear()
