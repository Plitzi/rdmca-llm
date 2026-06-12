"""
Long-Term Semantic Store (LTSS) — RDMCA §10.2 / Implementation Guide §2.1
Persistent memory: SQLite for node metadata + FAISS for vector similarity.
Stores abstracted semantic concepts (nodes) with causal/associative edges.
Target retrieval latency: < 10ms for top-5 among 100k nodes.
"""
from __future__ import annotations
import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import numpy as np


SCHEMA_NODES = """
CREATE TABLE IF NOT EXISTS ltss_nodes (
    id           TEXT PRIMARY KEY,
    embedding    BLOB,          -- float32 vector bytes (emb_dim)
    content      TEXT,
    modality     TEXT,
    sector       TEXT,
    created_at   REAL,
    last_accessed REAL,
    access_count INTEGER DEFAULT 0
);
"""

SCHEMA_EDGES = """
CREATE TABLE IF NOT EXISTS ltss_edges (
    src_id   TEXT,
    dst_id   TEXT,
    relation TEXT,   -- causal | associative | contradicts
    weight   REAL,
    PRIMARY KEY (src_id, dst_id, relation)
);
"""


@dataclass
class LTSSNode:
    id: str
    embedding: np.ndarray
    content: str
    modality: str = "text"
    sector: str = ""
    created_at: float = 0.0
    last_accessed: float = 0.0
    access_count: int = 0


class LTSS:
    """
    Long-Term Semantic Store.
    db_path:    path to SQLite file
    emb_dim:    embedding dimension (must match foundational model d_model)
    """

    def __init__(self, db_path: str = "data/runtime/ltss.db", emb_dim: int = 256):
        self.db_path = db_path
        self.emb_dim = emb_dim
        self._embeddings: List[np.ndarray] = []
        self._ids:        List[str]        = []
        self._conn: Optional[sqlite3.Connection] = None
        # Optional FAISS ANN index: search O(N)→O(log N) at scale. Trivial gain at
        # small N (numpy is already sub-ms), decisive past ~100k nodes — present from
        # the start so the progression is visible. Falls back to brute-force numpy
        # otherwise (the index stays None).
        #
        # OPT-IN via RDMCA_FAISS=1. faiss-cpu and PyTorch each link their own libomp,
        # which collides on macOS (OMP Error #15 → possible crash / wrong results).
        # On Linux/CUDA — where L4-L5 run at the scale where FAISS actually matters —
        # they share libgomp and coexist fine. So FAISS is off by default (numpy is
        # plenty for dev-scale stores) and enabled explicitly when deploying at scale.
        self._faiss = None
        if os.environ.get("RDMCA_FAISS", "").lower() in ("1", "true", "yes", "on"):
            try:
                import faiss
                self._faiss = faiss
            except Exception:
                self._faiss = None
        self._faiss_index = None
        self._load()

    def _new_faiss_index(self):
        """A fresh HNSW (inner-product) index, or None if faiss is unavailable.
        Vectors are L2-normalized on add/search, so inner product == cosine."""
        if self._faiss is None:
            return None
        idx = self._faiss.IndexHNSWFlat(self.emb_dim, 32, self._faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efSearch = 64
        return idx

    def _faiss_add(self, emb: np.ndarray) -> None:
        if self._faiss_index is None:
            return
        v = np.asarray(emb, dtype=np.float32).reshape(1, -1).copy()
        self._faiss.normalize_L2(v)
        self._faiss_index.add(v)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _load(self) -> None:
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute(SCHEMA_NODES)
        self._conn.execute(SCHEMA_EDGES)
        self._conn.commit()
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Load stored ids + embeddings on startup (persistence across restarts) and
        (re)build the FAISS index from them. Search uses FAISS when available, else
        brute-force numpy — both kept in sync here."""
        rows = self._conn.execute(
            "SELECT id, embedding FROM ltss_nodes ORDER BY created_at"
        ).fetchall()
        self._ids = []
        self._embeddings = []
        self._faiss_index = self._new_faiss_index()
        for node_id, blob in rows:
            self._ids.append(node_id)
            if blob is not None:
                emb = np.frombuffer(blob, dtype=np.float32).copy()
                self._embeddings.append(emb)
                self._faiss_add(emb)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, node: LTSSNode) -> None:
        """Persist a new node (embedding included) and update the in-memory index."""
        now = time.time()
        node.created_at    = node.created_at or now
        node.last_accessed = node.last_accessed or now
        emb = np.asarray(node.embedding, dtype=np.float32)
        self._conn.execute(
            "INSERT OR REPLACE INTO ltss_nodes "
            "(id, embedding, content, modality, sector, created_at, last_accessed, access_count) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (node.id, emb.tobytes(), node.content, node.modality, node.sector,
             node.created_at, node.last_accessed, node.access_count),
        )
        self._conn.commit()
        self._embeddings.append(emb)
        self._ids.append(node.id)
        self._faiss_add(emb)

    def add_edge(self, src_id: str, dst_id: str,
                 relation: str, weight: float = 1.0) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO ltss_edges (src_id,dst_id,relation,weight) "
            "VALUES (?,?,?,?)", (src_id, dst_id, relation, weight)
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(self, query: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """
        Return top-k (node_id, cosine_similarity) pairs. Uses the FAISS index when
        available (O(log N)); falls back to brute-force numpy (O(N), exact).
        """
        if not self._embeddings:
            return []
        # FAISS path — only when the index covers every id (all nodes embedded), so
        # the returned indices map 1:1 onto self._ids.
        if (self._faiss_index is not None
                and self._faiss_index.ntotal == len(self._ids)):
            q = np.asarray(query, dtype=np.float32).reshape(1, -1).copy()
            self._faiss.normalize_L2(q)
            sims, idx = self._faiss_index.search(q, min(k, len(self._ids)))
            return [(self._ids[i], float(s))
                    for s, i in zip(sims[0], idx[0]) if 0 <= i < len(self._ids)]
        # numpy fallback (exact brute force)
        mat = np.stack(self._embeddings, axis=0)   # [N, dim]
        q   = query / (np.linalg.norm(query) + 1e-8)
        mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
        sims = (mat @ q).tolist()
        ranked = sorted(zip(self._ids, sims), key=lambda x: -x[1])
        return ranked[:k]

    def max_cosine_similarity(self, query: np.ndarray) -> float:
        results = self.search(query, k=1)
        return results[0][1] if results else 0.0

    def get_content(self, node_id: str) -> Optional[str]:
        """Return the stored content for a node id (None if unknown). Used by the
        inference-time recall (MemoryRecall) to turn a search hit into text."""
        row = self._conn.execute(
            "SELECT content FROM ltss_nodes WHERE id = ?", (node_id,)).fetchone()
        return row[0] if row else None

    @property
    def global_centroid(self) -> Optional[np.ndarray]:
        if not self._embeddings:
            return None
        return np.mean(self._embeddings, axis=0)

    @property
    def global_std(self) -> Optional[np.ndarray]:
        if len(self._embeddings) < 2:
            return None
        return np.std(self._embeddings, axis=0)

    def __len__(self) -> int:
        return len(self._ids)
