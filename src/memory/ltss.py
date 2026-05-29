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

    def __init__(self, db_path: str = "data/ltss.db", emb_dim: int = 256):
        self.db_path = db_path
        self.emb_dim = emb_dim
        self._embeddings: List[np.ndarray] = []
        self._ids:        List[str]        = []
        self._conn: Optional[sqlite3.Connection] = None
        self._load()

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
        """Load all stored embeddings into the in-memory FAISS index."""
        try:
            import faiss
            rows = self._conn.execute(
                "SELECT id FROM ltss_nodes ORDER BY created_at"
            ).fetchall()
            # TODO: load embedding blobs and build faiss.IndexFlatIP index
            self._faiss_index = None   # placeholder until FAISS is wired
            self._ids = [r[0] for r in rows]
        except ImportError:
            self._faiss_index = None

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, node: LTSSNode) -> None:
        """Persist a new node and update the in-memory index."""
        now = time.time()
        node.created_at    = node.created_at or now
        node.last_accessed = node.last_accessed or now
        self._conn.execute(
            "INSERT OR REPLACE INTO ltss_nodes "
            "(id, content, modality, sector, created_at, last_accessed, access_count) "
            "VALUES (?,?,?,?,?,?,?)",
            (node.id, node.content, node.modality, node.sector,
             node.created_at, node.last_accessed, node.access_count),
        )
        self._conn.commit()
        self._embeddings.append(node.embedding)
        self._ids.append(node.id)

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
        Return top-k (node_id, cosine_similarity) pairs.
        Falls back to brute-force numpy if FAISS unavailable.
        """
        if not self._embeddings:
            return []
        mat = np.stack(self._embeddings, axis=0)   # [N, dim]
        q   = query / (np.linalg.norm(query) + 1e-8)
        mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
        sims = (mat @ q).tolist()
        ranked = sorted(zip(self._ids, sims), key=lambda x: -x[1])
        return ranked[:k]

    def max_cosine_similarity(self, query: np.ndarray) -> float:
        results = self.search(query, k=1)
        return results[0][1] if results else 0.0

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
