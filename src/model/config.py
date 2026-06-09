"""
Backend-neutral model configuration dataclasses.

These are pure data (no tensor framework imported), so configs can be built and
inspected without selecting a compute backend — important because the backend
must be chosen *before* the backend-bound model modules are imported.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 4
    ffn_dim: int = 1024
    context_len: int = 2048
    vocab_size: int = 32000
    mrl_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    dropout: float = 0.1
    rope_theta: float = 10000.0


@dataclass
class LoRAConfig:
    d_model: int
    n_layers: int
    sector_id: int
    rank: int
    alpha: float = 1.0   # scaling: weight = alpha / rank
