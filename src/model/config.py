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
    # Grouped-Query Attention: number of KV heads (each shared by n_heads/n_kv_heads
    # query heads). None → n_kv_heads = n_heads (plain multi-head attention). Set
    # below n_heads to shrink the KV cache n_heads/n_kv_heads× — the payoff grows
    # with scale, so it is a per-level knob (L5 12→4 = 3× smaller cache).
    n_kv_heads: int | None = None
    ffn_dim: int = 1024
    context_len: int = 2048
    vocab_size: int = 32000
    mrl_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    dropout: float = 0.1
    rope_theta: float = 10000.0

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        # MRL prefixes must be ascending, unique and ≤ d_model — head_at_dim slices
        # the tied embed.weight[:, :d], so a d > d_model would silently use the full matrix
        # (breaking the Matryoshka premise) and a wrong order would mis-weight the loss.
        if self.mrl_dims != sorted(set(self.mrl_dims)):
            raise ValueError(f"mrl_dims must be ascending and unique, got {self.mrl_dims}")
        if self.mrl_dims and self.mrl_dims[-1] > self.d_model:
            raise ValueError(f"mrl_dims max ({self.mrl_dims[-1]}) exceeds d_model ({self.d_model})")
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) not divisible by n_heads ({self.n_heads})")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads ({self.n_heads}) not divisible by "
                             f"n_kv_heads ({self.n_kv_heads})")

    @property
    def kv_dim(self) -> int:
        """Width of the K/V projections (n_kv_heads·head_dim); = d_model for MHA."""
        return self.n_kv_heads * (self.d_model // self.n_heads)


@dataclass
class LoRAConfig:
    d_model: int
    n_layers: int
    sector_id: int
    rank: int
    alpha: float = 1.0   # scaling: weight = alpha / rank
    # K/V LoRA deltas must match the (GQA-narrowed) K/V projection width
    # (n_kv_heads·head_dim). None → kv_dim = d_model (plain multi-head attention).
    kv_dim: int | None = None

    def __post_init__(self):
        if self.kv_dim is None:
            self.kv_dim = self.d_model
