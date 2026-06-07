"""
RDMCA Foundational Transformer — T2 Edge (d_model=256)
Decoder-only, RoPE, RMSNorm pre-norm, SwiGLU FFN.
MRL (Matryoshka Representation Learning) loss over nested prefix dims.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_map


# Compute precision (training & inference). bf16 is the paper default; fp16 is
# fastest for quick tests (no loss-scaling — use for smoke runs, not gates).
PRECISION_DTYPES = {
    "fp32": mx.float32,
    "bf16": mx.bfloat16,
    "fp16": mx.float16,
}
_FLOAT_DTYPES = (mx.float32, mx.bfloat16, mx.float16)


def set_model_precision(model: nn.Module, precision: str) -> None:
    """Cast all float parameters of a module to the given precision in place.
    Integer params (none in this model) and non-arrays are left untouched."""
    dtype = PRECISION_DTYPES[precision]

    def _cast(p):
        if isinstance(p, mx.array) and p.dtype in _FLOAT_DTYPES:
            return p.astype(dtype)
        return p

    model.update(tree_map(_cast, model.parameters()))
    mx.eval(model.parameters())


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

def _rope_freqs(dim: int, theta: float, max_len: int) -> mx.array:
    """Precompute cosine/sine rotation matrices [max_len, dim//2, 2]."""
    half = dim // 2
    exponents = mx.arange(0, half, dtype=mx.float32) * 2.0 / dim
    inv_freq = 1.0 / (theta ** exponents)           # [half]
    positions = mx.arange(max_len, dtype=mx.float32) # [max_len]
    freqs = mx.outer(positions, inv_freq)             # [max_len, half]
    return freqs                                       # use cos/sin on the fly


def apply_rope(x: mx.array, freqs: mx.array) -> mx.array:
    """
    x: [batch, seq, n_heads, head_dim]
    freqs: [seq, head_dim // 2]
    """
    B, S, H, D = x.shape
    half = D // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    # cos/sin are computed in fp32; cast to x.dtype so low-precision (bf16/fp16)
    # activations are not silently promoted back to fp32 here.
    cos = mx.cos(freqs)[None, :S, None, :].astype(x.dtype)   # [1, S, 1, half]
    sin = mx.sin(freqs)[None, :S, None, :].astype(x.dtype)
    rotated = mx.concatenate([x1 * cos - x2 * sin,
                               x1 * sin + x2 * cos], axis=-1)
    return rotated


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        norm = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return x / norm * self.weight


class SwiGLU(nn.Module):
    """SwiGLU FFN: two projections, swish gate, output projection."""
    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.up_proj   = nn.Linear(d_model, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads  = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.scale    = self.head_dim ** -0.5

        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

        # Precompute RoPE frequencies
        freqs = _rope_freqs(self.head_dim, cfg.rope_theta, cfg.context_len)
        self._freqs = freqs  # [max_len, head_dim//2]

        # Sector wiring (set by RDMCAFoundational.attach_sectors).
        # layer_idx: this block's index;  _sector_delta: a callable
        # (layer_idx, proj, x) -> mx.array delta, or None when no sectors.
        # Both are plain Python attributes, ignored by the MLX param tree.
        self.layer_idx = 0
        self._sector_delta = None

    def __call__(self, x: mx.array,
                 mask: Optional[mx.array] = None) -> mx.array:
        B, S, D = x.shape
        H, Hd = self.n_heads, self.head_dim

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Inject active LoRA sector deltas (zero-output at init).
        if self._sector_delta is not None:
            q = q + self._sector_delta(self.layer_idx, "q", x)
            k = k + self._sector_delta(self.layer_idx, "k", x)
            v = v + self._sector_delta(self.layer_idx, "v", x)
        q = q.reshape(B, S, H, Hd)
        k = k.reshape(B, S, H, Hd)
        v = v.reshape(B, S, H, Hd)

        # Apply RoPE
        freqs = self._freqs[:S]
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        # [B, H, S, Hd]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale  # [B, H, S, S]

        # Causal mask (match attn dtype so bf16/fp16 stays low-precision here)
        causal = mx.triu(mx.full((S, S), -1e9), k=1).astype(attn.dtype)
        attn = attn + causal[None, None, :, :]

        if mask is not None:
            attn = attn + mask

        attn = mx.softmax(attn.astype(mx.float32), axis=-1).astype(x.dtype)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, S, D)
        proj = self.o_proj(out)
        if self._sector_delta is not None:
            proj = proj + self._sector_delta(self.layer_idx, "o", out)
        return proj


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1  = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = RMSNorm(cfg.d_model)
        self.ffn  = SwiGLU(cfg.d_model, cfg.ffn_dim)
        self.drop = nn.Dropout(cfg.dropout)

    def __call__(self, x: mx.array,
                 mask: Optional[mx.array] = None) -> mx.array:
        x = x + self.drop(self.attn(self.ln1(x), mask))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class RDMCAFoundational(nn.Module):
    """
    Foundational decoder-only transformer for RDMCA T2 Edge.
    Supports MRL (Matryoshka) training loss over nested prefix dimensions.
    After Stage 5 all parameters are frozen — LoRA sectors build on top.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg    = cfg
        self.embed  = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop   = nn.Dropout(cfg.dropout)
        self.blocks = [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        self.ln_f   = RMSNorm(cfg.d_model)

        # Single shared output projection. MRL prefixes reuse a prefix of this
        # weight matrix (W[:, :d]) instead of one full vocab head per dim —
        # avoids ~|mrl_dims| × vocab × d_model parameters of head bloat.
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Sector state (populated by attach_sectors). Plain attributes so the
        # MLX parameter tree ignores them; sector adapters live under .sectors.
        self.sectors = None          # {sector_id: SectorAdapter} once attached
        self._routing = _Routing()   # holds the currently-active sectors

    # ------------------------------------------------------------------
    # Output projection at an MRL prefix dimension
    # ------------------------------------------------------------------

    def head_at_dim(self, h: mx.array, d: int) -> mx.array:
        """Project the first d hidden dims to vocab using the shared head."""
        # nn.Linear weight is [vocab, d_model]; prefix columns map prefix dims.
        w_d = self.head.weight[:, :d]            # [vocab, d]
        return h[..., :d] @ w_d.T                # [..., vocab]

    # ------------------------------------------------------------------

    def __call__(self, tokens: mx.array,
                 mask: Optional[mx.array] = None) -> mx.array:
        """Forward pass. Returns full-dim hidden states [B, S, d_model]."""
        x = self.drop(self.embed(tokens))
        for block in self.blocks:
            x = block(x, mask)
        return self.ln_f(x)

    def logits(self, tokens: mx.array) -> mx.array:
        """Convenience: returns logits at the largest MRL dim."""
        h = self(tokens)
        return self.head_at_dim(h, self.cfg.mrl_dims[-1])

    def eval_ce(self, tokens: mx.array) -> mx.array:
        """
        Plain next-token cross-entropy at full dimension — used for validation
        perplexity (exp(eval_ce)). tokens: [B, S+1]. No MRL weighting.
        """
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        logits = self.head_at_dim(self(inputs), self.cfg.mrl_dims[-1])
        B, S, V = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B * S, V), targets.reshape(B * S), reduction="mean")

    # ------------------------------------------------------------------
    # MRL loss (multi-scale Matryoshka)
    # ------------------------------------------------------------------

    def mrl_loss(self, tokens: mx.array) -> mx.array:
        """
        tokens: [B, S+1] — input + target shifted by 1.
        Returns scalar loss (weighted sum across MRL dims).
        """
        inputs  = tokens[:, :-1]   # [B, S]
        targets = tokens[:, 1:]    # [B, S]

        h = self(inputs)           # [B, S, d_model]

        total = mx.array(0.0)
        weights = [1.0 / d for d in self.cfg.mrl_dims]
        w_sum   = sum(weights)

        for w, d in zip(weights, self.cfg.mrl_dims):
            logits_d = self.head_at_dim(h, d)              # [B, S, vocab]
            B, S, V  = logits_d.shape
            loss_d   = nn.losses.cross_entropy(
                logits_d.reshape(B * S, V),
                targets.reshape(B * S),
                reduction="mean",
            )
            total = total + (w / w_sum) * loss_d

        return total

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count_params(self, include_sectors: bool = True) -> int:
        from mlx.utils import tree_flatten
        params = self.parameters()
        total = sum(v.size for _, v in tree_flatten(params))
        if not include_sectors:
            sector_params = self.sector_param_count()
            total -= sector_params
        return total

    def sector_param_count(self) -> int:
        from mlx.utils import tree_flatten
        if not self.sectors:
            return 0
        return sum(
            v.size
            for adapter in self.sectors.values()
            for _, v in tree_flatten(adapter.parameters())
        )

    # ------------------------------------------------------------------
    # Sector integration (post foundational freeze)
    # ------------------------------------------------------------------

    def attach_sectors(self, sectors: dict) -> None:
        """
        Register LoRA sector adapters and wire them into every attention block.
        `sectors` is a {sector_id: SectorAdapter} mapping. Adapters are
        zero-output at init, so attaching them does not change model behavior
        until they are trained. Call after the foundational core is frozen.
        """
        self.sectors = sectors
        for i, block in enumerate(self.blocks):
            block.attn.layer_idx = i
            block.attn._sector_delta = self._compute_delta

    def add_sector(self, sector_id: int, rank: int = 4):
        """Instantiate and register a new adaptive sector at runtime (PGQ new
        sector creation, §10.7.4). It participates in inference immediately
        because _compute_delta reads self.sectors live."""
        from src.model.lora import SectorAdapter, LoRAConfig
        if self.sectors is None:
            self.attach_sectors({})
        adapter = SectorAdapter(LoRAConfig(
            d_model=self.cfg.d_model, n_layers=len(self.blocks),
            sector_id=sector_id, rank=rank))
        self.sectors[sector_id] = adapter
        return adapter

    def set_active_sectors(self, pairs) -> None:
        """Set which sectors contribute deltas: list of (sector_id, weight)."""
        self._routing.active = list(pairs) if pairs else []

    def _compute_delta(self, layer_idx: int, proj: str, x: mx.array):
        """Sum of active sector LoRA deltas for one projection in one layer."""
        active = self._routing.active
        if not active or not self.sectors:
            return 0.0
        total = None
        for sid, weight in active:
            adapter = self.sectors.get(sid)
            if adapter is None:
                continue
            d = adapter.delta(layer_idx, proj, x) * weight
            total = d if total is None else total + d
        return total if total is not None else 0.0


class _Routing:
    """Plain (non-Module) holder for the active-sector list, so the MLX
    parameter tree never tries to traverse it."""
    def __init__(self):
        self.active = []
