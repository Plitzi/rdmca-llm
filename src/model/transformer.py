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
    cos = mx.cos(freqs)[None, :S, None, :]   # [1, S, 1, half]
    sin = mx.sin(freqs)[None, :S, None, :]
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

    def __call__(self, x: mx.array,
                 mask: Optional[mx.array] = None) -> mx.array:
        B, S, D = x.shape
        H, Hd = self.n_heads, self.head_dim

        q = self.q_proj(x).reshape(B, S, H, Hd)
        k = self.k_proj(x).reshape(B, S, H, Hd)
        v = self.v_proj(x).reshape(B, S, H, Hd)

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

        # Causal mask
        causal = mx.triu(mx.full((S, S), -1e9), k=1)
        attn = attn + causal[None, None, :, :]

        if mask is not None:
            attn = attn + mask

        attn = mx.softmax(attn.astype(mx.float32), axis=-1).astype(x.dtype)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, S, D)
        return self.o_proj(out)


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

        # Separate output heads per MRL dim — projects prefix to vocab
        self.heads: List[nn.Linear] = []
        for d in cfg.mrl_dims:
            self.heads.append(nn.Linear(d, cfg.vocab_size, bias=False))

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
        d = self.cfg.mrl_dims[-1]
        return self.heads[-1](h[..., :d])

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

        for w, d, head in zip(weights, self.cfg.mrl_dims, self.heads):
            logits_d = head(h[..., :d])                    # [B, S, vocab]
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

    def count_params(self) -> int:
        def _count(tree):
            if isinstance(tree, mx.array):
                return tree.size
            if isinstance(tree, dict):
                return sum(_count(v) for v in tree.values())
            if isinstance(tree, list):
                return sum(_count(v) for v in tree)
            return 0
        return _count(self.parameters())

    def freeze(self):
        """Mark all parameters as non-trainable (post Stage-5 freeze)."""
        def _freeze(p):
            if isinstance(p, mx.array):
                p.stop_gradient = True
        for p in self.parameters():
            _freeze(p)
