"""
RDMCA Foundational Transformer — T2 Edge (d_model=256)
Decoder-only, RoPE, RMSNorm pre-norm, SwiGLU FFN.
MRL (Matryoshka Representation Learning) loss over nested prefix dims.

Uses MLX fast kernels for attention/RoPE/RMSNorm:
  - mx.fast.scaled_dot_product_attention — Flash-Attention-style, O(n) memory
    (never materializes the [S, S] score matrix, so 2048-token contexts train
    without tripping the macOS Metal command-buffer watchdog)
  - nn.RoPE / mx.fast.rms_norm — fused kernels, fewer intermediate buffers
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn


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
    dtype: str = "bfloat16"   # bfloat16 (training) | float32 (debug)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """RMSNorm using the fused mx.fast.rms_norm kernel."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


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

        # Fused RoPE kernel (mx.fast.rope under the hood)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=cfg.rope_theta)

    def __call__(self, x: mx.array,
                 mask: Optional[mx.array] = None) -> mx.array:
        B, S, D = x.shape
        H, Hd = self.n_heads, self.head_dim

        # [B, H, S, Hd]
        q = self.q_proj(x).reshape(B, S, H, Hd).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, S, H, Hd).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, S, H, Hd).transpose(0, 2, 1, 3)

        q = self.rope(q)
        k = self.rope(k)

        # Flash-Attention-style kernel — O(n) memory, never builds [S, S].
        # mask="causal" applies the autoregressive mask internally.
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale,
            mask=mask if mask is not None else "causal",
        )

        out = out.transpose(0, 2, 1, 3).reshape(B, S, D)
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

        # Cast parameters to the configured compute dtype (bf16 by default).
        # bf16 is ~20x faster and ~4x smaller than fp32 on Apple Silicon,
        # and matches the "BF16 training" spec in docs/reference/architecture.md.
        if cfg.dtype != "float32":
            self.set_dtype(getattr(mx, cfg.dtype))

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
            # Upcast logits to fp32 for the softmax/cross-entropy reduction —
            # keeps the heavy matmuls in bf16 but the loss numerically stable.
            loss_d   = nn.losses.cross_entropy(
                logits_d.reshape(B * S, V).astype(mx.float32),
                targets.reshape(B * S),
                reduction="mean",
            )
            total = total + (w / w_sum) * loss_d

        return total

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count_params(self) -> int:
        from mlx.utils import tree_flatten
        return sum(v.size for _, v in tree_flatten(self.parameters()))

    def freeze_all(self):
        """
        Permanently freeze the foundational core after Stage 5 (paper §6.5.1).
        Uses nn.Module.freeze() so frozen params are excluded from gradient
        updates and from trainable_parameters() — LoRA sectors build on top.
        """
        self.freeze()   # nn.Module.freeze — recurse=True by default
