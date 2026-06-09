"""
RDMCA Foundational Transformer — T2 Edge (d_model=256)
Decoder-only, RoPE, RMSNorm pre-norm, SwiGLU FFN.
MRL (Matryoshka Representation Learning) loss over nested prefix dims.

Backend-neutral: written once against the active backend facade
(`src.backend.current()`), so the same code runs on MLX or PyTorch. Select the
backend (via `src.backend.select`) BEFORE importing this module.
"""
from __future__ import annotations
from typing import List, Optional

import numpy as np

import src.backend as backend
from src.model.config import ModelConfig   # re-exported below for compatibility

B = backend.current()
nn = B.nn
ops = B.ops


# Compatibility shim: precision is now owned by the engine. Older call sites
# import `set_model_precision` from here.
def set_model_precision(model, precision: str) -> None:
    """Cast all float parameters of a module to the given precision in place."""
    B.engine.set_precision(model, precision)


# ---------------------------------------------------------------------------
# RoPE — frequency tables precomputed on the host (numpy), so they never enter
# the backend parameter tree; converted to backend tensors on the fly per pass.
# ---------------------------------------------------------------------------

def _rope_tables(dim: int, theta: float, max_len: int):
    """Precompute cos/sin rotation tables [max_len, dim//2] as numpy arrays."""
    half = dim // 2
    exponents = np.arange(0, half, dtype=np.float32) * 2.0 / dim
    inv_freq = 1.0 / (theta ** exponents)                 # [half]
    positions = np.arange(max_len, dtype=np.float32)      # [max_len]
    freqs = np.outer(positions, inv_freq)                 # [max_len, half]
    return np.cos(freqs), np.sin(freqs)


def apply_rope(x, cos, sin):
    """
    x:   [batch, seq, n_heads, head_dim]
    cos/sin: [seq, head_dim // 2]  (backend tensors)
    """
    D = x.shape[-1]
    half = D // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    # cos/sin precomputed in fp32; cast to x.dtype so bf16/fp16 stays low-prec.
    cos = ops.astype(cos[None, :, None, :], x.dtype)      # [1, S, 1, half]
    sin = ops.astype(sin[None, :, None, :], x.dtype)
    return ops.concatenate([x1 * cos - x2 * sin,
                            x1 * sin + x2 * cos], axis=-1)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(ops.ones((dim,)))

    def __call__(self, x):
        norm = ops.sqrt(ops.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return x / norm * self.weight


class SwiGLU(nn.Module):
    """SwiGLU FFN: two projections, swish gate, output projection."""
    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.up_proj   = nn.Linear(d_model, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, d_model, bias=False)

    def __call__(self, x):
        return self.down_proj(ops.silu(self.gate_proj(x)) * self.up_proj(x))


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

        # RoPE tables (numpy host arrays — invisible to the param tree).
        self._rope_cos, self._rope_sin = _rope_tables(
            self.head_dim, cfg.rope_theta, cfg.context_len)

        # Sector wiring (set by RDMCAFoundational.attach_sectors). Plain Python
        # attributes, ignored by both backends' parameter trees.
        self.layer_idx = 0
        self._sector_delta = None

    def __call__(self, x, mask=None):
        Bsz, S, D = x.shape
        H, Hd = self.n_heads, self.head_dim

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Inject active LoRA sector deltas (zero-output at init).
        if self._sector_delta is not None:
            q = q + self._sector_delta(self.layer_idx, "q", x)
            k = k + self._sector_delta(self.layer_idx, "k", x)
            v = v + self._sector_delta(self.layer_idx, "v", x)
        q = q.reshape(Bsz, S, H, Hd)
        k = k.reshape(Bsz, S, H, Hd)
        v = v.reshape(Bsz, S, H, Hd)

        # Apply RoPE (convert host tables to backend tensors once per pass).
        cos = ops.array(self._rope_cos[:S])
        sin = ops.array(self._rope_sin[:S])
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # [B, H, S, Hd]
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        # Scaled dot-product attention
        attn = (q @ ops.transpose(k, (0, 1, 3, 2))) * self.scale  # [B, H, S, S]

        # Causal mask (match attn dtype so bf16/fp16 stays low-precision here)
        causal = ops.astype(ops.triu(ops.full((S, S), -1e9), k=1), attn.dtype)
        attn = attn + causal[None, None, :, :]

        if mask is not None:
            attn = attn + mask

        attn = ops.astype(ops.softmax(ops.astype(attn, ops.float32), axis=-1), x.dtype)
        attn = self.dropout(attn)

        out = ops.transpose((attn @ v), (0, 2, 1, 3)).reshape(Bsz, S, D)
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

    def __call__(self, x, mask=None):
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
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f   = RMSNorm(cfg.d_model)

        # Single shared output projection. MRL prefixes reuse a prefix of this
        # weight matrix (W[:, :d]) instead of one full vocab head per dim.
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Sector state (populated by attach_sectors). int-keyed dict for logic;
        # params are registered for the backend via engine.register_submodules.
        self.sectors = None
        self._routing = _Routing()

    # ------------------------------------------------------------------
    # Output projection at an MRL prefix dimension
    # ------------------------------------------------------------------

    def head_at_dim(self, h, d: int):
        """Project the first d hidden dims to vocab using the shared head."""
        w_d = self.head.weight[:, :d]            # [vocab, d]
        return h[..., :d] @ w_d.T                # [..., vocab]

    # ------------------------------------------------------------------

    def __call__(self, tokens, mask=None):
        """Forward pass. Returns full-dim hidden states [B, S, d_model]."""
        x = self.drop(self.embed(tokens))
        for block in self.blocks:
            x = block(x, mask)
        return self.ln_f(x)

    def logits(self, tokens):
        """Convenience: returns logits at the largest MRL dim."""
        h = self(tokens)
        return self.head_at_dim(h, self.cfg.mrl_dims[-1])

    def eval_ce(self, tokens):
        """
        Plain next-token cross-entropy at full dimension — used for validation
        perplexity (exp(eval_ce)). tokens: [B, S+1]. No MRL weighting.
        """
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        logits = self.head_at_dim(self(inputs), self.cfg.mrl_dims[-1])
        Bsz, S, V = logits.shape
        return ops.cross_entropy(
            logits.reshape(Bsz * S, V), targets.reshape(Bsz * S), reduction="mean")

    # ------------------------------------------------------------------
    # MRL loss (multi-scale Matryoshka)
    # ------------------------------------------------------------------

    def mrl_loss(self, tokens):
        """
        tokens: [B, S+1] — input + target shifted by 1.
        Returns scalar loss (weighted sum across MRL dims).
        """
        inputs  = tokens[:, :-1]   # [B, S]
        targets = tokens[:, 1:]    # [B, S]

        h = self(inputs)           # [B, S, d_model]

        total = ops.array(0.0)
        weights = [1.0 / d for d in self.cfg.mrl_dims]
        w_sum   = sum(weights)

        for w, d in zip(weights, self.cfg.mrl_dims):
            logits_d = self.head_at_dim(h, d)              # [B, S, vocab]
            Bsz, S, V = logits_d.shape
            loss_d   = ops.cross_entropy(
                logits_d.reshape(Bsz * S, V),
                targets.reshape(Bsz * S),
                reduction="mean",
            )
            total = total + (w / w_sum) * loss_d

        return total

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count_params(self, include_sectors: bool = True) -> int:
        total = B.engine.param_count(self)
        if not include_sectors:
            total -= self.sector_param_count()
        return total

    def sector_param_count(self) -> int:
        if not self.sectors:
            return 0
        return sum(B.engine.param_count(adapter) for adapter in self.sectors.values())

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
        B.engine.register_submodules(self, "_sector_store", list(sectors.values()))
        for i, block in enumerate(self.blocks):
            block.attn.layer_idx = i
            block.attn._sector_delta = self._compute_delta

    def add_sector(self, sector_id: int, rank: int = 4):
        """Instantiate and register a new adaptive sector at runtime (PGQ new
        sector creation, §10.7.4). It participates in inference immediately
        because _compute_delta reads self.sectors live."""
        from src.model.lora import SectorAdapter
        from src.model.config import LoRAConfig
        if self.sectors is None:
            self.attach_sectors({})
        adapter = SectorAdapter(LoRAConfig(
            d_model=self.cfg.d_model, n_layers=len(self.blocks),
            sector_id=sector_id, rank=rank))
        self.sectors[sector_id] = adapter
        # Re-register so the new adapter's params are tracked by the backend.
        B.engine.register_submodules(self, "_sector_store", list(self.sectors.values()))
        return adapter

    def set_active_sectors(self, pairs) -> None:
        """Set which sectors contribute deltas: list of (sector_id, weight)."""
        self._routing.active = list(pairs) if pairs else []

    def _compute_delta(self, layer_idx: int, proj: str, x):
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
    """Plain (non-Module) holder for the active-sector list, so the backend
    parameter tree never tries to traverse it."""
    def __init__(self):
        self.active = []
