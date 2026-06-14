"""Transformer building blocks — RoPE, RMSNorm, SwiGLU FFN, GQA causal attention,
and the residual TransformerBlock (with a KV-cached forward for generation).

Backend-neutral: written once against the active backend facade
(`src.core.backend.current()`). Select the backend BEFORE importing this module. Split
out of transformer.py so that file holds just the model that composes these.
"""

from __future__ import annotations

import numpy as np

import src.core.backend as backend
from src.core.model.config import ModelConfig

B = backend.current()
nn = B.nn
ops = B.ops


# ---------------------------------------------------------------------------
# RoPE — frequency tables precomputed on the host (numpy), so they never enter
# the backend parameter tree; converted to backend tensors on the fly per pass.
# ---------------------------------------------------------------------------


def _rope_tables(dim: int, theta: float, max_len: int):
    """Precompute cos/sin rotation tables [max_len, dim//2] as numpy arrays."""
    half = dim // 2
    exponents = np.arange(0, half, dtype=np.float32) * 2.0 / dim
    inv_freq = 1.0 / (theta**exponents)  # [half]
    positions = np.arange(max_len, dtype=np.float32)  # [max_len]
    freqs = np.outer(positions, inv_freq)  # [max_len, half]
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
    cos = ops.astype(cos[None, :, None, :], x.dtype)  # [1, S, 1, half]
    sin = ops.astype(sin[None, :, None, :], x.dtype)
    return ops.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


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
        self.up_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, d_model, bias=False)

    def __call__(self, x):
        return self.down_proj(ops.silu(self.gate_proj(x)) * self.up_proj(x))


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads or cfg.n_heads  # GQA: KV heads (≤ n_heads)
        self.head_dim = cfg.d_model // cfg.n_heads
        self.kv_dim = self.n_kv_heads * self.head_dim  # K/V projection width
        # RoPE splits head_dim into half cos / half sin pairs; an odd head_dim would
        # make x2 one element wider than cos/sin and break the rotation broadcast.
        assert self.head_dim % 2 == 0, f"head_dim must be even for RoPE, got {self.head_dim}"
        self.scale = self.head_dim**-0.5

        # GQA: Q is full width (n_heads·head_dim = d_model); K/V are narrower
        # (n_kv_heads·head_dim), so the KV cache shrinks n_heads/n_kv_heads×. The
        # KV heads are shared across query-head groups by the fused SDPA kernel.
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # RoPE tables (numpy host arrays — invisible to the param tree). Their
        # backend-tensor form is built lazily and cached on first forward (L9).
        self._rope_cos, self._rope_sin = _rope_tables(
            self.head_dim, cfg.rope_theta, cfg.context_len
        )
        self._rope_cos_t = None
        self._rope_sin_t = None

        # Sector wiring (set by RDMCAFoundational.attach_sectors). Plain Python
        # attributes, ignored by both backends' parameter trees.
        self.layer_idx = 0
        self._sector_delta = None  # model._compute_delta(layer, proj, x, route)
        self._sector_route = None  # model._route(x) -> per-token expert weights

    def __call__(self, x, mask=None, cache=None, pos_offset=0, return_kv=False):
        """Attention over `x`.

        Training/full forward (default): `cache=None, pos_offset=0` → standard
        causal attention over the S tokens, returns `proj`. This path is byte-for-
        byte the original behaviour.

        Cached generation: pass `cache=(past_k, past_v)` (layout [B, H, T_past, Hd])
        and `pos_offset` = number of already-processed tokens; q/k/v are computed
        only for the NEW tokens `x`, concatenated onto the cache, and RoPE is applied
        at the ABSOLUTE positions `pos_offset .. pos_offset+S`. With `return_kv=True`
        the updated `(k, v)` is returned alongside `proj` so the caller can thread it
        to the next step. For single-token decode (S==1) no causal mask is needed —
        the new query legitimately attends to every cached key plus itself."""
        Bsz, S, D = x.shape
        H, Hkv, Hd = self.n_heads, self.n_kv_heads, self.head_dim

        # Route once per block (per-token top-k over experts); reuse for q,k,v,o.
        route = self._sector_route(x) if self._sector_route is not None else None

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Inject active LoRA sector deltas (zero-output at init). K/V deltas are
        # GQA-width (kv_dim), matching the narrower K/V projections above.
        if self._sector_delta is not None:
            q = q + self._sector_delta(self.layer_idx, "q", x, route)
            k = k + self._sector_delta(self.layer_idx, "k", x, route)
            v = v + self._sector_delta(self.layer_idx, "v", x, route)
        q = q.reshape(Bsz, S, H, Hd)
        k = k.reshape(Bsz, S, Hkv, Hd)  # GQA: fewer KV heads than query heads
        v = v.reshape(Bsz, S, Hkv, Hd)

        # Apply RoPE at absolute positions [pos_offset, pos_offset+S). Convert the
        # host tables to backend tensors ONCE (cached) and slice per pass — re-
        # creating them from numpy every forward churned memory. The offset is what
        # makes the KV cache correct: a decoded token must rotate at its true
        # position, not at index 0 (the bug in the report's sketch used cos[-1:]).
        if self._rope_cos_t is None:
            self._rope_cos_t = ops.array(self._rope_cos)
            self._rope_sin_t = ops.array(self._rope_sin)
        cos = self._rope_cos_t[pos_offset : pos_offset + S]
        sin = self._rope_sin_t[pos_offset : pos_offset + S]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        q = ops.transpose(q, (0, 2, 1, 3))  # [B, H,   S, Hd]
        k = ops.transpose(k, (0, 2, 1, 3))  # [B, Hkv, S, Hd]
        v = ops.transpose(v, (0, 2, 1, 3))

        # KV cache: prepend the past keys/values (seq axis = 2) so the new queries
        # attend over the whole history without reprocessing it. Cache holds Hkv
        # heads — that is the GQA cache-size saving (n_heads/n_kv_heads× smaller).
        if cache is not None:
            past_k, past_v = cache
            k = ops.concatenate([past_k, k], axis=2)
            v = ops.concatenate([past_v, v], axis=2)
        new_kv = (k, v)

        # Fused scaled-dot-product attention (Flash / mem-efficient kernel). The
        # kernel broadcasts the Hkv KV heads across the H query heads (GQA) and
        # applies the causal mask internally. Causal only for multi-token prefill /
        # training (cache is None, S>1); single-token decode attends to all cached
        # keys, so no mask. Attention-weight dropout is dropped (residual + FFN
        # dropout remain) — modern small LLMs train fine with 0 attention dropout.
        out = ops.sdpa(
            q, k, v, scale=self.scale, is_causal=(cache is None and S > 1), attn_mask=mask
        )
        out = ops.transpose(out, (0, 2, 1, 3)).reshape(Bsz, S, D)
        proj = self.o_proj(out)
        if self._sector_delta is not None:
            proj = proj + self._sector_delta(self.layer_idx, "o", out, route)
        return (proj, new_kv) if return_kv else proj


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_dim)
        self.drop = nn.Dropout(cfg.dropout)

    def __call__(self, x, mask=None):
        x = x + self.drop(self.attn(self.ln1(x), mask))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x

    def forward_cached(self, x, cache=None, pos_offset=0):
        """Cached-generation forward: same residual structure as __call__, but the
        attention reads/writes the KV cache. Returns (x, (k, v)). Dropout is a no-op
        in eval, where this path runs."""
        a, kv = self.attn(self.ln1(x), cache=cache, pos_offset=pos_offset, return_kv=True)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x, kv
