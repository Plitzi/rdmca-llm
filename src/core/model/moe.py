"""
Mixture-of-Experts gating over the cognitive sectors (RDMCA §9 + MoE).

The cognitive sectors S1–S6 act as **experts**. Like the brain, not every expert
fires for every token: a learned gate scores the experts per token, keeps the
**top-k**, and only those contribute. As the sector pool grows (PGQ), routing
stays top-k, so the *active* compute stays bounded — modest machines keep running
the model even as it accumulates knowledge.

This module is backend-neutral (written against `src.core.backend.current()`):
  - `SectorGate`     — per-token top-k router (+ `grow_experts` for PGQ).
  - `expert_weights` — dense [..., n_experts] weight matrix, zero outside the
                       top-k, that the model multiplies each expert delta by.
  - `load_balance_loss` — standard MoE auxiliary loss to prevent expert collapse.

S7 (Behavioral/BCF) is intentionally NOT an expert here — it stays always-on and
isolated for the safety guarantee (see consolidation pipeline).
"""

from __future__ import annotations

import src.core.backend as backend

B = backend.current()
nn = B.nn
ops = B.ops


class SectorGate(nn.Module):
    """Per-token top-k router over `n_experts` sectors."""

    def __init__(self, d_model: int, n_experts: int, top_k: int = 2):
        super().__init__()
        self.n_experts = n_experts
        self._top_k_target = top_k  # configured k; restored as experts grow
        self.top_k = min(top_k, n_experts)
        # No bias: a freshly grown expert (zero row) starts at logit 0 and the
        # sector itself is zero-output at init (LoRA B=0), so growth is smooth.
        self.w = nn.Linear(d_model, n_experts, bias=False)

    def __call__(self, x):
        """x: [..., d_model] → (topk_idx [..., k], topk_w [..., k], logits [..., E]).
        `topk_w` is softmax over the selected top-k (sums to 1 per token)."""
        logits = self.w(x)  # [..., E]
        vals, idx = ops.topk(logits, self.top_k, axis=-1)  # [..., k]
        w = ops.softmax(vals, axis=-1)  # normalize over the k
        return idx, w, logits

    def grow_experts(self, delta: int) -> int:
        """Add `delta` experts (PGQ). New gate rows are zero (the new sector is
        zero-output at init, so routing is undisturbed). Returns new n_experts."""
        old = self.w.weight  # [E, D]
        in_dim = old.shape[1]
        new_w = ops.concatenate([old, ops.zeros((delta, in_dim))], axis=0)
        self.n_experts += delta
        # Restore k toward the configured target: if the gate was created with
        # fewer experts than top_k (so top_k got capped), growing must lift it
        # again — otherwise only the original capped number of experts is ever used.
        self.top_k = min(self._top_k_target, self.n_experts)
        self.w = nn.Linear(in_dim, self.n_experts, bias=False)
        self.w.weight = nn.Parameter(new_w)
        return self.n_experts


def expert_weights(topk_idx, topk_w, n_experts: int):
    """Scatter the top-k weights into a dense [..., n_experts] matrix (0 outside
    the top-k), so the model can do `sum_e w[..., e] * delta_e`. Avoids dynamic
    shapes → works on both backends."""
    ar = ops.arange(n_experts)  # [E]
    # eq[..., k, e] = 1 if topk_idx[..., k] == e
    eq = ops.astype(topk_idx[..., None] == ar, ops.float32)  # [..., k, E]
    return ops.sum(topk_w[..., None] * eq, axis=-2)  # [..., E]


def load_balance_loss(logits, topk_idx, n_experts: int):
    """Standard MoE load-balancing aux loss (GShard/Switch): encourages tokens to
    spread across experts. ≈ n_experts · Σ_e (f_e · P_e), with f_e = fraction of
    routed tokens and P_e = mean gate prob for expert e."""
    flat_logits = logits.reshape(-1, n_experts)  # [T, E]
    flat_idx = topk_idx.reshape(-1, topk_idx.shape[-1])  # [T, k]
    probs = ops.softmax(flat_logits, axis=-1)  # [T, E]
    ar = ops.arange(n_experts)
    eq = ops.astype(flat_idx[..., None] == ar, ops.float32)  # [T, k, E]
    dispatch = ops.sum(eq, axis=1)  # [T, E]  (0/1 per expert)
    f_e = ops.mean(dispatch, axis=0)  # [E]
    P_e = ops.mean(probs, axis=0)  # [E]
    return n_experts * ops.sum(f_e * P_e)
