"""
Multi-Token Prediction (MTP) heads — RDMCA performance report (DeepSeek-V3 style).

N auxiliary heads each predict a token FURTHER into the future than the main head:
head k predicts token t+(k+2) from the SAME transformer hidden state `h` (one
forward pass for all of them). This adds a denser per-token learning signal —
each position now supervises several future tokens, not just the next one — which
improves sample efficiency and sharpens the shared representations the MAIN head
also reads. (The heads are independent of the weight-tied output head: they learn
offset-specific projections.)

The heads form a small sequential residual chain (DeepSeek-V3): head 0 conditions
on the next token's embedding; head k>0 conditions on head k-1's hidden. Only cheap
linear projections + a tiny SwiGLU are sequential — the expensive transformer stack
is shared.

Two deliberate departures from the source report (its snippets predate this
codebase): (1) shapes are aligned to our `mrl_loss(tokens[B,S+1], mask)` and our
`ops.cross_entropy`; (2) output heads are zero-initialised so MTP contributes
nothing at step 0 and ramps up — it never destabilises the main loss or a
warm-started core. The heads serialise as ordinary Module children (no checkpoint
plumbing) and freeze with the core after BCF, since they are part of the cognitive
core. They are built only when `cfg.n_mtp_heads > 0`, so the default path is
untouched.

Speculative decoding (verify N drafted tokens in one pass) is the INFERENCE payoff
and is intentionally NOT wired yet: it needs trained heads to draft usefully and is
addable later with ZERO retrain — exactly the part that should follow, not precede,
training. What MUST exist from the start is the training head (it shapes the
checkpoint); that is what this module provides.
"""

from __future__ import annotations

import src.backend as backend

B = backend.current()
nn = B.nn
ops = B.ops

# Reuse the core building blocks (now in src/model/blocks.py).
from src.model.blocks import RMSNorm, SwiGLU


class MTPModule(nn.Module):
    """`n_heads` auxiliary multi-token-prediction heads over a shared hidden state.

    Per head k (predicting offset k+2):
        x   = silu(proj_in(LN_in(h_slice))) * proj_res(LN_res(res_slice))
        h_k = SwiGLU(x)
        logits_k = head(h_k)                          # [B, S-(k+1), vocab]
    where the residual `res` is the next-token embedding for head 0, then head
    k-1's hidden for k>0 (so dims differ between head 0 and the rest)."""

    def __init__(self, cfg, n_heads: int, hidden: int):
        super().__init__()
        self.n_heads = n_heads
        self.hidden = hidden
        d = cfg.d_model

        # residual source width: head 0 ← token embedding (d_model); head k>0 ←
        # previous head's hidden (`hidden`).
        def res_dim(k):
            return d if k == 0 else hidden

        self.ln_in = nn.ModuleList([RMSNorm(d) for _ in range(n_heads)])
        self.ln_res = nn.ModuleList([RMSNorm(res_dim(k)) for k in range(n_heads)])
        self.proj_in = nn.ModuleList([nn.Linear(d, hidden, bias=False) for _ in range(n_heads)])
        self.proj_res = nn.ModuleList(
            [nn.Linear(res_dim(k), hidden, bias=False) for k in range(n_heads)]
        )
        self.block = nn.ModuleList([SwiGLU(hidden, hidden * 4) for _ in range(n_heads)])
        self.head = nn.ModuleList(
            [nn.Linear(hidden, cfg.vocab_size, bias=False) for _ in range(n_heads)]
        )
        # Zero-init the output heads → uniform logits at start, no perturbation of
        # the main loss while the heads warm up (same trick as the LoRA B matrix).
        for hd in self.head:
            hd.weight = nn.Parameter(ops.zeros(hd.weight.shape))

    def __call__(self, h, embed_next):
        """h: [B,S,d_model] (post-ln_f hidden). embed_next: [B,S,d_model] token
        embeddings of the inputs (its [:,1:] slice is each position's NEXT-token
        embedding, the head-0 residual). Returns a list of `n_heads` logits tensors,
        logits[k] = [B, S-(k+1), vocab] predicting tokens at offset k+2."""
        S = h.shape[1]
        logits = []
        prev = embed_next  # residual chain seed (d_model)
        for k in range(self.n_heads):
            h_k = h[:, : S - (k + 1)]  # [B, S-(k+1), d_model]
            res_k = prev[:, 1:]  # [B, S-(k+1), res_dim] (shift forward)
            x = ops.silu(self.proj_in[k](self.ln_in[k](h_k))) * self.proj_res[k](
                self.ln_res[k](res_k)
            )
            hk = self.block[k](x)  # [B, S-(k+1), hidden]
            logits.append(self.head[k](hk))  # [B, S-(k+1), vocab]
            prev = hk  # head k+1 chains on this hidden
        return logits
