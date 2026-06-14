"""
Per-Layer Embeddings (PLE) — RDMCA performance report (Gemma-style compression).

A standard transformer embeds a token ONCE at the input; in a deep stack that
token-identity signal dilutes as it flows through the residual stream, so deep
layers must reconstruct "which word is this" from context. PLE gives EVERY block
its own small, dedicated embedding lookup — a "fresh reminder" of token identity —
combined with a context-aware projection of the running hidden state and injected
back into the stream. The lookup tables are large but near-zero-FLOP (and
quantizable / memory-mappable), so they buy capacity cheaply: more "intelligence
per active parameter", with the biggest relative gain in small/deep models.

This is a SIDE pathway — it does not touch attention, MRL, weight tying, or LoRA.
It is applied identically in training and cached generation (threaded through both
forwards), so generation never diverges from training.

One deliberate fix vs the source report: the report adds the d_ple-wide PLE vector
straight into the d_model residual stream (a dimension mismatch). PLE here keeps
the lookup cheap at `d_ple` and adds a per-layer **up-projection** d_ple→d_model so
the injection is dimensionally correct. The up-projection is zero-initialised, so
PLE is a no-op at step 0 and ramps up — enabling it never destabilises training or
a warm-started core. Built only when `cfg.ple_dim > 0`, so the default path is
untouched. (`ops.embedding` does not exist in our backend surface; we use one
`nn.Embedding` per layer for the lookup.)
"""

from __future__ import annotations

import src.backend as backend

B = backend.current()
nn = B.nn
ops = B.ops


class PerLayerEmbeddings(nn.Module):
    """One cheap token-identity lookup + gated context merge + up-projection per
    decoder layer. `__call__(token_ids, hidden, layer_idx)` returns a [B,S,d_model]
    vector to add into that layer's residual stream."""

    def __init__(self, cfg):
        super().__init__()
        self.n_layers = cfg.n_layers
        self.gated = cfg.ple_gated
        d_ple = cfg.ple_dim
        self.emb = nn.ModuleList(
            [nn.Embedding(cfg.vocab_size, d_ple) for _ in range(cfg.n_layers)]
        )  # lookup
        self.ctx = nn.ModuleList(
            [nn.Linear(cfg.d_model, d_ple, bias=False) for _ in range(cfg.n_layers)]
        )  # context
        self.up = nn.ModuleList(
            [nn.Linear(d_ple, cfg.d_model, bias=False) for _ in range(cfg.n_layers)]
        )  # inject
        if self.gated:
            self.gate = nn.ModuleList(
                [nn.Linear(cfg.d_model, d_ple, bias=False) for _ in range(cfg.n_layers)]
            )
        # Zero-init the up-projection → PLE injects 0 at start (a no-op) and ramps
        # up as it trains, so turning PLE on never perturbs step 0 / a warm core.
        for up in self.up:
            up.weight = nn.Parameter(ops.zeros(up.weight.shape))

    def __call__(self, token_ids, hidden, layer_idx: int):
        raw = self.emb[layer_idx](token_ids)  # [B,S,d_ple] token identity
        ctx = self.ctx[layer_idx](hidden)  # [B,S,d_ple] context-aware
        if self.gated:
            g = ops.sigmoid(self.gate[layer_idx](hidden))
            mix = g * raw + (1.0 - g) * ctx  # learn when to trust identity
        else:
            mix = raw + ctx
        return self.up[layer_idx](mix)  # [B,S,d_model] → residual add
