"""
RDMCA Foundational Transformer — T2 Edge (d_model=256)
Decoder-only, RoPE, RMSNorm pre-norm, SwiGLU FFN.
MRL (Matryoshka Representation Learning) loss over nested prefix dims.

Backend-neutral: written once against the active backend facade
(`src.backend.current()`), so the same code runs on MLX or PyTorch. Select the
backend (via `src.backend.select`) BEFORE importing this module.
"""

from __future__ import annotations

import src.backend as backend
from src.model.blocks import (
    RMSNorm,
    TransformerBlock,
    apply_rope,  # noqa: F401  (re-exported for compatibility)
)
from src.model.config import ModelConfig  # re-exported below for compatibility

B = backend.current()
nn = B.nn
ops = B.ops

# Sectors that are NEVER MoE experts: the Behavioral/BCF sector (S7) stays
# always-on and isolated for the safety guarantee (trained only on the
# adversarial buffer). All other sectors are gated MoE experts.
SAFETY_SECTOR_IDS = (7,)


# Precision is owned by the engine; this is the canonical helper call sites use to
# cast a module's float params (training setup, checkpoint loading, the daemon).
def set_model_precision(model, precision: str) -> None:
    """Cast all float parameters of a module to the given precision in place."""
    B.engine.set_precision(model, precision)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class RDMCAFoundational(nn.Module):
    """
    Foundational decoder-only transformer for RDMCA T2 Edge.
    Supports MRL (Matryoshka) training loss over nested prefix dimensions.
    After the ethics/BCF stage (stage 6) all parameters are frozen — LoRA sectors build on top.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        # TODO(multimodal / "M1"): embed is sized to cfg.vocab_size, which is the
        # TEXT vocab (8192) — deliberately, to avoid wasting params and creating
        # "phantom logits" over the unused image/audio ranges of the unified layout
        # (text [0,8192) · image [8192,16384) · audio [16384,20480)). CONSEQUENCE:
        # feeding an image/audio token id (≥8192) would index out of bounds. Text-only
        # training is fine today, but BEFORE enabling multimodal, add a
        # `model.extend_vocab(new_size)` that grows embed.weight (new rows =
        # mean(existing) + small noise) with a migration checkpoint — do NOT just
        # train at the full 20480 vocab (that brings back the phantom-logit problem
        # the text_vocab_size fix removed). Note: embed is WEIGHT-TIED to the output
        # projection (see head_at_dim), so growing embed.weight grows both at once.
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = RMSNorm(cfg.d_model)

        # Optional performance modules — both off by default (cfg defaults 0), both
        # part of the cognitive core (freeze with it after BCF), both serialise as
        # plain Module children. PLE refreshes per-layer token identity (compression);
        # MTP adds auxiliary future-token heads (denser training signal). Built only
        # when enabled, so the default model is byte-for-byte unchanged.
        self.ple = None
        if cfg.ple_dim > 0:
            from src.model.ple import PerLayerEmbeddings

            self.ple = PerLayerEmbeddings(cfg)
        self.mtp = None
        if cfg.n_mtp_heads > 0:
            from src.model.mtp import MTPModule

            self.mtp = MTPModule(cfg, cfg.n_mtp_heads, cfg.mtp_hidden_dim or cfg.d_model // 2)

        # Output projection is WEIGHT-TIED to the input embedding (no separate head):
        # logits = h @ embed.weight.T. This is the canonical Matryoshka setup — a single
        # nested [vocab, d_model] matrix serves input lookup AND output projection at
        # every MRL prefix dim (head_at_dim slices embed.weight[:, :d]), so a tier
        # truncation (embed.weight[:, :d]) stays consistent on both ends. Tying also
        # reclaims ~vocab·d params (here ~2.1M, ~20% of L1) for the transformer body.

        # Sector state (populated by attach_sectors). int-keyed dict for logic;
        # params are registered for the backend via engine.register_submodules.
        self.sectors = None
        self.gate = None  # SectorGate (MoE router over S1..S6)
        self._expert_ids = []  # sector ids routed by the gate (experts)
        self._safety_ids = []  # always-on, isolated safety sectors (S7)
        self._moe_capacity_factor = 1.25  # per-expert capacity slack (MLX dispatch)
        self._routing = _Routing()
        self._aux_accum = 0.0  # MoE load-balance loss accumulated per forward
        self._aux_count = 0

    # ------------------------------------------------------------------
    # Output projection at an MRL prefix dimension
    # ------------------------------------------------------------------

    def head_at_dim(self, h, d: int):
        """Project the first d hidden dims to vocab using the tied embedding as the
        output head (weight tying): logits = h[:, :d] @ embed.weight[:, :d].T."""
        w_d = self.embed.weight[:, :d]  # [vocab, d] — tied input/output matrix
        return h[..., :d] @ w_d.T  # [..., vocab]

    # ------------------------------------------------------------------

    def __call__(self, tokens, mask=None):
        """Forward pass. Returns full-dim hidden states [B, S, d_model]."""
        self._aux_accum = 0.0  # reset MoE aux loss for this forward
        self._aux_count = 0
        x = self.drop(self.embed(tokens))
        ckpt = getattr(self.cfg, "gradient_checkpointing", False)
        for i, block in enumerate(self.blocks):
            # Optionally recompute the block in backward instead of storing its
            # activations (engine.checkpoint checkpoints the block MODULE's params;
            # it is a no-op at inference). PLE is injected OUTSIDE the checkpoint so
            # its own params get ordinary gradients — a fresh per-layer token-identity
            # signal added once after each block.
            x = B.engine.checkpoint(block, x, mask) if ckpt else block(x, mask)
            if self.ple is not None:
                x = x + self.ple(tokens, x, i)
        return self.ln_f(x)

    def forward_cached(self, tokens, caches=None, pos_offset=0):
        """Forward with a per-layer KV cache for autoregressive generation.

        `tokens`  : [B, S] — the NEW tokens only (S = prompt length on the prefill
                    call, then S = 1 per decode step).
        `caches`  : list of per-layer (k, v) from the previous step, or None to
                    prefill from scratch.
        `pos_offset`: absolute position of the first token in `tokens` (0 on
                    prefill, then the running length) — used for RoPE and so the
                    new queries attend over the whole cached history.

        Returns (h [B, S, d_model], new_caches). Turns the O(n) per-step reprocessing
        of the full sequence into O(1) new-token work, i.e. generation O(n³)→O(n²).
        The training forward (`__call__`) is untouched."""
        x = self.drop(self.embed(tokens))
        new_caches = []
        for i, block in enumerate(self.blocks):
            past = caches[i] if caches is not None else None
            x, kv = block.forward_cached(x, cache=past, pos_offset=pos_offset)
            if self.ple is not None:  # same per-layer PLE as training
                x = x + self.ple(tokens, x, i)
            new_caches.append(kv)
        return self.ln_f(x), new_caches

    def logits_cached(self, tokens, caches=None, pos_offset=0):
        """Cached counterpart of `logits()`: returns (logits, new_caches) at the
        largest MRL dim. Callers feed only the new token(s) each step."""
        h, new_caches = self.forward_cached(tokens, caches, pos_offset)
        return self.head_at_dim(h, self.cfg.mrl_dims[-1]), new_caches

    def aux_loss(self):
        """Mean MoE load-balance loss over the last forward (0 if no MoE routing)."""
        if self._aux_count == 0:
            return 0.0
        return self._aux_accum / self._aux_count

    def logits(self, tokens):
        """Convenience: returns logits at the largest MRL dim."""
        h = self(tokens)
        return self.head_at_dim(h, self.cfg.mrl_dims[-1])

    def eval_ce(self, tokens, mask=None):
        """
        Next-token cross-entropy at full dimension — used for validation perplexity
        (exp(eval_ce)). tokens: [B, S+1]. No MRL weighting.

        mask: [B, S+1] or None. When given, the CE is COMPLETION-ONLY — averaged over
        the unmasked (assistant) target positions, MIRRORING the completion-only
        training loss. Without it, validation would also average the user/system
        CONTEXT tokens, which the model is deliberately NOT trained to predict — that
        inflates perplexity far above what the model actually optimizes (measured: ppl
        ~120 unmasked vs ~18 masked on the same stage-1 checkpoint). Prose (mask all-1)
        is unaffected: masked == plain mean.
        """
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        logits = self.head_at_dim(self(inputs), self.cfg.mrl_dims[-1])
        Bsz, S, V = logits.shape
        if mask is None:
            return ops.cross_entropy(
                logits.reshape(Bsz * S, V), targets.reshape(Bsz * S), reduction="mean"
            )
        ce = ops.cross_entropy(
            logits.reshape(Bsz * S, V), targets.reshape(Bsz * S), reduction="none"
        )
        m = ops.astype(mask[:, 1:].reshape(Bsz * S), ce.dtype)
        return ops.sum(ce * m) / (ops.sum(m) + 1e-6)  # mean over assistant targets

    # ------------------------------------------------------------------
    # MRL loss (multi-scale Matryoshka)
    # ------------------------------------------------------------------

    def mrl_loss(self, tokens, mask=None):
        """
        tokens: [B, S+1] — input + target shifted by 1.
        mask:   [B, S+1] or None — per-token loss weight (1=train, 0=ignore). When
                given, the cross-entropy is COMPLETION-ONLY: averaged over the
                unmasked target positions (the assistant turns), so the model learns
                to ANSWER instead of modelling the user/system context. None keeps
                the plain mean over all tokens (pretraining / validation).
        Returns scalar loss (weighted sum across MRL dims).
        """
        inputs = tokens[:, :-1]  # [B, S]
        targets = tokens[:, 1:]  # [B, S]
        tmask = mask[:, 1:] if mask is not None else None  # align with targets

        h = self(inputs)  # [B, S, d_model]

        # Accumulate into `total` lazily (start from the first weighted term) rather
        # than from a float32 `ops.array(0.0)` — adding a float32 scalar to a bf16
        # cross-entropy raises a dtype mismatch on the torch backend.
        # Uniform weights across MRL prefixes. A 1/d weighting put ~67% of the loss
        # on the smallest dim and only ~33% on the full dim — yet inference uses the
        # FULL d_model head, so its upper columns were undertrained. Equal weight
        # gives the full head a fair share (≥50%).
        total = None
        weights = [1.0 for _ in self.cfg.mrl_dims]
        w_sum = sum(weights)

        for w, d in zip(weights, self.cfg.mrl_dims, strict=False):
            logits_d = self.head_at_dim(h, d)  # [B, S, vocab]
            Bsz, S, V = logits_d.shape
            if tmask is None:
                loss_d = ops.cross_entropy(
                    logits_d.reshape(Bsz * S, V),
                    targets.reshape(Bsz * S),
                    reduction="mean",
                )
            else:
                ce = ops.cross_entropy(  # [Bsz*S] per-token loss
                    logits_d.reshape(Bsz * S, V),
                    targets.reshape(Bsz * S),
                    reduction="none",
                )
                m = ops.astype(tmask.reshape(Bsz * S), ce.dtype)
                # Mean over the unmasked (assistant) targets. +1e-6 guards the rare
                # packed window with no response token (loss 0 ⇒ no gradient).
                loss_d = ops.sum(ce * m) / (ops.sum(m) + 1e-6)
            term = (w / w_sum) * loss_d
            total = term if total is None else total + term

        # Multi-Token Prediction auxiliary loss (optional): each head k predicts the
        # token at offset k+2 off the SAME hidden `h` (one forward). Reuses the
        # completion-only mask so MTP, like the main loss, learns on the response
        # tokens only. No-op when MTP is disabled.
        if self.mtp is not None:
            embed_next = self.embed(inputs)  # [B,S,d_model]
            for k, logits_k in enumerate(self.mtp(h, embed_next)):
                tgt_k = targets[:, k + 1 :]  # head k → tokens[:, k+2:]
                Bk, Sk, Vk = logits_k.shape
                if tmask is None:
                    loss_k = ops.cross_entropy(
                        logits_k.reshape(Bk * Sk, Vk), tgt_k.reshape(Bk * Sk), reduction="mean"
                    )
                else:
                    ce = ops.cross_entropy(
                        logits_k.reshape(Bk * Sk, Vk), tgt_k.reshape(Bk * Sk), reduction="none"
                    )
                    mk = ops.astype(tmask[:, k + 1 :].reshape(Bk * Sk), ce.dtype)
                    loss_k = ops.sum(ce * mk) / (ops.sum(mk) + 1e-6)
                total = total + self.cfg.mtp_loss_weight * loss_k

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

    def attach_sectors(
        self, sectors: dict, moe: bool = True, top_k: int = 2, capacity_factor: float = 1.25
    ) -> None:
        """
        Register LoRA sector adapters and wire them into every attention block.
        `sectors` is a {sector_id: SectorAdapter} mapping. Adapters are
        zero-output at init, so attaching them does not change model behavior
        until they are trained. Call after the foundational core is frozen.

        With `moe=True`, the non-safety sectors (S1..S6) become **MoE experts**
        routed per token by a learned `SectorGate` (top-k); the safety sector(s)
        S7 stay always-on and isolated. With `moe=False` the model uses the
        legacy explicit routing (`set_active_sectors`).
        """
        from src.model.moe import SectorGate

        self.sectors = sectors
        self._moe_capacity_factor = capacity_factor
        B.engine.register_submodules(self, "_sector_store", list(sectors.values()))
        self._safety_ids = [s for s in sectors if s in SAFETY_SECTOR_IDS]
        self._expert_ids = sorted(s for s in sectors if s not in SAFETY_SECTOR_IDS)

        if moe and self._expert_ids:
            self.gate = SectorGate(self.cfg.d_model, len(self._expert_ids), top_k=top_k)
            B.engine.align_module(self.gate, self)  # match model device/dtype
            self._routing.mode = "moe"
        else:
            self.gate = None
            self._routing.mode = "explicit"

        for i, block in enumerate(self.blocks):
            block.attn.layer_idx = i
            block.attn._sector_delta = self._compute_delta
            block.attn._sector_route = self._route

    def add_sector(self, sector_id: int, rank: int = 4):
        """Instantiate and register a new adaptive sector at runtime (PGQ new
        sector creation, §10.7.4). It participates in inference immediately
        because _compute_delta reads self.sectors live. If MoE is active and the
        sector is an expert (not safety), the gate grows by one column."""
        from src.model.config import LoRAConfig
        from src.model.lora import SectorAdapter

        if self.sectors is None:
            self.attach_sectors({})
        adapter = SectorAdapter(
            LoRAConfig(
                d_model=self.cfg.d_model,
                n_layers=len(self.blocks),
                sector_id=sector_id,
                rank=rank,
                kv_dim=self.cfg.kv_dim,
            )
        )
        self.sectors[sector_id] = adapter
        # Re-register so the new adapter's params are tracked by the backend.
        B.engine.register_submodules(self, "_sector_store", list(self.sectors.values()))
        if sector_id in SAFETY_SECTOR_IDS:
            if sector_id not in self._safety_ids:
                self._safety_ids.append(sector_id)
        else:
            if sector_id not in self._expert_ids:
                self._expert_ids.append(sector_id)
                if self.gate is not None:
                    self.gate.grow_experts(1)  # new expert column (zero-init)
        return adapter

    def use_moe(self) -> None:
        """Switch back to MoE routing (after a temporary explicit/core-only mode)."""
        self._routing.mode = "moe" if self.gate is not None else "explicit"

    def set_active_sectors(self, pairs) -> None:
        """Legacy explicit routing: only the listed (sector_id, weight) pairs
        contribute, the gate is bypassed. Used for the isolated S7 update and for
        BCF core-only reads (`set_active_sectors([])`)."""
        self._routing.mode = "explicit"
        self._routing.active = list(pairs) if pairs else []

    def _route(self, x):
        """Per-token MoE routing weights over the experts (S1..S6), or None when
        not in MoE mode. Also accumulates the load-balance aux loss."""
        if self.gate is None or self._routing.mode != "moe":
            return None
        from src.model.moe import expert_weights, load_balance_loss

        idx, w, logits = self.gate(x)
        self._aux_accum = self._aux_accum + load_balance_loss(logits, idx, len(self._expert_ids))
        self._aux_count += 1
        return expert_weights(idx, w, len(self._expert_ids))  # [B, S, n_experts]

    def _compute_delta(self, layer_idx: int, proj: str, x, route=None):
        """Combine sector deltas for one projection in one layer.
        MoE mode: experts weighted by the per-token gate (`route`) + always-on
        safety sectors. Explicit mode: sum the `set_active_sectors` list."""
        if not self.sectors:
            return 0.0
        if self._routing.mode == "moe" and self.gate is not None and route is not None:
            return self._moe_combine(layer_idx, proj, x, route)
        # explicit/legacy: sum the listed (sector, weight) pairs
        total = None
        for sid, weight in self._routing.active:
            adapter = self.sectors.get(sid)
            if adapter is None:
                continue
            d = adapter.delta(layer_idx, proj, x) * weight
            total = d if total is None else total + d
        return total if total is not None else 0.0

    def _moe_combine(self, layer_idx: int, proj: str, x, route):
        """Combine the routed expert deltas (S1..S6) + always-on safety (S7).

        Both dispatch paths compute each expert ONLY on its routed tokens
        (real top-k saving — bounded as the expert pool grows):
          • `_moe_sparse` (backends with dynamic `nonzero`, e.g. PyTorch): exact,
            no token drops.
          • `_moe_capacity` (static-shape backends, e.g. MLX): GShard-style
            fixed-capacity dispatch (gather/scatter by index, no dynamic shapes);
            tokens beyond an expert's capacity are dropped (rare with the default
            capacity factor)."""
        Bsz, S, D = x.shape
        flat = x.reshape(-1, D)  # [T, D]
        w = route.reshape(-1, len(self._expert_ids))  # [T, E]
        if getattr(ops, "nonzero", None) is not None:
            out = self._moe_sparse(layer_idx, proj, flat, w)
        else:
            out = self._moe_capacity(layer_idx, proj, flat, w)
        for sid in self._safety_ids:  # safety: always on
            out = out + self.sectors[sid].delta(layer_idx, proj, flat)
        return out.reshape(Bsz, S, D)

    def _moe_sparse(self, layer_idx, proj, flat, w):
        """Exact sparse dispatch: gather each expert's routed tokens via nonzero,
        run the expert on that subset, scatter the weighted result back."""
        out = ops.zeros((flat.shape[0], flat.shape[1]), dtype=flat.dtype)
        for pos, sid in enumerate(self._expert_ids):
            we = w[:, pos]  # [T] gate weight
            nz = ops.nonzero(we > 0.0)  # tokens routed here
            if nz.shape[0] == 0:
                continue
            xe = ops.index_select(flat, nz, 0)  # [m, D]
            de = self.sectors[sid].delta(layer_idx, proj, xe)
            out = ops.index_add(out, nz, de * we[nz][:, None])
        return out

    def _moe_capacity(self, layer_idx, proj, flat, w):
        """Static-shape capacity dispatch (GShard-style) for backends without a
        dynamic nonzero. Each expert processes a fixed C = ceil(factor·k·T/E)
        token slots, so total expert compute ≈ O(top_k·T) regardless of E."""
        T, D = flat.shape
        E = len(self._expert_ids)
        k = self.gate.top_k
        C = max(int(self._moe_capacity_factor * k * T / E) + 1, 1)

        # Routing positions/indices are non-differentiable by nature — keep them
        # out of the autograd graph (stop_gradient), or MLX errors trying to take
        # a VJP w.r.t. scatter indices. The combine WEIGHTS stay differentiable so
        # the gate still learns.
        keep = ops.astype(w > 0.0, ops.float32)  # [T, E] routed?
        pos = ops.astype(ops.cumsum(keep, axis=0), ops.int_) - 1  # queue position per expert
        within = keep * ops.astype(pos < C, ops.float32)  # 1 if kept within capacity
        keep_i = ops.astype(within, ops.int_)  # [T, E] int 0/1

        erange = ops.astype(ops.arange(E), ops.int_)[None, :]  # [1, E]
        slot = erange * C + pos  # [T, E] target slot per expert
        trash = E * C  # overflow / not-routed sink
        target = ops.stop_gradient(slot * keep_i + trash * (1 - keep_i))  # [T, E] int (const)
        tok = ops.astype(ops.arange(T), ops.int_)[:, None] + ops.zeros((1, E), dtype=ops.int_)

        disp = ops.stop_gradient(
            ops.index_add(
                ops.zeros((E * C + 1,), dtype=ops.int_), target.reshape(-1), tok.reshape(-1)
            )
        )  # token id per slot
        comb = ops.index_add(
            ops.zeros((E * C + 1,)), target.reshape(-1), w.reshape(-1)
        )  # gate weight per slot (diff)
        disp = disp[: E * C].reshape(E, C)
        comb = comb[: E * C].reshape(E, C)

        out = ops.zeros((T, D), dtype=flat.dtype)
        for e, sid in enumerate(self._expert_ids):
            idx_e = ops.stop_gradient(disp[e])  # [C] token indices (const)
            xe = ops.index_select(flat, idx_e, 0)  # [C, D]
            de = self.sectors[sid].delta(layer_idx, proj, xe)  # [C, D]
            out = ops.index_add(out, idx_e, de * comb[e][:, None])
        return out


class _Routing:
    """Plain (non-Module) holder for routing state, so the backend parameter tree
    never tries to traverse it. `mode` is 'moe' (gated experts + safety) or
    'explicit' (only the `active` list)."""

    def __init__(self):
        self.mode = "explicit"
        self.active = []
