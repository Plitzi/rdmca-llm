"""
PyTorch backend — implements the Backend facade on torch.nn / torch.optim so
the same model code runs on CUDA (Linux/cloud), MPS (Mac) or CPU.

Design notes:
  - Model modules subclass `torch.nn.Module` and define `__call__` (MLX
    convention). Overriding `__call__` skips nn.Module forward-hooks, which
    this codebase does not use; parameter registration (via __setattr__) and
    autograd are unaffected.
  - `ops` mirror MLX signatures (`axis=`, `keepdims=`) and place new tensors on
    the selected device.
  - Gradient accumulation is kept faithful to the MLX path: `value_and_grad`
    zeroes grads at the start of each call, so after the micro-batch loop only
    the last micro-batch's gradient remains (matching the current MLX loop,
    which overwrites its `grads` variable each iteration).
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as torch_nn
import torch.nn.functional as F

from src.core.backend.base import Backend
from src.core.backend.torch_device import DEVICE
from src.core.backend.torch_quant import quantize as _quantize

_PRECISION = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}

_GRAD_SENTINEL = object()  # torch keeps grads on tensors; engine returns this


# ───────────────────────── nn namespace ──────────────────────────────────────
_nn = SimpleNamespace(
    Module=torch_nn.Module,
    Linear=torch_nn.Linear,
    Embedding=torch_nn.Embedding,
    Dropout=torch_nn.Dropout,
    Conv1d=torch_nn.Conv1d,
    Conv2d=torch_nn.Conv2d,
    ConvTranspose1d=torch_nn.ConvTranspose1d,
    ConvTranspose2d=torch_nn.ConvTranspose2d,
    Parameter=lambda a: torch_nn.Parameter(
        a if isinstance(a, torch.Tensor) else torch.as_tensor(a, device=DEVICE)
    ),
    ModuleList=lambda items: torch_nn.ModuleList(items),
    ModuleDict=lambda d: torch_nn.ModuleDict(d),
)


# ───────────────────────── ops namespace ─────────────────────────────────────
def _array(x, dtype=None):
    t = x.to(DEVICE) if isinstance(x, torch.Tensor) else torch.as_tensor(x, device=DEVICE)
    return t.to(dtype) if dtype is not None else t


def _arange(n, dtype=None):
    return torch.arange(n, device=DEVICE, dtype=dtype)


def _mean(x, axis=None, keepdims=False):
    return x.mean() if axis is None else x.mean(dim=axis, keepdim=keepdims)


def _sum(x, axis=None, keepdims=False):
    return x.sum() if axis is None else x.sum(dim=axis, keepdim=keepdims)


_ops = SimpleNamespace(
    array=_array,
    arange=_arange,
    zeros=lambda shape, dtype=None: torch.zeros(shape, device=DEVICE, dtype=dtype),
    ones=lambda shape, dtype=None: torch.ones(shape, device=DEVICE, dtype=dtype),
    full=lambda shape, val: torch.full(tuple(shape), float(val), device=DEVICE),
    randn=lambda shape: torch.randn(tuple(shape), device=DEVICE),
    cos=torch.cos,
    sin=torch.sin,
    sqrt=torch.sqrt,
    sigmoid=torch.sigmoid,
    mean=_mean,
    sum=_sum,
    concatenate=lambda arrays, axis=0: torch.cat(list(arrays), dim=axis),
    outer=torch.outer,
    softmax=lambda x, axis=-1: torch.softmax(x, dim=axis),
    # Fused scaled-dot-product attention (selects Flash / mem-efficient kernels).
    # q:[B,H,S,Hd], k/v:[B,Hkv,T,Hd] — enable_gqa handles Hkv<H natively. Uses the
    # built-in causal mask when `is_causal` (and no explicit additive mask).
    sdpa=lambda q, k, v, scale, is_causal=False, attn_mask=None: F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        scale=scale,
        is_causal=(is_causal and attn_mask is None),
        enable_gqa=(q.shape[1] != k.shape[1]),
    ),
    triu=lambda x, k=0: torch.triu(x, diagonal=k),
    argmax=lambda x, axis=-1: torch.argmax(x, dim=axis),
    argmin=lambda x, axis=-1: torch.argmin(x, dim=axis),
    transpose=lambda x, axes: x.permute(*axes),
    astype=lambda x, dtype: x.to(dtype),
    stop_gradient=lambda x: x.detach(),
    topk=lambda x, k, axis=-1: torch.topk(x, k, dim=axis),  # -> (values, indices)
    take_along_axis=lambda x, idx, axis: torch.take_along_dim(x, idx, dim=axis),
    index_select=lambda x, idx, axis=0: x.index_select(axis, idx),
    index_add=lambda out, idx, vals, axis=0: out.index_add(axis, idx.long(), vals),
    nonzero=lambda mask: torch.nonzero(mask, as_tuple=False).flatten(),
    cumsum=lambda x, axis=0: torch.cumsum(x, dim=axis),
    where=lambda cond, a, b: torch.where(cond, a, b),
    int_=torch.int32,  # match MLX so `ops.int_` means the same on both backends
    # (torch promotes int32 indices fine; only the MLX static
    #  capacity path actually consumes ops.int_).
    silu=F.silu,
    relu=F.relu,
    cross_entropy=lambda logits, targets, reduction="mean": F.cross_entropy(
        logits, targets.long(), reduction=reduction
    ),
    bce_with_logits=lambda logits, labels, reduction="mean": F.binary_cross_entropy_with_logits(
        logits, labels.to(logits.dtype), reduction=reduction
    ),
    to_numpy=lambda x: x.detach().to(torch.float32).cpu().numpy(),
    from_numpy=lambda a: torch.as_tensor(a, device=DEVICE),
    float32=torch.float32,
    bfloat16=torch.bfloat16,
    float16=torch.float16,
)


# ───────────────────────── engine namespace ──────────────────────────────────
def _value_and_grad(model, fn):
    def run(*args):
        model.zero_grad(set_to_none=True)
        loss = fn(*args)
        loss.backward()
        return loss.detach(), _GRAD_SENTINEL

    return run


def _optimizer_step(opt, model, grads):
    opt.step()


def _save_optimizer(opt, path: str) -> None:
    """Persist optimizer state (AdamW m/v/step) so --resume continues with warm
    moments. Torch-native blob (resume runs on the same backend)."""
    tmp = str(path) + ".tmp"
    torch.save(opt.state_dict(), tmp)
    os.replace(tmp, str(path))


def _load_optimizer(opt, path: str) -> bool:
    """Restore optimizer state saved by _save_optimizer. Returns False if absent."""
    if not os.path.exists(path):
        return False
    opt.load_state_dict(torch.load(str(path), map_location=DEVICE))
    return True


def _set_lr(opt, lr):
    for g in opt.param_groups:
        g["lr"] = lr


def _grad_norm(model, grads) -> float:
    sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sq += float(p.grad.detach().pow(2).sum().item())
    return sq**0.5


def _accumulate_grads(running, grads, model):
    """True gradient accumulation. `grads` is the sentinel; the real gradients are
    on p.grad (just produced by this micro-batch's backward, which value_and_grad
    zeroed at its start). Snapshot and sum them across micro-batches into a dict."""
    snap = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    if running is None:
        return snap
    for n, v in snap.items():
        running[n] = running[n] + v if n in running else v
    return running


def _finalize_grads(running, scale: float, model):
    """Write the accumulated (scaled) gradients back onto p.grad so optimizer_step
    and clip_grads — which both read p.grad — operate on the summed gradient."""
    for n, p in model.named_parameters():
        if running is not None and n in running:
            p.grad = running[n] * scale
    return _GRAD_SENTINEL


def _clip_grads(model, grads, max_norm: float):
    """Global-norm gradient clipping. Grads live on the params (set by backward),
    so this clips them in place via torch's utility; the sentinel is returned
    unchanged so the call site stays backend-agnostic."""
    torch_nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return grads


def _set_precision(model, precision: str) -> None:
    model.to(device=DEVICE, dtype=_PRECISION[precision])


def _save_weights(model, path: str) -> None:
    """Neutral checkpoint: float32 numpy .npz keyed by state_dict names —
    identical naming to the MLX backend, so checkpoints are cross-loadable."""
    flat = {k: v.detach().to(torch.float32).cpu().numpy() for k, v in model.state_dict().items()}
    tmp = str(path) + ".tmp"  # atomic: write fully, then rename into place
    with open(tmp, "wb") as f:  # file object → np.savez does NOT append .npz
        np.savez(f, **flat)
    os.replace(tmp, str(path))


def _load_weights(model, path: str) -> None:
    data = np.load(str(path))
    sd = {k: torch.as_tensor(data[k]) for k in data.files}
    from src.core.backend.base import warn_load_mismatch

    warn_load_mismatch(
        {n: tuple(p.shape) for n, p in model.state_dict().items()},
        {k: tuple(data[k].shape) for k in data.files},
        str(path),
    )
    model.load_state_dict(sd, strict=False)
    model.to(DEVICE)


def _state_dict(module) -> dict:
    """Module params as a {name: float32 numpy} dict (neutral, cross-backend)."""
    return {k: v.detach().to(torch.float32).cpu().numpy() for k, v in module.state_dict().items()}


def _load_state_dict(module, mapping: dict) -> None:
    sd = {k: torch.as_tensor(v) for k, v in mapping.items()}
    module.load_state_dict(sd, strict=False)
    module.to(DEVICE)


def _freeze_all(model) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def _set_trainable(model, modules) -> None:
    _freeze_all(model)
    for m in modules:
        for p in m.parameters():
            p.requires_grad_(True)


def _register_submodules(parent, name, modules) -> None:
    """Register a collection of Modules so their params appear in
    parent.parameters()/state_dict(). A plain dict/list attribute on a
    torch.nn.Module is NOT registered, so we wrap the modules in a ModuleList
    stored under `name`. Callers keep their own int-keyed dict for logic.

    Newly built modules default to CPU/float32; align them to the parent's
    current device and dtype so deltas interoperate with the (possibly already
    moved/cast) base model without a separate set_precision call."""
    ml = torch_nn.ModuleList(list(modules))
    setattr(parent, name, ml)
    try:
        ref = next(
            p
            for p in parent.parameters()
            if p.numel() and id(p) not in {id(q) for q in ml.parameters()}
        )
        ml.to(device=ref.device, dtype=ref.dtype)
    except StopIteration:
        ml.to(DEVICE)


def _align_module(module, model) -> None:
    """Move a newly created submodule (e.g. the MoE gate) to the model's current
    device/dtype, so it interoperates whether it was added before or after
    set_precision()."""
    try:
        own = {id(q) for q in module.parameters()}
        ref = next(p for p in model.parameters() if id(p) not in own)
        module.to(device=ref.device, dtype=ref.dtype)
    except StopIteration:
        module.to(DEVICE)


def _memory_stats() -> dict:
    if torch.cuda.is_available():
        return {"peak": torch.cuda.max_memory_allocated(), "active": torch.cuda.memory_allocated()}
    return {"peak": 0, "active": 0}


def _make_optimizer(model, lr, weight_decay, states=None):
    """AdamW with a selectable optimizer-state precision.

    `states`:
      • None / 'bf16' / 'default' → torch AdamW. Its moments (exp_avg/exp_avg_sq)
        are created as `zeros_like(param)`, so with the model cast to bf16 they are
        ALREADY bf16 (~2 bytes/state) — the fp32→bf16 saving the report describes is
        thus already in effect here.
      • 'int8' / '8bit' → bitsandbytes 8-bit AdamW (block-wise quantized states,
        ~1 byte/state) — a further saving on big CUDA runs. OPT-IN and graceful:
        falls back to bf16-state AdamW (with a warning) when bitsandbytes is missing
        or there is no CUDA device, so it never becomes a hard dependency."""
    if states and str(states).lower() in ("int8", "8bit", "8-bit"):
        reason = None
        try:
            import bitsandbytes as bnb

            if torch.cuda.is_available():
                return bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=weight_decay)
            reason = "no CUDA device"
        except Exception as e:  # not installed / import error
            reason = f"bitsandbytes unavailable ({e})"
        import warnings

        warnings.warn(
            f"optimizer_states='int8' requested but unused ({reason}); using bf16-state AdamW.",
            stacklevel=2,
        )
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def _checkpoint(module, *args):
    """Gradient (activation) checkpointing of a MODULE: run `module(*args)` WITHOUT
    keeping its intermediate activations, recomputing them in the backward pass
    instead — trades ~20-30% extra compute for a large drop in activation memory (the
    dominant term for long contexts). `use_reentrant=False` preserves the RNG state,
    so dropout is identical on the forward and the recompute (correct gradients), and
    the module's parameters receive gradients normally. A no-op outside a grad context
    (inference)."""
    if torch.is_grad_enabled():
        import torch.utils.checkpoint as _ckpt

        return _ckpt.checkpoint(module, *args, use_reentrant=False)
    return module(*args)


def _set_seed(seed: int) -> None:
    """Seed every RNG that affects a training run (Python, numpy, torch CPU+CUDA),
    so weight init + dropout + sampling are reproducible across runs."""
    import random as _random

    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_engine = SimpleNamespace(
    value_and_grad=_value_and_grad,
    make_optimizer=_make_optimizer,
    optimizer_step=_optimizer_step,
    set_lr=_set_lr,
    eval=lambda *xs: None,
    item=lambda x: float(x.item()),
    set_precision=_set_precision,
    quantize=_quantize,
    set_train=lambda model: model.train(),
    set_eval=lambda model: model.eval(),
    save_weights=_save_weights,
    load_weights=_load_weights,
    state_dict=_state_dict,
    load_state_dict=_load_state_dict,
    set_trainable=_set_trainable,
    freeze_all=_freeze_all,
    register_submodules=_register_submodules,
    align_module=_align_module,
    grad_norm=_grad_norm,
    clip_grads=_clip_grads,
    accumulate_grads=_accumulate_grads,
    finalize_grads=_finalize_grads,
    save_optimizer=_save_optimizer,
    load_optimizer=_load_optimizer,
    param_count=lambda module: sum(p.numel() for p in module.parameters()),
    memory_stats=_memory_stats,
    set_seed=_set_seed,
    checkpoint=_checkpoint,
)


class TorchBackend(Backend):
    name = "torch"
    nn = _nn
    ops = _ops
    engine = _engine
    device = DEVICE


def build() -> TorchBackend:
    return TorchBackend()
