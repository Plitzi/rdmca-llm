"""
MLX backend — wraps mlx.core / mlx.nn / mlx.optimizers behind the Backend
facade. This is the reference implementation; `ops` signatures are MLX-native
(`axis=`, `keepdims=`) and the Torch backend translates to match.
"""
from __future__ import annotations
import os
from types import SimpleNamespace

import numpy as np
import mlx.core as mx
import mlx.nn as mlx_nn
import mlx.optimizers as mlx_optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from src.backend.base import Backend


_PRECISION = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}
_FLOAT_DTYPES = (mx.float32, mx.bfloat16, mx.float16)


# ───────────────────────── nn namespace ──────────────────────────────────────
# MLX convs are channels-last (NHWC / NLC); the model code is written
# channels-first (NCHW / NCL, the PyTorch convention) so it is backend-neutral.
# These thin wrappers permute around the MLX conv so callers always pass NCHW.
class _Conv2dNCHW(mlx_nn.Module):
    def __init__(self, *a, **k):
        super().__init__()
        self.conv = mlx_nn.Conv2d(*a, **k)

    def __call__(self, x):                       # x: [N, C, H, W]
        x = mx.transpose(x, (0, 2, 3, 1))        # -> NHWC
        x = self.conv(x)
        return mx.transpose(x, (0, 3, 1, 2))     # -> NCHW


class _ConvTranspose2dNCHW(mlx_nn.Module):
    def __init__(self, *a, **k):
        super().__init__()
        self.conv = mlx_nn.ConvTranspose2d(*a, **k)

    def __call__(self, x):
        x = mx.transpose(x, (0, 2, 3, 1))
        x = self.conv(x)
        return mx.transpose(x, (0, 3, 1, 2))


class _Conv1dNCL(mlx_nn.Module):
    def __init__(self, *a, **k):
        super().__init__()
        self.conv = mlx_nn.Conv1d(*a, **k)

    def __call__(self, x):                       # x: [N, C, L]
        x = mx.transpose(x, (0, 2, 1))           # -> NLC
        x = self.conv(x)
        return mx.transpose(x, (0, 2, 1))        # -> NCL


class _ConvTranspose1dNCL(mlx_nn.Module):
    def __init__(self, *a, **k):
        super().__init__()
        self.conv = mlx_nn.ConvTranspose1d(*a, **k)

    def __call__(self, x):
        x = mx.transpose(x, (0, 2, 1))
        x = self.conv(x)
        return mx.transpose(x, (0, 2, 1))


# MLX auto-registers any mx.array attribute as a parameter and walks plain
# lists/dicts of Modules, so Parameter/ModuleList/ModuleDict are identities.
_nn = SimpleNamespace(
    Module=mlx_nn.Module,
    Linear=mlx_nn.Linear,
    Embedding=mlx_nn.Embedding,
    Dropout=mlx_nn.Dropout,
    Conv1d=_Conv1dNCL,
    Conv2d=_Conv2dNCHW,
    ConvTranspose1d=_ConvTranspose1dNCL,
    ConvTranspose2d=_ConvTranspose2dNCHW,
    Parameter=lambda a: a,
    ModuleList=lambda items: list(items),
    ModuleDict=lambda d: dict(d),
)


# ───────────────────────── ops namespace ─────────────────────────────────────
def _array(x, dtype=None):
    a = mx.array(x)
    return a.astype(dtype) if dtype is not None else a


_ops = SimpleNamespace(
    array=_array,
    arange=lambda n, dtype=None: mx.arange(n, dtype=dtype) if dtype else mx.arange(n),
    zeros=lambda shape, dtype=None: mx.zeros(shape, dtype=dtype) if dtype else mx.zeros(shape),
    ones=lambda shape, dtype=None: mx.ones(shape, dtype=dtype) if dtype else mx.ones(shape),
    full=lambda shape, val: mx.full(shape, val),
    randn=lambda shape: mx.random.normal(shape),
    cos=mx.cos, sin=mx.sin, sqrt=mx.sqrt, sigmoid=mx.sigmoid,
    mean=lambda x, axis=None, keepdims=False: mx.mean(x, axis=axis, keepdims=keepdims),
    sum=lambda x, axis=None, keepdims=False: mx.sum(x, axis=axis, keepdims=keepdims),
    concatenate=lambda arrays, axis=0: mx.concatenate(arrays, axis=axis),
    outer=mx.outer,
    softmax=lambda x, axis=-1: mx.softmax(x, axis=axis),
    # Fused scaled-dot-product attention (MLX "flash" kernel). q:[B,H,S,Hd],
    # k/v:[B,Hkv,T,Hd] — GQA (Hkv<H) is handled natively. `is_causal` → built-in
    # causal mask; otherwise `attn_mask` (additive array or None) is used.
    sdpa=lambda q, k, v, scale, is_causal=False, attn_mask=None:
        mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask=("causal" if is_causal else attn_mask)),
    triu=lambda x, k=0: mx.triu(x, k=k),
    argmax=lambda x, axis=-1: mx.argmax(x, axis=axis),
    argmin=lambda x, axis=-1: mx.argmin(x, axis=axis),
    transpose=lambda x, axes: mx.transpose(x, axes),
    astype=lambda x, dtype: x.astype(dtype),
    stop_gradient=mx.stop_gradient,
    # top-k along the last axis (order within the k does not matter — softmax over them).
    topk=lambda x, k, axis=-1: (
        mx.take_along_axis(x, mx.argsort(x, axis=axis)[..., -k:], axis=axis),
        mx.argsort(x, axis=axis)[..., -k:]),
    take_along_axis=lambda x, idx, axis: mx.take_along_axis(x, idx, axis=axis),
    index_select=lambda x, idx, axis=0: mx.take(x, idx, axis=axis),
    index_add=lambda out, idx, vals, axis=0: out.at[idx].add(vals),
    nonzero=None,    # MLX has no static-shape nonzero; the MLX path uses capacity dispatch
    cumsum=lambda x, axis=0: mx.cumsum(x, axis=axis),
    where=lambda cond, a, b: mx.where(cond, a, b),
    int_=mx.int32,
    silu=mlx_nn.silu,
    relu=mlx_nn.relu,
    cross_entropy=lambda logits, targets, reduction="mean": mlx_nn.losses.cross_entropy(
        logits, targets, reduction=reduction),
    bce_with_logits=lambda logits, labels, reduction="mean": mlx_nn.losses.binary_cross_entropy(
        logits, labels, with_logits=True, reduction=reduction),
    # numpy has no bfloat16; cast float types to float32 before converting.
    to_numpy=lambda x: np.array(
        x.astype(mx.float32) if isinstance(x, mx.array) and x.dtype in _FLOAT_DTYPES else x),
    from_numpy=lambda a: mx.array(a),
    float32=mx.float32, bfloat16=mx.bfloat16, float16=mx.float16,
)


# ───────────────────────── engine namespace ──────────────────────────────────
def _set_precision(model, precision: str) -> None:
    dtype = _PRECISION[precision]

    def _cast(p):
        if isinstance(p, mx.array) and p.dtype in _FLOAT_DTYPES:
            return p.astype(dtype)
        return p

    model.update(tree_map(_cast, model.parameters()))
    mx.eval(model.parameters())


def _quantize(model, bits: int = 4, group_size: int = 64,
              skip_names: tuple = ("embed",)) -> None:
    """In-place weight quantization at any MLX-supported bit-width (2/3/4/6/8).

    Uses MLX grouped affine quantization on Linear/Embedding layers whose feature
    dimension is divisible by `group_size`; any layer that doesn't divide evenly
    (small tiers can hit this) is left in its float dtype rather than erroring.
    MLX packs each weight at the true `bits` width, so memory scales with `bits`.
    `skip_names` (last path component) are left in float — by default `embed`, which
    is weight-tied as the output projection (sliced by `.weight` for MRL) and is the
    most quant-sensitive layer."""
    skipped = []

    def predicate(path, m):
        if path.split(".")[-1] in skip_names:
            return False
        w = getattr(m, "weight", None)
        if not (hasattr(m, "to_quantized") and isinstance(w, mx.array)):
            return False
        if w.shape[-1] % group_size != 0:
            skipped.append(path)
            return False
        return True

    mlx_nn.quantize(model, group_size=group_size, bits=bits, class_predicate=predicate)
    mx.eval(model.parameters())
    if skipped:
        print(f"  [quant] {len(skipped)} layer(s) not divisible by group_size={group_size} "
              f"kept in float dtype")


def _optimizer_step(opt, model, grads):
    opt.update(model, grads)
    mx.eval(model.parameters(), opt.state)


def _save_optimizer(opt, path: str) -> None:
    """Persist optimizer state (AdamW m/v/step) so --resume continues with warm
    moments instead of cold ones (a cold restart spikes the loss). Saved as a flat
    .npz of the state tree's array leaves."""
    flat = {k: np.array(v.astype(mx.float32)) for k, v in tree_flatten(opt.state)
            if isinstance(v, mx.array)}
    if not flat:
        # No step taken yet → no moments to persist. Warn (don't silently skip):
        # a later --resume would then find no .opt and start with cold moments.
        import sys
        print(f"  [opt] no optimizer state to save yet (no step taken) — "
              f"skipping {os.path.basename(str(path))}", file=sys.stderr)
        return
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **flat)
    os.replace(tmp, str(path))


def _load_optimizer(opt, path: str) -> bool:
    """Restore optimizer state saved by _save_optimizer. Returns False if absent."""
    if not os.path.exists(path):
        return False
    data = np.load(path)
    opt.state = tree_unflatten([(k, mx.array(data[k])) for k in data.files])
    return True


def _grad_norm(model, grads) -> float:
    sq = 0.0
    for _, g in tree_flatten(grads):
        if isinstance(g, mx.array) and g.size > 0:
            sq += float((g * g).sum().item())
    return sq ** 0.5


def _accumulate_grads(running, grads, model):
    """Sum micro-batch gradient trees for true gradient accumulation. MLX's
    value_and_grad returns a FRESH tree each call, so we add them ourselves."""
    if running is None:
        return grads
    return tree_map(lambda a, b: a + b, running, grads)


def _finalize_grads(running, scale: float, model):
    """Scale the accumulated tree (typically by 1/grad_acc) before the step."""
    if scale == 1.0:
        return running
    return tree_map(lambda a: a * scale, running)


def _clip_grads(model, grads, max_norm: float):
    """Global-norm gradient clipping (returns the scaled grad tree). Mirrors
    torch.nn.utils.clip_grad_norm_: if ‖g‖₂ > max_norm, scale all grads by
    max_norm/‖g‖₂. No-op when already within the threshold."""
    norm = _grad_norm(model, grads)
    if norm <= max_norm or norm == 0.0:
        return grads
    factor = max_norm / (norm + 1e-6)
    return tree_map(lambda g: g * factor if isinstance(g, mx.array) else g, grads)


def _save_weights(model, path: str) -> None:
    """Neutral checkpoint: a .npz of float32 numpy arrays keyed by param name.
    Loadable by any backend (same names) regardless of training precision."""
    flat = {k: np.array(v.astype(mx.float32)) for k, v in tree_flatten(model.parameters())}
    tmp = str(path) + ".tmp"           # atomic: write fully, then rename into place
    with open(tmp, "wb") as f:         # file object → np.savez does NOT append .npz
        np.savez(f, **flat)
    os.replace(tmp, str(path))


def _load_weights(model, path: str) -> None:
    data = np.load(str(path))
    if len(data.files) == 0:                    # empty/corrupt .npz → load_weights([]) is a no-op
        import sys
        print(f"  [load] {path} has no arrays — model stays UNINITIALIZED "
              f"(checkpoint empty or corrupt).", file=sys.stderr)
        return
    from src.backend.base import warn_load_mismatch
    warn_load_mismatch({k: tuple(v.shape) for k, v in tree_flatten(model.parameters())},
                       {k: tuple(data[k].shape) for k in data.files}, str(path))
    model.load_weights([(k, mx.array(data[k])) for k in data.files], strict=False)
    mx.eval(model.parameters())


def _state_dict(module) -> dict:
    """Module params as a {name: float32 numpy} dict (neutral, cross-backend)."""
    return {k: np.array(v.astype(mx.float32)) for k, v in tree_flatten(module.parameters())}


def _load_state_dict(module, mapping: dict) -> None:
    module.load_weights([(k, mx.array(v)) for k, v in mapping.items()], strict=False)
    mx.eval(module.parameters())


def _set_trainable(model, modules) -> None:
    model.freeze()
    for m in modules:
        m.unfreeze()


def _param_count(module) -> int:
    return sum(v.size for _, v in tree_flatten(module.parameters()))


def _memory_stats() -> dict:
    return {"peak": mx.get_peak_memory(), "active": mx.get_active_memory()}


def _checkpoint(module, *args):
    """Gradient (activation) checkpointing of a MODULE: recompute its activations in
    the backward pass instead of storing them — trades compute for a large drop in
    activation memory. Uses `mlx.nn.utils.checkpoint`, which (unlike the bare
    `mx.checkpoint`) differentiates w.r.t. the module's TRAINABLE PARAMETERS as well
    as its inputs (the bare version only handles inputs, so module weights would get
    zero gradient). NOTE: MLX recomputes with the live RNG, so use it with dropout=0
    for exact gradients (the L4-L5 scale where checkpointing matters runs on torch,
    which preserves the RNG)."""
    import mlx.nn.utils as _nnu
    return _nnu.checkpoint(module)(*args)


def _set_seed(seed: int) -> None:
    """Seed every RNG that affects a training run (Python, numpy, MLX), so weight
    init + dropout + sampling are reproducible across runs."""
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


_engine = SimpleNamespace(
    value_and_grad=lambda model, fn: mlx_nn.value_and_grad(model, fn),
    # MLX AdamW keeps its moments in the PARAM dtype (already bf16 when the model is
    # bf16 → ~2 bytes/state). There is no native 8-bit optimizer, so a `states=int8`
    # request is accepted but simply stays bf16 (the saving is on the CUDA backend).
    make_optimizer=lambda model, lr, weight_decay, states=None: mlx_optim.AdamW(
        learning_rate=lr, weight_decay=weight_decay, bias_correction=True),
    optimizer_step=_optimizer_step,
    set_lr=lambda opt, lr: setattr(opt, "learning_rate", lr),
    eval=lambda *xs: mx.eval(*xs) if xs else None,
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
    freeze_all=lambda model: model.freeze(),
    # MLX walks dict/list-of-Module attributes automatically, so attaching the
    # sectors dict already registers their params — nothing extra to do.
    register_submodules=lambda parent, name, modules: None,
    align_module=lambda module, model: None,   # MLX unified memory: no-op
    grad_norm=_grad_norm,
    clip_grads=_clip_grads,
    accumulate_grads=_accumulate_grads,
    finalize_grads=_finalize_grads,
    save_optimizer=_save_optimizer,
    load_optimizer=_load_optimizer,
    param_count=_param_count,
    memory_stats=_memory_stats,
    set_seed=_set_seed,
    checkpoint=_checkpoint,
)


class MLXBackend(Backend):
    name = "mlx"
    nn = _nn
    ops = _ops
    engine = _engine


def build() -> MLXBackend:
    return MLXBackend()
