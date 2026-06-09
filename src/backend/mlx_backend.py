"""
MLX backend — wraps mlx.core / mlx.nn / mlx.optimizers behind the Backend
facade. This is the reference implementation; `ops` signatures are MLX-native
(`axis=`, `keepdims=`) and the Torch backend translates to match.
"""
from __future__ import annotations
from types import SimpleNamespace

import numpy as np
import mlx.core as mx
import mlx.nn as mlx_nn
import mlx.optimizers as mlx_optim
from mlx.utils import tree_flatten, tree_map

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
    triu=lambda x, k=0: mx.triu(x, k=k),
    argmax=lambda x, axis=-1: mx.argmax(x, axis=axis),
    argmin=lambda x, axis=-1: mx.argmin(x, axis=axis),
    transpose=lambda x, axes: mx.transpose(x, axes),
    astype=lambda x, dtype: x.astype(dtype),
    stop_gradient=mx.stop_gradient,
    silu=mlx_nn.silu,
    relu=mlx_nn.relu,
    cross_entropy=lambda logits, targets, reduction="mean": mlx_nn.losses.cross_entropy(
        logits, targets, reduction=reduction),
    bce_with_logits=lambda logits, labels, reduction="mean": mlx_nn.losses.binary_cross_entropy(
        logits, labels, with_logits=True, reduction=reduction),
    to_numpy=lambda x: np.array(x),
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


def _optimizer_step(opt, model, grads):
    opt.update(model, grads)
    mx.eval(model.parameters(), opt.state)


def _grad_norm(model, grads) -> float:
    sq = 0.0
    for _, g in tree_flatten(grads):
        if isinstance(g, mx.array) and g.size > 0:
            sq += float((g * g).sum().item())
    return sq ** 0.5


def _save_weights(model, path: str) -> None:
    """Neutral checkpoint: a .npz of float32 numpy arrays keyed by param name.
    Loadable by any backend (same names) regardless of training precision."""
    flat = {k: np.array(v.astype(mx.float32)) for k, v in tree_flatten(model.parameters())}
    np.savez(str(path), **flat)


def _load_weights(model, path: str) -> None:
    data = np.load(str(path))
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


_engine = SimpleNamespace(
    value_and_grad=lambda model, fn: mlx_nn.value_and_grad(model, fn),
    make_optimizer=lambda model, lr, weight_decay: mlx_optim.AdamW(
        learning_rate=lr, weight_decay=weight_decay),
    optimizer_step=_optimizer_step,
    set_lr=lambda opt, lr: setattr(opt, "learning_rate", lr),
    eval=lambda *xs: mx.eval(*xs) if xs else None,
    item=lambda x: float(x.item()),
    set_precision=_set_precision,
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
    grad_norm=_grad_norm,
    param_count=_param_count,
    memory_stats=_memory_stats,
)


class MLXBackend(Backend):
    name = "mlx"
    nn = _nn
    ops = _ops
    engine = _engine


def build() -> MLXBackend:
    return MLXBackend()
