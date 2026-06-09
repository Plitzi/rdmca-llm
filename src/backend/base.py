"""
Backend interface — the single seam every compute backend implements.

A `Backend` bundles three namespaces so the model/training code can be written
exactly once against the active backend:

  - `nn`      neural-net building blocks (Module base + layer factories)
  - `ops`     tensor functions, normalized to MLX-style signatures
              (`axis=`, `keepdims=`) since MLX is the reference implementation
  - `engine`  runtime/training glue (autograd, optimizer, checkpoints, …)

Adding a new backend (e.g. JAX) = subclass `Backend`, fill the three
namespaces, and register it in `registry.py`. No model code changes.

The model is written once against `B = src.backend.current()` and uses
`B.nn.*` / `B.ops.*`; the entrypoints drive training through `B.engine.*`.
"""
from __future__ import annotations
from types import SimpleNamespace


class Backend:
    """Base class. Concrete backends set `name` and the three namespaces.

    Namespaces are plain `SimpleNamespace` objects populated by each backend
    module; we keep them duck-typed rather than over-specifying an ABC, so a
    backend only has to provide what the model actually uses (the inventoried
    surface — see the docstrings in `mlx_backend` / `torch_backend`)."""

    name: str = "base"
    nn: SimpleNamespace
    ops: SimpleNamespace
    engine: SimpleNamespace

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Backend {self.name!r}>"


# Canonical method/function surface each backend is expected to provide. Kept as
# documentation + a light self-check used by the test-suite, not as hard ABCs.
NN_SURFACE = (
    "Module", "Linear", "Embedding", "Dropout",
    "Conv1d", "Conv2d", "ConvTranspose1d", "ConvTranspose2d",
    "Parameter", "ModuleList", "ModuleDict",
)

OPS_SURFACE = (
    "array", "arange", "zeros", "ones", "full", "randn",
    "cos", "sin", "sqrt", "mean", "sum", "concatenate", "outer",
    "softmax", "sigmoid", "triu", "argmax", "argmin",
    "transpose", "astype", "stop_gradient",
    "silu", "relu", "cross_entropy", "bce_with_logits",
    "to_numpy", "from_numpy",
    "float32", "bfloat16", "float16",
)

ENGINE_SURFACE = (
    "value_and_grad", "make_optimizer", "optimizer_step", "set_lr",
    "eval", "item", "set_precision", "set_train", "set_eval",
    "save_weights", "load_weights", "state_dict", "load_state_dict",
    "set_trainable", "freeze_all", "register_submodules",
    "grad_norm", "param_count", "memory_stats",
)


def check_surface(backend: Backend) -> list[str]:
    """Return the list of missing attributes (empty == complete). Used by tests
    to catch a backend that forgot to implement part of the contract."""
    missing = []
    for name in NN_SURFACE:
        if not hasattr(backend.nn, name):
            missing.append(f"nn.{name}")
    for name in OPS_SURFACE:
        if not hasattr(backend.ops, name):
            missing.append(f"ops.{name}")
    for name in ENGINE_SURFACE:
        if not hasattr(backend.engine, name):
            missing.append(f"engine.{name}")
    return missing
