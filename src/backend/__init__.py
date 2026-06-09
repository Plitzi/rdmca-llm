"""
Compute-backend selection — the single seam between the model code and the
underlying framework (MLX on Apple Silicon, PyTorch on CUDA/MPS/CPU).

Usage:
    import src.backend as backend
    backend.select("torch")          # do this BEFORE importing model modules
    B = backend.current()
    class Foo(B.nn.Module): ...

The model/training code is written once against `B.nn` / `B.ops` / `B.engine`.

IMPORTANT ordering: the base class `B.nn.Module` is resolved when a model class
is *defined*, so `select()` must run before model modules are imported.
Entrypoints call `select(get_backend(cfg))` as their first action and import
model modules afterwards (function-local imports). If no explicit selection is
made, the first `current()` lazily picks a default (MLX if importable, else
torch) so ad-hoc scripts and tests still work.
"""
from __future__ import annotations
import os
from typing import Optional

from src.backend.base import Backend
from src.backend import registry

_active: Optional[Backend] = None


def _default_name() -> str:
    """Default backend when none was explicitly selected. Honors the
    RDMCA_BACKEND env var, else prefers MLX (importable only on Apple Silicon),
    else torch."""
    env = os.environ.get("RDMCA_BACKEND")
    if env:
        return env.lower()
    try:
        import mlx.core  # noqa: F401
        return "mlx"
    except ImportError:
        return "torch"


def select(name: str) -> Backend:
    """Activate a backend by name. Returns the active Backend. Re-selecting the
    same backend is a no-op; switching after model modules are imported is
    unsupported (their base class is already bound) and emits a warning."""
    global _active
    name = (name or "").lower()
    if _active is not None and _active.name == name:
        return _active
    if _active is not None and _active.name != name:
        import warnings
        warnings.warn(
            f"Switching backend {_active.name!r} -> {name!r} after it was already "
            "active; model classes already imported remain bound to the old "
            "backend. Select the backend before importing model modules.",
            stacklevel=2)
    _active = registry.build(name)
    return _active


def current() -> Backend:
    """Return the active backend, lazily selecting a default if none was set."""
    global _active
    if _active is None:
        _active = registry.build(_default_name())
    return _active


def is_selected() -> bool:
    return _active is not None


def name() -> str:
    return current().name
