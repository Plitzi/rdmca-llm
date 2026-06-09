"""
Backend registry — maps a backend name to its builder. Builders are imported
lazily so a backend's heavy dependency (e.g. torch) is only required when that
backend is actually selected.

To add a third backend: write `src/backend/<name>_backend.py` exposing
`build() -> Backend`, then add one entry here.
"""
from __future__ import annotations
from typing import Callable, Dict

from src.backend.base import Backend


def _build_mlx() -> Backend:
    from src.backend import mlx_backend
    return mlx_backend.build()


def _build_torch() -> Backend:
    from src.backend import torch_backend
    return torch_backend.build()


BUILDERS: Dict[str, Callable[[], Backend]] = {
    "mlx": _build_mlx,
    "torch": _build_torch,
}


def available() -> tuple[str, ...]:
    return tuple(BUILDERS.keys())


def build(name: str) -> Backend:
    name = (name or "").lower()
    if name not in BUILDERS:
        raise ValueError(
            f"Unknown backend '{name}'. Available: {', '.join(BUILDERS)}")
    return BUILDERS[name]()
