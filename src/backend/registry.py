"""
Backend registry — maps a backend name to its builder. Builders are imported
lazily so a backend's heavy dependency (e.g. torch) is only required when that
backend is actually selected.

To add a third backend: write `src/backend/<name>_backend.py` exposing
`build() -> Backend`, then add one entry here.
"""
from __future__ import annotations
import importlib.util
from typing import Callable, Dict

from src.backend.base import Backend

# Lightweight import probe per backend: the module whose presence means the
# backend can be built. `find_spec` checks installability WITHOUT importing the
# heavy package, so we can fall back (e.g. mlx → torch on Linux) BEFORE a fatal
# `import mlx.core` ever runs.
_PROBE: Dict[str, str] = {"mlx": "mlx.core", "torch": "torch"}


def is_available(name: str) -> bool:
    """True if backend `name` can be imported here (no heavy import performed)."""
    mod = _PROBE.get((name or "").lower())
    if not mod:
        return False
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):       # parent package itself missing/broken
        return False


def resolve(name: str) -> str:
    """Return the backend to actually build: `name` if importable here, else an
    available fallback (the other backend). Raises if none is installed. This is
    the pre-check that turns a fatal `import mlx` on Linux into a torch fallback."""
    name = (name or "").lower()
    if name not in BUILDERS:
        raise ValueError(f"Unknown backend '{name}'. Available: {', '.join(BUILDERS)}")
    if is_available(name):
        return name
    for alt in BUILDERS:                    # deterministic fallback order
        if alt != name and is_available(alt):
            return alt
    raise ImportError(
        f"Backend '{name}' is not importable and no fallback backend is installed. "
        f"Install one of: {', '.join(BUILDERS)} (e.g. `pip install torch`).")


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
