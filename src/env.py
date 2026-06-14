"""Project environment loading.

Reads key=value pairs from a project-root `.env` into os.environ so secrets and
config (e.g. HF_TOKEN, RDMCA_BACKEND) live in one standard, gitignored place.
`.env.example` is the tracked template — copy it to `.env` and fill in.

Real environment variables always win over `.env` values. Loading is idempotent
and runs once on import. Uses python-dotenv when installed, else a minimal
built-in parser so it works before dependencies are installed.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
_loaded = False


def _minimal_load(path: Path) -> None:
    """Tiny KEY=VALUE parser (no python-dotenv dependency). Real env wins."""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:  # real env takes precedence
            os.environ[key] = val


def load_env(path: Path = ENV_FILE) -> None:
    """Load `.env` into os.environ once. No-op if already loaded or file absent."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv  # standard, if available

        load_dotenv(path, override=False)
    except ImportError:
        _minimal_load(path)


load_env()
