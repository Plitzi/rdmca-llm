"""
Shared configuration helpers — single source of truth for languages,
config-path resolution and tokenizer/vocab metadata.

The language selector is config-driven: `model.languages` in a profile/config
is the canonical list. At tokenizer-training time the chosen languages (and the
multimodal vocabulary layout) are persisted to dist/tokenizer/tokenizer_info.json,
which every runtime component reads — so there is no language list hardcoded in
the source.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional

import yaml

from .env import load_env

load_env()                              # populate os.environ from the project .env


TOKENIZER_INFO = "dist/tokenizer/tokenizer_info.json"

# Educational LEVELS replace the old hardware profiles. A level's size is set by
# the INFORMATION it teaches (vocab/context/width/depth); the hardware only
# limits how high a level you can run. Levels 1..5 = preescolar..universidad.
LEVELS_DIR = "configs/levels"
DEFAULT_LEVEL = 2                       # primaria — a sensible laptop default


def available_levels() -> List[int]:
    """All level numbers present as `configs/levels/level{N}.yaml`, sorted.
    Single source of truth for the level range — drop in a new `levelN.yaml`
    and it is recognized automatically (no code change needed)."""
    import glob
    import re
    levels = []
    for path in glob.glob(f"{LEVELS_DIR}/level*.yaml"):
        m = re.search(r"level(\d+)\.yaml$", path)
        if m:
            levels.append(int(m.group(1)))
    return sorted(levels)


# Global level bounds, derived from the configs present (fallbacks if none yet).
_LEVELS = available_levels()
MIN_LEVEL = _LEVELS[0] if _LEVELS else 0   # level 0 = throwaway smoke/test tier
MAX_LEVEL = _LEVELS[-1] if _LEVELS else 5
DEFAULT_CONFIG = f"{LEVELS_DIR}/level{DEFAULT_LEVEL}.yaml"

# Single source of truth: the registry's builder map. Adding a backend there now
# updates this automatically (no second list to keep in sync).
from src.backend.registry import available as _available_backends
SUPPORTED_BACKENDS = _available_backends()
SUPPORTED_PRECISIONS = ("fp32", "bf16", "fp16")


def level_config_path(level: int) -> str:
    """Path to a level's config YAML (configs/levels/level{N}.yaml)."""
    return f"{LEVELS_DIR}/level{int(level)}.yaml"


def resolve_config_path(config: Optional[str] = None,
                        level: Optional[int] = None) -> str:
    """A `--level N` wins over an explicit `--config path`; else the default
    level. Levels replace the old `--profile` selector."""
    if level is not None:
        lvl = int(level)
        levels = available_levels()
        if lvl not in levels:
            raise ValueError(
                f"Level {lvl} not found — available: {levels}.")
        return level_config_path(lvl)
    return config or DEFAULT_CONFIG


def get_level(cfg: dict) -> Optional[int]:
    """The level number declared in a config, or None for a custom config."""
    lvl = cfg.get("level")
    return int(lvl) if lvl is not None else None


BASE_CONFIG_NAME = "_base.yaml"          # shared defaults every level inherits


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` INTO a copy of `base` — the override always wins.
    Dicts merge key-by-key (so a level can set just curriculum.stage3.n_tokens without
    redeclaring the stage); lists and scalars REPLACE wholesale (a level's `sources: […]`
    fully replaces the base's). Used for level-config inheritance from _base.yaml."""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str) -> dict:
    """Load a level config, deep-merged OVER the shared `_base.yaml` in the same dir (if
    present), so each level declares only what DIFFERS — the shared training/model/moe
    defaults and the curriculum structure (stage names + entry_levels) live once in the
    base. The level's values always win. A config opts out with `inherit_base: false`
    (e.g. the level-0 smoke tier, which has its own minimal structure)."""
    p = Path(path)
    with open(p) as f:
        cfg = yaml.safe_load(f) or {}
    base_path = p.parent / BASE_CONFIG_NAME
    if (cfg.get("inherit_base", True) and base_path.exists()
            and p.resolve() != base_path.resolve()):
        with open(base_path) as f:
            base = yaml.safe_load(f) or {}
        base.pop("inherit_base", None)
        cfg = _deep_merge(base, cfg)
    cfg.pop("inherit_base", None)
    return cfg


def get_languages(cfg: dict) -> List[str]:
    """Canonical language list for a config. Defaults to ['en']."""
    langs = (cfg.get("model", {}) or {}).get("languages")
    return list(langs) if langs else ["en"]


# ---------------------------------------------------------------------------
# Compute backend (mlx | torch) and training/inference precision
# ---------------------------------------------------------------------------

def get_backend(cfg: dict) -> str:
    """Selected compute backend. Top-level `backend:` key, default 'mlx'."""
    return (cfg.get("backend") or "mlx").lower()


def require_backend(cfg: dict) -> str:
    """
    Activate and return the configured compute backend (mlx | torch). Selects
    it in `src.backend` so subsequently-imported model modules bind to it, then
    fails loudly on an unknown name. Call this BEFORE importing model modules.
    """
    name = get_backend(cfg)
    if name not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown backend '{name}'. Supported: {', '.join(SUPPORTED_BACKENDS)}")
    import src.backend as backend
    backend.select(name)            # may fall back (e.g. mlx → torch) if unavailable
    return backend.name()           # the backend actually activated


def get_precision(cfg: dict) -> str:
    """Training/inference precision from `training.precision`, default 'bf16'."""
    prec = ((cfg.get("training", {}) or {}).get("precision") or "bf16").lower()
    if prec not in SUPPORTED_PRECISIONS:
        raise ValueError(
            f"Unknown precision '{prec}'. Supported: {', '.join(SUPPORTED_PRECISIONS)}")
    return prec


def load_tokenizer_info(path: str = TOKENIZER_INFO) -> Optional[dict]:
    """Return persisted tokenizer/vocab metadata, or None if not trained yet."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def unified_vocab_size(info: Optional[dict], fallback: int) -> int:
    """
    Total embedding vocabulary = text ∪ image ∪ audio. `vocab_size` in
    tokenizer_info is already the unified total once modalities are registered;
    older info files only carry the text size, which is still correct for a
    text-only deployment.
    """
    if not info:
        return fallback
    return int(info.get("vocab_size", fallback))
