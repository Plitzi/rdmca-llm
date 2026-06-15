"""
Shared configuration helpers — single source of truth for languages,
config-path resolution and tokenizer/vocab metadata.

The language selector is config-driven: `model.languages` in a profile/config
is the canonical list. At tokenizer-training time the chosen languages (and the
multimodal vocabulary layout) are persisted to dist/<model>/tokenizer/tokenizer_info.json,
which every runtime component reads — so there is no language list hardcoded in
the source.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from .env import load_env

load_env()  # populate os.environ from the project .env


# ── Per-model build artifacts (dist/<model>/…) ──────────────────────────────
# Everything a run PRODUCES — tokenizer, checkpoints, snapshots, benchmarks — lives
# under a PER-MODEL root so two models never clobber each other's dist. Single source
# of truth for the on-disk layout; defaults to the active model (set via select_model).
def model_dist_root(model: str | None = None) -> Path:
    from src.plugins import active_model

    return Path("dist") / (model or active_model())


def tokenizer_dir(model: str | None = None) -> Path:
    """Where a model's tokenizer assets live: dist/<model>/tokenizer/."""
    return model_dist_root(model) / "tokenizer"


def tokenizer_model_path(model: str | None = None) -> Path:
    return tokenizer_dir(model) / "rdmca_spm.model"


def tokenizer_info_path(model: str | None = None) -> Path:
    return tokenizer_dir(model) / "tokenizer_info.json"


# Educational LEVELS replace the old hardware profiles. A level's size is set by
# the INFORMATION it teaches (vocab/context/width/depth); the hardware only
# limits how high a level you can run. Levels 1..5 = preescolar..universidad.
#
# Level configs are PER-MODEL: each model owns its ladder under
# models/<model>/configs/levels/, so multiple models never share one global folder.
# The repeated boilerplate (training cadence, size tiers) is factored into the level
# CONSTRUCTOR (src/levels.py), pulled in by a config's `tier:` key.
DEFAULT_LEVEL = 2  # primaria — a sensible laptop default
DEFAULT_MODEL = "cognition"  # the default model whose levels back the module-level bounds


def level_config_dir(model: str | None = None) -> Path:
    """Where a model's level configs live: models/<model>/configs/levels/ (active model
    by default). Single source of truth for the per-model level layout."""
    if model is None:
        from src.plugins import active_model

        model = active_model()
    return Path("models") / model / "configs" / "levels"


def available_levels(model: str | None = None) -> list[int]:
    """All level numbers present as `models/<model>/configs/levels/level{N}.yaml`, sorted.
    Drop in a new `levelN.yaml` under a model and it is recognized automatically."""
    import re

    levels = []
    directory = level_config_dir(model)
    if directory.is_dir():
        for path in directory.glob("level*.yaml"):
            m = re.match(r"level(\d+)\.yaml$", path.name)
            if m:
                levels.append(int(m.group(1)))
    return sorted(levels)


def level_config_path(level: int, model: str | None = None) -> str:
    """Path to a level's config YAML (models/<model>/configs/levels/level{N}.yaml)."""
    return str(level_config_dir(model) / f"level{int(level)}.yaml")


# Global level bounds, derived from the DEFAULT model's configs (fallbacks if none yet).
_LEVELS = available_levels(DEFAULT_MODEL)
MIN_LEVEL = _LEVELS[0] if _LEVELS else 0  # level 0 = throwaway smoke/test tier
MAX_LEVEL = _LEVELS[-1] if _LEVELS else 5
DEFAULT_CONFIG = level_config_path(DEFAULT_LEVEL, DEFAULT_MODEL)

# Single source of truth: the registry's builder map. Adding a backend there now
# updates this automatically (no second list to keep in sync).
from src.backend.registry import available as _available_backends

SUPPORTED_BACKENDS = _available_backends()
SUPPORTED_PRECISIONS = ("fp32", "bf16", "fp16")


def resolve_config_path(
    config: str | None = None, level: int | None = None, model: str | None = None
) -> str:
    """A `--level N` (resolved under `model`'s own levels) wins over an explicit
    `--config path`; else the default level. Levels are per-model, so the model picks
    which ladder `--level` indexes."""
    if level is not None:
        lvl = int(level)
        levels = available_levels(model)
        if lvl not in levels:
            who = model or DEFAULT_MODEL
            raise ValueError(f"Level {lvl} not found for model '{who}' — available: {levels}.")
        return level_config_path(lvl, model)
    return config or DEFAULT_CONFIG


def get_level(cfg: dict) -> int | None:
    """The level number declared in a config, or None for a custom config."""
    lvl = cfg.get("level")
    return int(lvl) if lvl is not None else None


def select_model(cfg: dict, override: str | None = None) -> str:
    """Activate the model whose stages a CLI run targets and return its name. A CLI
    `--model` override wins over the config's `model_name` (registry default otherwise).
    The SINGLE place scripts choose the model, so stage discovery + data/checkpoint
    paths all resolve under it."""
    from src.plugins import active_model, set_active_model

    set_active_model(override or cfg.get("model_name"))
    return active_model()


BASE_CONFIG_NAME = "_base.yaml"  # shared defaults every level inherits


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
    """Load a level config and layer the shared defaults UNDER it (the config always wins),
    so each level declares only what DIFFERS. Two independent layers, lowest precedence first:

      1. `tier: <name>` → the level CONSTRUCTOR's scaffold (src/levels.py) — the boilerplate
         repeated across models (training cadence, size tier, resource estimates);
      2. `inherit_base: true` (default) → the sibling `_base.yaml` (the MODEL's shared base —
         e.g. cognition's curriculum structure + moe defaults).

    A config opts out of the base with `inherit_base: false` (e.g. a model whose base differs
    entirely); `tier` is independent of that. Unknown/absent tier contributes nothing."""
    from src.levels import scaffold

    p = Path(path)
    with open(p) as f:
        cfg = yaml.safe_load(f) or {}

    merged: dict = scaffold(cfg.get("tier"))  # layer 1 (lowest precedence)
    base_path = p.parent / BASE_CONFIG_NAME
    if cfg.get("inherit_base", True) and base_path.exists() and p.resolve() != base_path.resolve():
        with open(base_path) as f:
            base = yaml.safe_load(f) or {}
        base.pop("inherit_base", None)
        merged = _deep_merge(merged, base)  # layer 2 (base wins over scaffold)
    merged = _deep_merge(merged, cfg)  # the level config always wins
    merged.pop("inherit_base", None)
    return merged


def get_languages(cfg: dict) -> list[str]:
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
        raise ValueError(f"Unknown backend '{name}'. Supported: {', '.join(SUPPORTED_BACKENDS)}")
    import src.backend as backend

    backend.select(name)  # may fall back (e.g. mlx → torch) if unavailable
    return backend.name()  # the backend actually activated


def get_precision(cfg: dict) -> str:
    """Training/inference precision from `training.precision`, default 'bf16'."""
    prec = ((cfg.get("training", {}) or {}).get("precision") or "bf16").lower()
    if prec not in SUPPORTED_PRECISIONS:
        raise ValueError(
            f"Unknown precision '{prec}'. Supported: {', '.join(SUPPORTED_PRECISIONS)}"
        )
    return prec


def load_tokenizer_info(path: str | None = None) -> dict | None:
    """Return persisted tokenizer/vocab metadata, or None if not trained yet.
    Defaults to the active model's tokenizer_info.json (dist/<model>/tokenizer/)."""
    p = Path(path) if path is not None else tokenizer_info_path()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def unified_vocab_size(info: dict | None, fallback: int) -> int:
    """
    Total embedding vocabulary = text ∪ image ∪ audio. `vocab_size` in
    tokenizer_info is already the unified total once modalities are registered;
    older info files only carry the text size, which is still correct for a
    text-only deployment.
    """
    if not info:
        return fallback
    return int(info.get("vocab_size", fallback))
