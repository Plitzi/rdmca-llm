"""
Stage registry — discovers the stage plugins of the ACTIVE MODEL and answers every
"what stages exist and how do they behave?" question the trainer, dashboard and data
pipeline ask.

A **model** is one training scenario living under `models/<name>/` (e.g.
`cognition` = the conversational/agentic LLM curriculum; `hands_recognition` = a VR
hand-pose model). Discovery is automatic within the active model: every sub-package
named `stageNN_<slug>` under `models/<name>/stages/` exposing a `PLUGIN` is loaded. Drop a
new `models/<name>/stages/stage11_*/` and it joins that model's curriculum with no other
edits. `set_active_model(name)` (driven by `cfg["model_name"]`) switches the active model.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
from collections.abc import Iterator
from pathlib import Path

from src.plugins.base import StagePlugin

_PACKAGE_RE = re.compile(r"^stage(\d+)_")
DEFAULT_MODEL = "cognition"
_ACTIVE_MODEL = DEFAULT_MODEL
_CACHE: dict[str, dict[int, StagePlugin]] = {}


def set_active_model(name: str | None) -> None:
    """Select the active model (the `models/<name>/` package whose stages the
    framework trains). None / "" keeps the default. Idempotent."""
    global _ACTIVE_MODEL
    if name:
        _ACTIVE_MODEL = name


def active_model() -> str:
    return _ACTIVE_MODEL


def model_hook(name: str):
    """An OPTIONAL model-provided callable looked up by name on the active model package
    (e.g. `post_stage`), or None. Lets the agnostic core invoke model-specific side
    effects without importing the model — the same discovery pattern as the ModelSpec."""
    pkg = importlib.import_module(f"models.{_ACTIVE_MODEL}")
    return getattr(pkg, name, None)


def _models_root() -> Path:
    """The repo's models/ directory (this file is src/plugins/registry.py → repo is 3 up)."""
    return Path(__file__).resolve().parents[2] / "models"


def available_models() -> list[str]:
    """Every model package under models/ — a directory with an __init__.py. The SINGLE
    source of truth for "what models exist" (the CLI and trainer both ask here)."""
    root = _models_root()
    if not root.is_dir():
        return []
    return sorted(
        child.name
        for child in root.iterdir()
        if child.is_dir() and (child / "__init__.py").exists()
    )


def model_uses(model: str) -> dict[str, Path]:
    """A model's runnable USE CASES: {app: run_<app>.py} for each models/<model>/uses/<app>/
    that ships a run_<app>.py. Use cases belong to the model, NOT the framework — so they
    are discovered here per model (common/, tests/ and stub dirs lack a run_ file and are
    skipped) rather than hardcoded in the CLI. Empty for a model with no uses/ yet."""
    found: dict[str, Path] = {}
    uses_dir = _models_root() / model / "uses"
    if uses_dir.is_dir():
        for child in sorted(uses_dir.iterdir()):
            run = child / f"run_{child.name}.py"
            if child.is_dir() and run.exists():
                found[child.name] = run
    return found


def _discover(model: str) -> dict[int, StagePlugin]:
    """Import every stageNN_* sub-package of `models/<model>/stages/` and collect its
    PLUGIN, keyed by number. Stages are grouped under `stages/` so the model's base
    folder stays tidy. Validates unique numbers and ≤1 freeze point."""
    pkg = importlib.import_module(f"models.{model}.stages")

    found: dict[int, StagePlugin] = {}
    for mod in pkgutil.iter_modules(pkg.__path__):
        if not _PACKAGE_RE.match(mod.name):
            continue
        plugin = importlib.import_module(f"models.{model}.stages.{mod.name}").PLUGIN
        if plugin.number in found:
            raise ValueError(
                f"duplicate stage number {plugin.number}: "
                f"{found[plugin.number].package} vs {plugin.package}"
            )
        found[plugin.number] = plugin
    freeze_points = [p.number for p in found.values() if p.is_freeze_point]
    if len(freeze_points) > 1:
        raise ValueError(f"more than one freeze-point stage declared: {freeze_points}")
    return dict(sorted(found.items()))


def _registry() -> dict[int, StagePlugin]:
    if _ACTIVE_MODEL not in _CACHE:
        _CACHE[_ACTIVE_MODEL] = _discover(_ACTIVE_MODEL)
    return _CACHE[_ACTIVE_MODEL]


# ── lookups ──────────────────────────────────────────────────────────────────
def all_stages() -> list[StagePlugin]:
    """Every declared stage, ordered by number (includes disabled ones)."""
    return list(_registry().values())


def get_stage(number: int) -> StagePlugin:
    return _registry()[number]


def has_stage(number: int) -> bool:
    return number in _registry()


def enabled_stages() -> list[StagePlugin]:
    """Stages not switched off via their `enabled` flag."""
    return [p for p in all_stages() if p.enabled]


def active_stages(level: int) -> list[StagePlugin]:
    """Enabled stages whose entry_level is at or below `level` — the curriculum a
    given level actually runs."""
    return [p for p in enabled_stages() if p.entry_level <= level]


# ── freeze point / kinds ───────────────────────────────────────────────────────
def bcf_stage() -> int | None:
    """The Behavioral-Cognitive Freeze stage: the model's freeze point, or None when
    the model declares none (e.g. a single-stage, non-conversational scenario that
    never freezes a cognitive core)."""
    for plugin in all_stages():
        if plugin.is_freeze_point:
            return plugin.number
    return None


def is_behavioral(number: int) -> bool:
    return get_stage(number).is_behavioral


# ── data sources ───────────────────────────────────────────────────────────────
def stream_source(
    key: str,
    *,
    langs: list[str],
    n_tokens: int,
    arithmetic_level: int = 1,
    limit_mb: int | None = None,
    extra_streamers: dict | None = None,
) -> Iterator[dict] | None:
    """Resolve a source key to {'text','lang'} records by asking the stage that owns
    it (replacing the old monolithic graded.stream_source dispatcher). Synthetic
    generators are sized from the token budget (~6 tokens per short example). The
    full real corpora (wikipedia/arc/gsm8k/math) are supplied by the data-prep
    pipeline via `extra_streamers`; a key owned by no stage falls through to those.
    Returns None for an unknown key."""
    approx_examples = max(n_tokens // 6, 1000)
    for plugin in all_stages():
        builder = plugin.sources.get(key)
        if builder is not None:
            return builder(
                langs=langs,
                n_tokens=n_tokens,
                arithmetic_level=arithmetic_level,
                limit_mb=limit_mb,
                extra_streamers=extra_streamers,
                approx_examples=approx_examples,
            )
    if extra_streamers and key in extra_streamers:
        return extra_streamers[key]()
    return None


def owns_source(key: str) -> StagePlugin | None:
    for plugin in all_stages():
        if key in plugin.sources:
            return plugin
    return None


# ── data location ───────────────────────────────────────────────────────────
def stage_data_dir(number: int, cfg: dict | None = None) -> str:
    """Where a stage's prepared corpus lives. A model keeps ALL its stages' corpora under
    a single per-model folder (`models/<model>/data/<package>/level{L}/`) — one place to
    find/clean the data instead of hunting a `data/` inside every stage package. A
    per-level `curriculum.stageN.data_dir` override wins."""
    stage_cfg = ((cfg or {}).get("curriculum", {}) or {}).get(f"stage{number}", {}) or {}
    if stage_cfg.get("data_dir"):
        return stage_cfg["data_dir"]
    package = get_stage(number).package if has_stage(number) else f"stage{number:02d}"
    base = f"models/{_ACTIVE_MODEL}/data/{package}"
    level = (cfg or {}).get("level")
    return f"{base}/level{level}" if level is not None else f"{base}/default"
