"""
Stage registry — discovers the stage plugins and answers every "what stages exist
and how do they behave?" question that the trainer, dashboard and data pipeline
used to answer from scattered dicts.

Discovery is automatic: every sub-package named `stageNN_<slug>` that exposes a
`PLUGIN` is loaded. Drop a new `src/stages/stage11_*/` package and it joins the
curriculum with no other edits.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
from collections.abc import Iterator

from src.stages.base import StagePlugin

_PACKAGE_RE = re.compile(r"^stage(\d+)_")
_CACHE: dict[int, StagePlugin] | None = None


def _discover() -> dict[int, StagePlugin]:
    """Import every stageNN_* sub-package and collect its PLUGIN, keyed by number.
    Validates that numbers are unique and that at most one stage is the freeze point."""
    import src.stages as pkg

    found: dict[int, StagePlugin] = {}
    for mod in pkgutil.iter_modules(pkg.__path__):
        if not _PACKAGE_RE.match(mod.name):
            continue
        plugin = importlib.import_module(f"src.stages.{mod.name}").PLUGIN
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
    global _CACHE
    if _CACHE is None:
        _CACHE = _discover()
    return _CACHE


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
def bcf_stage() -> int:
    """The Behavioral-Cognitive Freeze stage: the latest possible freeze point."""
    for plugin in all_stages():
        if plugin.is_freeze_point:
            return plugin.number
    raise ValueError("no freeze-point stage declared")


def is_behavioral(number: int) -> bool:
    return get_stage(number).is_behavioral


def mood_stages() -> set[int]:
    """Stages whose completion (re)trains the mood head."""
    return {p.number for p in all_stages() if p.trains_mood}


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
    """Where a stage's prepared corpus lives. Each stage OWNS its data folder inside
    its own package (`src/stages/stageNN_<slug>/data/level{L}/`), so a stage is fully
    self-contained. A per-level `curriculum.stageN.data_dir` override still wins."""
    stage_cfg = ((cfg or {}).get("curriculum", {}) or {}).get(f"stage{number}", {}) or {}
    if stage_cfg.get("data_dir"):
        return stage_cfg["data_dir"]
    package = get_stage(number).package if has_stage(number) else f"stage{number:02d}"
    base = f"src/stages/{package}/data"
    level = (cfg or {}).get("level")
    return f"{base}/level{level}" if level is not None else f"{base}/default"
