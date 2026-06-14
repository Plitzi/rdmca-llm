"""
Stage plugin contract — the SINGLE shape every curriculum stage follows.

A stage used to be described by scattered dicts (gates, names, rehearsal, lr_scale,
mood set, freeze point) in src/training/stages.py, plus data generators in
src/data/graded.py and structure in the level YAMLs. Now each stage is a
self-contained plugin: one `StagePlugin` instance that carries its metadata AND its
data sources, lives in its own package under src/stages/stageNN_<slug>/, and is
discovered automatically (see registry.py). Adding a stage = drop a new package;
nothing else needs editing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum

# A data source builder yields {"text": str, "lang": str} records for one source key.
# It is sized from the per-(level, stage) token budget; `limit_mb` caps real-corpus
# streaming during data preparation. Synthetic generators ignore `limit_mb`.
SourceBuilder = Callable[..., Iterator[dict]]


class StageKind(str, Enum):
    """Cognitive stages train (and then freeze) the shared core; behavioral stages
    train LoRA sectors on top of the already-frozen core (tool/MCP/skills)."""

    COGNITIVE = "cognitive"
    BEHAVIORAL = "behavioral"


@dataclass(frozen=True)
class StageGate:
    """Graduation bar for a stage. `metric_key` is the benchmark this gate will use
    once wired (today the trainer uses a masked validation-perplexity proxy); the
    `label` is the human description shown in logs and the dashboard."""

    metric_key: str
    threshold: float
    label: str


@dataclass(frozen=True)
class StagePlugin:
    """Everything that defines one curriculum stage, in one place.

    Per-stage anti-forgetting defaults (`rehearsal_fraction`, `lr_scale`) live here
    because they are a property of the stage's NATURE, not of the level — a narrow
    low-entropy stage erodes the shared core the same way at every level. A level's
    YAML may still override them per stage for a genuine exception.
    """

    number: int
    slug: str
    name: str
    entry_level: int
    kind: StageKind
    # Anti-forgetting profile (fraction of replayed batches; LR multiplier).
    rehearsal_fraction: float = 0.15
    lr_scale: float = 1.0
    # Graduation gate (None for behavioral stages, which ship as LoRA sectors).
    gate: StageGate | None = None
    # Whether the mood head is (re)trained when this stage completes — only
    # meaningful at conversational stages (see registry.mood_stages).
    trains_mood: bool = False
    # The core freezes right after the last ACTIVE stage flagged here (the BCF stage).
    is_freeze_point: bool = False
    # A stage can be switched off without deleting it (see registry.enabled_stages
    # and the per-level `curriculum.stageN.enabled` override).
    enabled: bool = True
    # Source key -> builder. The registry resolves a source key by asking the plugin
    # that owns it, replacing the old monolithic graded.stream_source dispatcher.
    sources: dict[str, SourceBuilder] = field(default_factory=dict)

    @property
    def is_behavioral(self) -> bool:
        return self.kind is StageKind.BEHAVIORAL

    @property
    def package(self) -> str:
        """Directory/package name, e.g. 'stage01_language'."""
        return f"stage{self.number:02d}_{self.slug}"
