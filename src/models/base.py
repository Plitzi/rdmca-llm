"""
Stage plugin contract — the SINGLE shape every curriculum stage follows.

A stage used to be described by scattered dicts (gates, names, rehearsal, lr_scale,
mood set, freeze point) in src/training/stages.py, plus data generators in
src/data/graded.py and structure in the level YAMLs. Now each stage is a
self-contained plugin: one `StagePlugin` instance that carries its metadata AND its
data sources, lives in its own package under src/models/<model>/stageNN_<slug>/, and is
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
    # Does this stage train the SHARED cognitive core that is frozen once the base
    # curriculum completes? Each stage declares this EXPLICITLY (no number/threshold
    # rule). True → cognitive (part of the frozen base); False → behavioral, i.e. it
    # trains a LoRA sector on top of the already-frozen core (tool/MCP/skills).
    frozen_base: bool
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
        """A stage that is NOT part of the frozen base trains a LoRA sector on it."""
        return not self.frozen_base

    @property
    def kind(self) -> StageKind:
        """Human-readable kind, derived from the stage's `frozen_base` declaration."""
        return StageKind.COGNITIVE if self.frozen_base else StageKind.BEHAVIORAL

    @property
    def package(self) -> str:
        """Directory/package name, e.g. 'stage01_language'."""
        return f"stage{self.number:02d}_{self.slug}"


@dataclass(frozen=True)
class ModelSpec:
    """How the framework BUILDS, TRAINS and EVALUATES one model — the seam that makes
    the engine task/modality-agnostic. The default (a text LLM) lives in
    `src/core/training/model_spec.py`; a model package overrides it by exposing a module-
    level `SPEC = ModelSpec(...)` (or `build_spec(cfg) -> ModelSpec`), so the same
    trainer can train, say, a hand-pose model instead of a conversational LLM.

    Callables (all receive what the trainer has at that point):
      • build_model(stage, cfg, root) -> (model, model_cfg, adapter, precision, seed)
      • build_loader(stage, cfg)      -> a loader with .next_batch()/telemetry
      • objective(model, batch)        -> scalar training loss (INCLUDING any aux term)
      • evaluate(model, stage, val_batches, cfg, **kw) -> (score, passed); by convention
        LOWER `score` is better (a higher-is-better metric returns e.g. 1-accuracy), so
        the trainer's ratchet/early-stop stays metric-agnostic.

    `gate_metric` is the human label for what `evaluate` measures (e.g. "perplexity",
    "pck").
    """

    name: str
    build_model: Callable
    build_loader: Callable
    objective: Callable
    evaluate: Callable
    gate_metric: str = "perplexity"
