"""
DEPRECATED compatibility shim.

The graded-data generators now live WITH their stage, as plugins under
`src/stages/stageNN_<slug>/sources.py`, with cross-stage helpers in
`src/stages/_shared/`. The source dispatcher is the stage registry
(`src.stages.stream_source`).

This module re-exports the old public + private names from their new homes so
existing importers (a few tests, scripts/prepare_data.py) keep working unchanged.
New code should import from `src.stages` / the per-stage `sources` modules.
"""

from __future__ import annotations

# Source dispatcher → registry.
from src.stages import stream_source  # noqa: F401

# Shared helpers.
from src.stages._shared.blend import (  # noqa: F401
    _blend,
    _cycle_records,
    _interleave,
    blend,
    cycle_records,
    interleave,
)
from src.stages._shared.dictionary import (  # noqa: F401
    _DICT_TIER1,
    _DICT_TIER2,
    _DICT_TIERS,
)
from src.stages._shared.persona import (  # noqa: F401
    _STORY_PROMPTS,
    _hash01,
    _persona_for,
    _prepend_system,
)
from src.stages._shared.text import (  # noqa: F401
    _stable_hash,
    flesch_kincaid_grade,
    passes_filter,
)

# Stage 1 — language.
from src.stages.stage01_language.sources import (  # noqa: F401
    _COMPARATIVE,
    _DIALOGUE_CORPORA,
    _PAST_IRREG,
    _PLURAL_IRREG,
    _format_dialogue,
    _stream_empathetic_balanced,
    gen_basic_chat,
    gen_definitions,
    gen_grammar,
    stream_dialogue,
    stream_instruct,
    stream_simple_wikipedia,
    stream_tinystories,
)

# Stage 2 — perception.
from src.stages.stage02_perception.sources import (  # noqa: F401
    _ANALOGY_RELATIONS,
    gen_analogies,
)

# Stage 3 — abstraction (arithmetic).
from src.stages.stage03_abstraction.sources import (  # noqa: F401
    _add_worked,
    _sub_worked,
    gen_arithmetic,
    stream_arithmetic,
)

# Stage 4 — causal.
from src.stages.stage04_causal.sources import gen_causal, stream_causal  # noqa: F401

# Stage 5 — reasoning.
from src.stages.stage05_reasoning.sources import gen_cot, stream_reasoning  # noqa: F401

# Stage 6 — memory.
from src.stages.stage06_memory.sources import gen_memory  # noqa: F401

# Stage 7 — ethics.
from src.stages.stage07_ethics.sources import gen_ethics  # noqa: F401

# Stage 8/9/10 — behavioral.
from src.stages.stage08_tools.sources import stream_agentic  # noqa: F401
from src.stages.stage09_mcp.sources import stream_mcp  # noqa: F401
from src.stages.stage10_skills.sources import stream_skills  # noqa: F401
