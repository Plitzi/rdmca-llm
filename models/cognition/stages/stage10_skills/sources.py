"""Stage 10 data sources — skills.

Real procedures (Super-NaturalInstructions) serialized like Claude Code skills: a
SKILL.md with YAML frontmatter (name, description) + instructions, applied to a real
input→target. Capped per task so coverage is broad (many skills) not deep (one drilled).
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from src.plugins.sdk import stable_hash

_SKILL_SYS = (
    "You have Skills — reusable procedures defined with YAML frontmatter "
    "(name, description) and instructions. When a request matches a Skill's "
    "description, use it and follow its instructions."
)
_SKILL_CAP_PER_TASK = 40


def _skill_slug(task_name: str) -> str:
    """'task001_quoref_question_generation' → 'quoref-question-generation'."""
    slug = re.sub(r"^task\d+_", "", task_name or "").replace("_", "-").strip("-")
    return slug or "skill"


def stream_skills(langs: list[str], limit_mb: int | None = None) -> Iterator[dict]:
    """Stream real skills (EN) as Claude-style SKILL.md + an applied input→target."""
    if "en" not in {lang.lower() for lang in langs}:
        return
    from datasets import load_dataset

    try:
        ds = load_dataset("Muennighoff/natural-instructions", split="train", streaming=True)
    except Exception as e:
        print(f"    [skills] {e}")
        return
    seen: set = set()
    per_task: dict = {}
    for ex in ds:
        definition = " ".join((ex.get("definition") or "").split())
        inp = " ".join((ex.get("inputs") or "").split())
        target = " ".join((ex.get("targets") or "").split())
        if not (definition and inp and target):
            continue
        task = ex.get("task_name") or ""
        if per_task.get(task, 0) >= _SKILL_CAP_PER_TASK:  # breadth over depth
            continue
        slug = _skill_slug(task)
        text = (
            f"System: {_SKILL_SYS}\n"
            f"Skill:\n---\nname: {slug}\n"
            f"description: Use this skill to {slug.replace('-', ' ')}.\n---\n"
            f"{definition}\n"
            f"User: {inp}\n"
            f"Assistant: {target}"
        )
        h = stable_hash(text)
        if h in seen:
            continue
        seen.add(h)
        per_task[task] = per_task.get(task, 0) + 1
        yield {"text": text, "lang": "en"}


def _build_skills(*, langs, limit_mb=None, **_):
    return stream_skills(langs, limit_mb)


SOURCES = {"skills": _build_skills}
