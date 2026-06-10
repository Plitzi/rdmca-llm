"""
Experience queue — the bridge between live interaction and consolidation.

Active interaction (uses/chat/run_chat.py) appends each turn here; the consolidation daemon
drains the queue, scores/filters/consolidates it, then clears it. This is the
"experience and memory as the true training signal" loop of RDMCA §6.5.2 — the
model evolves only from what it actually experienced.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import List

EXPERIENCE_LOG = "data/runtime/experiences.jsonl"


def log_experience(text: str, lang: str = "en", modality: str = "text",
                   path: str = EXPERIENCE_LOG) -> None:
    if not text or not text.strip():
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps({"text": text, "lang": lang, "modality": modality,
                            "timestamp": time.time()}, ensure_ascii=False) + "\n")


def load_experiences(path: str = EXPERIENCE_LOG) -> List[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def clear_experiences(path: str = EXPERIENCE_LOG) -> None:
    p = Path(path)
    if p.exists():
        p.unlink()
