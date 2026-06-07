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


DEFAULT_CONFIG = "configs/rdmca_t2.yaml"
TOKENIZER_INFO = "dist/tokenizer/tokenizer_info.json"


def resolve_config_path(config: Optional[str] = None,
                        profile: Optional[str] = None) -> str:
    """A profile name wins over an explicit config path; else the default."""
    if profile:
        return f"configs/profiles/{profile}.yaml"
    return config or DEFAULT_CONFIG


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_languages(cfg: dict) -> List[str]:
    """Canonical language list for a config. Defaults to ['en']."""
    langs = (cfg.get("model", {}) or {}).get("languages")
    return list(langs) if langs else ["en"]


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
