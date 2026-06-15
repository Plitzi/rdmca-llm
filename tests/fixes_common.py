"""Shared helpers for the test_fix_* regression suites (split out of the old
test_fixes.py). Binds the MLX backend (always present here) and exposes the tiny
model + fake tokenizer + corpus-writer the suites reuse. Not collected as tests."""

import json
from typing import ClassVar

import src.backend as backend

backend.select("mlx")  # bind model/loader modules to MLX
B = backend.current()

from src.model.config import ModelConfig
from src.model.transformer import RDMCAFoundational


def tiny_model() -> RDMCAFoundational:
    cfg = ModelConfig(
        d_model=64, n_heads=2, ffn_dim=128, context_len=32, vocab_size=256, mrl_dims=[64]
    )
    return RDMCAFoundational(cfg)


class FakeTok:
    """Minimal tokenizer for the loader: deterministic ids, no special tokens."""

    lang_tokens: ClassVar[dict] = {}

    def encode(self, text, lang="en", add_bos=True, add_eos=True):
        ids = [(ord(c) % 250) + 3 for c in text][:40]
        return ids or [3]

    def encode_raw(self, text):
        return [(ord(c) % 250) + 5 for c in text]


def write_corpus(d):
    """Big "story" file and small "dialogue" file, each tagged so a test can
    identify the source."""
    with open(d / "story.jsonl", "w") as f:
        for _ in range(2000):
            f.write(json.dumps({"text": "STORY " + "x" * 60}) + "\n")
    with open(d / "dialogue.jsonl", "w") as f:
        for _ in range(400):
            f.write(json.dumps({"text": "DIALOG " + "y" * 60}) + "\n")
