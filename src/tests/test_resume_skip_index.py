"""
Tests for the --resume fast-skip token-length index (src/data/loader.py):

  - skip() with a loaded index reproduces the EXACT downstream stream position
    (same records, same batches) as a plain run — without re-tokenizing;
  - the index round-trips to disk and is keyed to the tokenizer (a changed
    tokenizer signature invalidates it, so skip falls back to a correct live skip);
  - a missing index makes load_skip_index() report False (slow-but-correct path).

The fast skip is exact from the SECOND post-resume batch on: ≤1 batch at the resume
point carries placeholder tokens (the partial carry-over), by design.
"""

import json
import random
from pathlib import Path
from typing import ClassVar

import numpy as np

from src.data.loader import DataLoader, TextDataset


class _VarTok:
    """Deterministic char-level tokenizer whose record length varies with the text,
    so batch boundaries fall at non-trivial places (exercises the length index)."""

    lang_tokens: ClassVar[dict] = {}
    ready = True

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.text_vocab_size = 256

    def encode(self, text, lang="en", add_bos=True, add_eos=True):
        ids = [ord(c) % 200 + 3 for c in text]
        if add_bos:
            ids = [1, *ids]
        if add_eos:
            ids = [*ids, 2]
        return ids

    def encode_raw(self, text):
        return [ord(c) % 200 + 3 for c in text]

    def decode(self, ids):
        return ""


def _corpus(tmp_path: Path) -> Path:
    r = random.Random(0)
    alpha = "abcdefghijklmnopqrstuvwxyz"
    # Distinctive, variable-length records so batches are distinguishable (a uniform
    # corpus would make any two batches compare equal and defeat alignment checks).
    lines = [
        json.dumps({"text": "".join(r.choice(alpha) for _ in range(r.randint(5, 60)))})
        for _ in range(3000)
    ]
    (tmp_path / "corpus.jsonl").write_text("\n".join(lines))
    return tmp_path


def _loader(data_dir: Path, tok_path: Path, seed=1234) -> DataLoader:
    ds = TextDataset(
        str(data_dir), _VarTok(tok_path), seq_len=8, batch_size=2, shuffle=True, seed=42
    )
    return DataLoader(ds, seed=seed)


def _next_n(loader: DataLoader, n: int):
    return [loader.next_batch().copy() for _ in range(n)]


def test_fast_skip_matches_plain_stream(tmp_path):
    """After loading the index and fast-skipping K batches, the loader must produce
    batches that are BIT-EXACT to a plain straight-through read — at the same stream
    position (modulo < 1 batch of over-skip, which the flush rounds away). No resumed
    batch may contain the placeholder sentinel."""
    data = _corpus(tmp_path)
    tok_path = tmp_path / "tok.model"
    tok_path.write_text("dummy")
    K, N = 30, 12

    # Ground truth: a fresh loader read straight through (no skipping). Capture a few
    # extra batches so we can locate the (slightly over-skipped) resume offset.
    ref = _next_n(_loader(data, tok_path), K + N + 8)

    # Build + persist the index (live skip populates the length cache).
    builder = _loader(data, tok_path)
    builder.skip(K)
    idx_path = tmp_path / "skip_index.npz"
    builder.save_skip_index(idx_path)
    assert idx_path.exists()

    fast = _loader(data, tok_path)
    assert fast.load_skip_index(idx_path) is True
    assert fast.skip(K) == K
    fast_tail = _next_n(fast, N)

    # No sentinel leaks into the resumed stream.
    assert all(not bool((b == -1).any()) for b in fast_tail)

    # The resumed batches are a contiguous, bit-exact block of the plain stream,
    # starting at K (or a few batches later — the flush rounds the < 1 batch of
    # over-skipped real tokens up to a batch boundary).
    offsets = [
        o
        for o in range(K, K + 8)
        if all(np.array_equal(fast_tail[i], ref[o + i]) for i in range(N))
    ]
    assert offsets, "fast-skip stream did not align bit-exactly with the plain stream"


def test_skip_index_invalidated_by_tokenizer_change(tmp_path):
    """A changed tokenizer signature must make load_skip_index() refuse the cache."""
    data = _corpus(tmp_path)
    tok_path = tmp_path / "tok.model"
    tok_path.write_text("v1")

    builder = _loader(data, tok_path)
    builder.skip(20)
    idx_path = tmp_path / "skip_index.npz"
    builder.save_skip_index(idx_path)

    tok_path.write_text("v2-changed-bigger")  # retrained tokenizer → new sig
    fresh = _loader(data, tok_path)
    assert fresh.load_skip_index(idx_path) is False  # stale → ignored (slow but safe)


def test_load_skip_index_missing_is_false(tmp_path):
    data = _corpus(tmp_path)
    tok_path = tmp_path / "tok.model"
    tok_path.write_text("x")
    assert _loader(data, tok_path).load_skip_index(tmp_path / "nope.npz") is False
