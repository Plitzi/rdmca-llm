"""
Streaming JSONL / TXT DataLoader for RDMCA curriculum training.
Accepts: .jsonl  {"text": "..."}   or  .txt  (one document per line)
Handles: multi-file directories, shuffling within a shard, resume by token offset.
"""
from __future__ import annotations
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np


class TextDataset:
    """
    Streams tokenized batches from a directory of .jsonl / .txt files.
    Fills a ring-buffer and yields (batch_size, seq_len+1) token arrays.
    The +1 extra token is for next-token prediction (input = [:-1], target = [1:]).
    """

    def __init__(self,
                 data_dir: str,
                 tokenizer,
                 seq_len: int = 2048,
                 batch_size: int = 8,
                 shuffle: bool = True,
                 seed: int = 42,
                 shuffle_buffer: int = 50000,
                 source_weights: Optional[Dict[str, float]] = None,
                 val: bool = False):
        self.data_dir   = Path(data_dir)
        self.tokenizer  = tokenizer
        self.seq_len    = seq_len
        self.batch_size = batch_size
        self.shuffle    = shuffle
        # val=True loads ONLY the held-out `*.val.jsonl` files; val=False loads the
        # training files and EXCLUDES them — so the two never overlap.
        self.val        = val
        self.rng        = random.Random(seed)
        # Records held in memory to randomize order across the interleaved file
        # stream (see _iter_records). Big enough to break up file-sized blocks.
        self.shuffle_buffer = shuffle_buffer
        # Per-source sampling weights, keyed by file stem (e.g. {"dialogue": 3.0}).
        # Multiplies the size-proportional draw weight so a small but important
        # source (dialogue) can be oversampled relative to a large one (stories),
        # shifting the mixture the model trains on without needing more raw data.
        self.source_weights = source_weights or {}
        # Corpus-cycling telemetry: `passes` counts complete reads of the corpus,
        # `epoch_tokens` is the token count of one full pass (set after the first
        # epoch). The trainer uses these to cap re-cycling and avoid overfitting a
        # small corpus toward an oversized token target.
        self.passes       = 0
        self.epoch_tokens: Optional[int] = None
        self._files     = self._find_files()
        if not self._files:
            raise FileNotFoundError(
                f"No .jsonl or .txt files found in {data_dir}. "
                "Run: python scripts/prepare_data.py --stage N first."
            )

    def _find_files(self) -> List[Path]:
        if self.val:
            files = list(self.data_dir.glob("*.val.jsonl"))     # held-out split only
        else:
            files = [f for f in self.data_dir.glob("*.jsonl")
                     if not f.name.endswith(".val.jsonl")]       # training, exclude val
            files += list(self.data_dir.glob("*.txt"))
        if self.shuffle:
            self.rng.shuffle(files)
        return files

    def _records_in_file(self, path: Path) -> Iterator[dict]:
        """Yield parsed records from ONE file, in file order. JSONL rows are dicts;
        TXT lines become {"text": line}. A record may carry pre-tokenized multimodal
        ids under "tokens" (unified vocab) instead of "text"."""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if path.suffix == ".jsonl":
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
                else:
                    yield {"text": line}

    def _iter_records(self) -> Iterator[dict]:
        """Yield records from all files, INTERLEAVED across sources.

        Reading one file fully and then the next (the old behavior) means the
        model trains on a whole distribution — e.g. ~19M tokens of pure dialogue —
        and then switches to another (pure stories), so it catastrophically forgets
        the first within a single epoch (measured: held-out perplexity jumped from
        ~67 to ~108 exactly at the file boundary). Instead we draw records from all
        files at once, weighted by file size so they drain together (no single-
        source tail), then pass them through a shuffle buffer for local randomness.
        With shuffle=False we keep the simple sequential read (deterministic)."""
        files = list(self._files)
        if not self.shuffle:
            for path in files:
                yield from self._records_in_file(path)
            return

        self.rng.shuffle(files)
        n       = len(files)
        iters   = [self._records_in_file(p) for p in files]
        # Draw weight = file size × the source's configured weight. Size keeps the
        # files draining together (no single-source tail / catastrophic forgetting);
        # source_weights lets a small source be oversampled into the mixture.
        weights = [max(p.stat().st_size, 1) * float(self.source_weights.get(p.stem, 1.0))
                   for p in files]
        idx     = list(range(n))
        done    = [False] * n          # each source has been read fully ≥ once

        def _draw() -> Optional[dict]:
            # Run until EVERY file has been read through at least once. A source
            # that exhausts early (small, or oversampled) is RESTARTED so it keeps
            # appearing throughout the epoch instead of leaving a single-source tail.
            while not all(done):
                i = self.rng.choices(idx, weights=weights)[0]
                try:
                    return next(iters[i])
                except StopIteration:
                    done[i] = True
                    if all(done):
                        return None
                    iters[i] = self._records_in_file(files[i])   # cycle
            return None

        buf: List[dict] = []
        K = max(self.shuffle_buffer, 1)
        while True:
            rec = _draw()
            if rec is None:
                break
            buf.append(rec)
            if len(buf) >= K:
                j = self.rng.randrange(len(buf))
                buf[j], buf[-1] = buf[-1], buf[j]
                yield buf.pop()
        self.rng.shuffle(buf)
        yield from buf

    def _iter_tokens(self) -> Iterator[int]:
        """Yield token IDs from the full corpus (text or pre-tokenized multimodal)."""
        for rec in self._iter_records():
            tokens = rec.get("tokens")
            if tokens:
                yield from tokens
                continue
            text = rec.get("text", "")
            if not text.strip():
                continue
            try:
                ids = self.tokenizer.encode(
                    text, lang=rec.get("lang", "en"), add_bos=True, add_eos=True)
                yield from ids
            except Exception:
                continue

    def batches(self) -> Iterator[np.ndarray]:
        """
        Yields a numpy int32 array of shape [batch_size, seq_len+1]. Backend-
        neutral — the training step converts to a backend tensor via
        `B.ops.array(batch)`. Loops over the corpus indefinitely.
        """
        tokens_needed = self.batch_size * (self.seq_len + 1)
        buf: List[int] = []

        while True:
            epoch_count = 0
            for tok in self._iter_tokens():
                buf.append(tok)
                epoch_count += 1
                if len(buf) >= tokens_needed:
                    arr = np.array(buf[:tokens_needed], dtype=np.int32)
                    arr = arr.reshape(self.batch_size, self.seq_len + 1)
                    yield arr
                    buf = buf[tokens_needed:]
            # One full pass over the corpus finished; record its size once so the
            # trainer can detect (and cap) re-cycling of a small corpus.
            self.passes += 1
            if self.epoch_tokens is None:
                self.epoch_tokens = epoch_count

    def __iter__(self):
        return self.batches()


class DataLoader:
    """Wraps TextDataset with token counting; yields fixed-shape batches.

    Optional REHEARSAL: a list of `replay` datasets (earlier stages' corpora) and
    a `replay_fraction` ∈ [0,1]. That fraction of batches is drawn from a random
    replay dataset instead of the primary one, so training a later cognitive stage
    keeps refreshing earlier skills (esp. conversation) and the frozen core does
    not forget them before the freeze. Pass/token telemetry tracks the PRIMARY
    dataset only (replay is supplementary)."""

    def __init__(self, dataset: TextDataset, replay: Optional[List[TextDataset]] = None,
                 replay_fraction: float = 0.0, seed: int = 1234):
        self._ds       = dataset
        self._iter     = dataset.batches()
        self.tokens_per_batch = dataset.batch_size * dataset.seq_len
        self._replay_iters = [d.batches() for d in (replay or [])]
        self._replay_fraction = float(replay_fraction) if self._replay_iters else 0.0
        self._rng = random.Random(seed)

    def next_batch(self) -> np.ndarray:
        if self._replay_iters and self._rng.random() < self._replay_fraction:
            return next(self._rng.choice(self._replay_iters))
        return next(self._iter)

    @property
    def passes(self) -> int:
        """Number of complete passes over the corpus so far."""
        return self._ds.passes

    @property
    def epoch_tokens(self) -> Optional[int]:
        """Tokens in one full pass of the corpus (None until the first completes)."""
        return self._ds.epoch_tokens

    @classmethod
    def from_config(cls, stage: int, cfg: dict, tokenizer,
                    replay_dirs: Optional[List[str]] = None,
                    replay_fraction: float = 0.0, val: bool = False) -> "DataLoader":
        """Build a DataLoader directly from the YAML config + stage number.

        `replay_dirs` (earlier stages' corpora) + `replay_fraction` enable
        rehearsal: that fraction of batches comes from those dirs so a later
        cognitive stage keeps the earlier skills fresh. Missing replay dirs are
        skipped silently. With `val=True` it loads only the held-out `*.val.jsonl`
        split (no replay, no oversampling) for honest generalization measurement."""
        mcfg    = cfg["model"]
        tcfg    = cfg["training"]
        stage_cfg = cfg["curriculum"][f"stage{stage}"]   # key-based (levels may omit stages)
        lvl       = cfg.get("level")                      # per-level layout: data/level{N}/stage{S}
        default_dir = f"data/level{lvl}/stage{stage}" if lvl is not None else f"data/stage{stage}"
        data_dir  = stage_cfg.get("data_dir", default_dir)
        # Optional per-source oversampling, e.g. data.source_weights:{dialogue:3.0}
        # to push the conversational share of the training mixture up.
        source_weights = None if val else (stage_cfg.get("data", {}) or {}).get("source_weights")

        def _ds(path: str) -> TextDataset:
            return TextDataset(data_dir=path, tokenizer=tokenizer,
                               seq_len=mcfg["context_len"],
                               batch_size=tcfg["batch_size"], shuffle=True,
                               source_weights=source_weights, val=val)

        if val:
            return cls(_ds(data_dir))        # held-out split, no replay

        ds = _ds(data_dir)
        replay: List[TextDataset] = []
        for d in (replay_dirs or []):
            try:
                replay.append(_ds(d))
            except FileNotFoundError:
                continue                                  # earlier stage not prepared → skip
        return cls(ds, replay=replay, replay_fraction=replay_fraction)
