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
from typing import Iterator, List, Optional

import numpy as np
import mlx.core as mx


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
                 seed: int = 42):
        self.data_dir   = Path(data_dir)
        self.tokenizer  = tokenizer
        self.seq_len    = seq_len
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.rng        = random.Random(seed)
        self._files     = self._find_files()
        if not self._files:
            raise FileNotFoundError(
                f"No .jsonl or .txt files found in {data_dir}. "
                "Run: python scripts/prepare_data.py --stage N first."
            )

    def _find_files(self) -> List[Path]:
        files = (list(self.data_dir.glob("*.jsonl")) +
                 list(self.data_dir.glob("*.txt")))
        if self.shuffle:
            self.rng.shuffle(files)
        return files

    def _iter_records(self) -> Iterator[dict]:
        """Yield records from all files. JSONL rows are dicts; TXT lines become
        {"text": line}. A record may carry pre-tokenized multimodal ids under
        "tokens" (unified vocab) instead of "text"."""
        files = list(self._files)
        if self.shuffle:
            self.rng.shuffle(files)
        for path in files:
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

    def batches(self) -> Iterator[mx.array]:
        """
        Yields mx.array of shape [batch_size, seq_len+1].
        Loops over the corpus indefinitely.
        """
        tokens_needed = self.batch_size * (self.seq_len + 1)
        buf: List[int] = []

        while True:
            for tok in self._iter_tokens():
                buf.append(tok)
                if len(buf) >= tokens_needed:
                    arr = np.array(buf[:tokens_needed], dtype=np.int32)
                    arr = arr.reshape(self.batch_size, self.seq_len + 1)
                    yield mx.array(arr)
                    buf = buf[tokens_needed:]

    def __iter__(self):
        return self.batches()


class DataLoader:
    """Wraps TextDataset with token counting; yields fixed-shape batches."""

    def __init__(self, dataset: TextDataset):
        self._ds       = dataset
        self._iter     = dataset.batches()
        self.tokens_per_batch = dataset.batch_size * dataset.seq_len

    def next_batch(self) -> mx.array:
        return next(self._iter)

    @classmethod
    def from_config(cls, stage: int, cfg: dict, tokenizer) -> "DataLoader":
        """Build a DataLoader directly from the YAML config + stage number."""
        stages  = list(cfg["curriculum"].values())
        mcfg    = cfg["model"]
        tcfg    = cfg["training"]
        stage_cfg = stages[stage - 1]
        data_dir  = stage_cfg.get("data_dir", f"data/stage{stage}_language")

        ds = TextDataset(
            data_dir=data_dir,
            tokenizer=tokenizer,
            seq_len=mcfg["context_len"],
            batch_size=tcfg["batch_size"],
            shuffle=True,
        )
        return cls(ds)
