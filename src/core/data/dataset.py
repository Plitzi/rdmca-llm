"""Streaming JSONL / TXT TextDataset for RDMCA curriculum training.

Accepts: .jsonl  {"text": "..."}   or  .txt  (one document per line). Interleaves
across files (size-weighted) to avoid catastrophic forgetting at file boundaries,
shuffles within a buffer, applies completion-only loss masking to conversational
transcripts, and supports a --resume fast-skip via cached per-record token lengths.

The rehearsal + resume-index wrapper (DataLoader) lives in loader.py.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from src.core.modalities.text import BOS_ID, EOS_ID

# Conversational transcripts are line-prefixed by role ("System:/User:/Assistant:"
# …, see src/data/graded.py). A record whose FIRST line opens with a role marker
# is a dialogue/instruction transcript; prose (stories, wiki) never does. RESPONSE
# roles are the turns the MODEL produces (trained on); every other role is CONTEXT
# the model is GIVEN (loss-masked) so it learns to ANSWER, not to model the whole
# transcript and drift into writing the user's next turn.
_TURN_RE = re.compile(r"^(User|Assistant|System|Tools|Action|Observation):")
_RESPONSE_ROLES = {"Assistant", "Action"}

# Sentinel emitted (in place of real token ids) for cache-known records during a
# --resume fast-skip: it is never a real id (ids are ≥ 0), so the loader can detect
# and flush any partial carry-over it lands in, keeping the resumed stream exact.
_SKIP_FILL = -1


def _split_turns(text: str) -> list[tuple[str, str]] | None:
    """Split a transcript into [(role, block), …] when conversational (first line
    opens with a role marker), else None (prose → train on every token). A turn's
    block is its role line plus any continuation lines up to the next marker, so
    multi-line answers (e.g. a `<think>…</think>` scratchpad) stay whole."""
    lines = text.split("\n")
    if not lines or not _TURN_RE.match(lines[0]):
        return None
    turns: list[tuple[str, str]] = []
    role: str | None = None
    buf: list[str] = []
    for ln in lines:
        m = _TURN_RE.match(ln)
        if m:
            if role is not None:
                turns.append((role, "\n".join(buf)))
            role, buf = m.group(1), [ln]
        else:
            buf.append(ln)
    if role is not None:
        turns.append((role, "\n".join(buf)))
    return turns


class TextDataset:
    """
    Streams tokenized batches from a directory of .jsonl / .txt files.
    Fills a ring-buffer and yields (batch_size, seq_len+1) token arrays.
    The +1 extra token is for next-token prediction (input = [:-1], target = [1:]).
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer,
        seq_len: int = 2048,
        batch_size: int = 8,
        shuffle: bool = True,
        seed: int = 42,
        shuffle_buffer: int = 50000,
        source_weights: dict[str, float] | None = None,
        val: bool = False,
        with_mask: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        # val=True loads ONLY the held-out `*.val.jsonl` files; val=False loads the
        # training files and EXCLUDES them — so the two never overlap.
        self.val = val
        # with_mask=True makes `batches()` yield (tokens, loss_mask) pairs so the
        # trainer can apply COMPLETION-ONLY loss on conversational data (train the
        # assistant's tokens, mask the user/system context). False (default) yields
        # bare token arrays — unchanged behaviour for val / consolidation / prose.
        self.with_mask = with_mask
        self.rng = random.Random(seed)
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
        self.passes = 0
        self.epoch_tokens: int | None = None
        # --resume fast-forward support. `_len_cache` maps a file name → the token
        # LENGTH of each record (in read order), filled as records are encoded during
        # training and persisted with the checkpoint. When `_skip_mode` is on (set by
        # DataLoader.skip), a record whose length is cached is returned as dummy ids of
        # that length: the batch boundaries — hence the exact downstream stream
        # position — are reproduced WITHOUT paying the (expensive) BPE tokenization.
        self._skip_mode = False
        self._len_cache: dict[str, list[int]] = {}
        self._files = self._find_files()
        if not self._files:
            raise FileNotFoundError(
                f"No .jsonl or .txt files found in {data_dir}. "
                "Run: python scripts/prepare_data.py --stage N first."
            )

    def _find_files(self) -> list[Path]:
        if self.val:
            files = list(self.data_dir.glob("*.val.jsonl"))  # held-out split only
        else:
            files = [
                f for f in self.data_dir.glob("*.jsonl") if not f.name.endswith(".val.jsonl")
            ]  # training, exclude val
            files += list(self.data_dir.glob("*.txt"))
        if self.shuffle:
            self.rng.shuffle(files)
        return files

    def _records_in_file(self, path: Path) -> Iterator[dict]:
        """Yield parsed records from ONE file, in file order. JSONL rows are dicts;
        TXT lines become {"text": line}. A record may carry pre-tokenized multimodal
        ids under "tokens" (unified vocab) instead of "text"."""
        stem = path.name
        idx = 0  # counts YIELDED records (stable key)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if path.suffix == ".jsonl":
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(rec, dict):
                        continue
                else:
                    rec = {"text": line}
                # (file, record#) — the key the length-cache uses to fast-skip on resume.
                rec["__cache"] = (stem, idx)
                idx += 1
                yield rec

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
        n = len(files)
        iters = [self._records_in_file(p) for p in files]
        # Draw weight = file size × the source's configured weight. Size keeps the
        # files draining together (no single-source tail / catastrophic forgetting);
        # source_weights lets a small source be oversampled into the mixture.
        weights = [
            max(p.stat().st_size, 1) * float(self.source_weights.get(p.stem, 1.0)) for p in files
        ]
        idx = list(range(n))
        done = [False] * n  # each source has been read fully ≥ once

        def _draw() -> dict | None:
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
                    iters[i] = self._records_in_file(files[i])  # cycle
            return None

        buf: list[dict] = []
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

    def _encode_record(self, rec: dict) -> tuple[list[int], list[int]]:
        """Tokenize one record, with a fast-skip shortcut. During a --resume fast-
        forward (`_skip_mode`) the token VALUES are discarded, so for a cache-known
        record we return dummy ids of the cached LENGTH — that reproduces the exact
        batch boundaries (and thus the downstream stream position) without the BPE
        cost. Misses and normal training fall through to a real encode, whose length
        is recorded so a later resume can skip it cheaply."""
        ref = rec.get("__cache")
        if ref is not None and getattr(self, "_skip_mode", False):
            arr = self._len_cache.get(ref[0])
            if arr is not None and ref[1] < len(arr) and arr[ref[1]] >= 0:
                L = int(arr[ref[1]])  # ≥ 0 = a known length (0 is a valid
                return [_SKIP_FILL] * L, [0] * L  # empty record); < 0 = not cached → miss
        t_ids, m_ids = self._encode_record_live(rec)
        if ref is not None:
            self._record_len(ref[0], ref[1], len(t_ids))
        return t_ids, m_ids

    def _record_len(self, stem: str, idx: int, length: int) -> None:
        """Remember that record `idx` of file `stem` produced `length` tokens. The
        cache is keyed by the record's position in FILE order, but records are drawn
        SHUFFLED, so the slot can be far ahead of what's filled — gaps are padded with
        -1 (= "not cached yet"), which `_encode_record` treats as a miss (real encode).
        A genuinely empty record stores 0, a valid known length."""
        arr = self._len_cache.get(stem)
        if arr is None:
            arr = []
            self._len_cache[stem] = arr
        if idx == len(arr):
            arr.append(length)
        elif idx < len(arr):
            arr[idx] = length
        else:  # shuffled draw landed past the fill point
            arr.extend([-1] * (idx - len(arr)))
            arr.append(length)

    def len_cache_arrays(self) -> dict[str, np.ndarray]:
        """Per-file token-length arrays gathered so far (for persistence)."""
        return {stem: np.asarray(v, dtype=np.int32) for stem, v in self._len_cache.items()}

    def load_len_cache(self, arrays: dict[str, np.ndarray]) -> None:
        for stem, a in arrays.items():
            self._len_cache[stem] = [int(x) for x in a]

    def _encode_record_live(self, rec: dict) -> tuple[list[int], list[int]]:
        """Return parallel (token_ids, loss_mask) for one record — mask=1 trains the
        token, 0 ignores it in the loss. Pre-tokenized multimodal and prose train on
        EVERY token. Conversational transcripts train ONLY the assistant/action turns
        (plus a turn-final EOS, so the model learns to STOP after answering) and mask
        the user/system/tool context the model is merely GIVEN."""
        tokens = rec.get("tokens")
        if tokens:
            return list(tokens), [1] * len(tokens)
        text = rec.get("text", "")
        if not text.strip():
            return [], []
        lang = rec.get("lang", "en")
        turns = _split_turns(text)
        if turns is None:  # prose → full next-token loss
            try:
                ids = self.tokenizer.encode(text, lang=lang, add_bos=True, add_eos=True)
            except Exception:
                return [], []
            return ids, [1] * len(ids)
        return self._encode_turns(turns, lang)

    def _encode_turns(self, turns: list[tuple[str, str]], lang: str) -> tuple[list[int], list[int]]:
        """Completion-only encoding: BOS (+ lang) prefix, then each turn tokenized in
        place; assistant/action turns and a trailing EOS get mask=1, the rest mask=0.
        Records with no response turn carry no training signal and are dropped."""
        if not any(role in _RESPONSE_ROLES for role, _ in turns):
            return [], []
        ids: list[int] = [BOS_ID]
        mask: list[int] = [0]
        lang_map = getattr(self.tokenizer, "lang_tokens", None) or {}
        if lang in lang_map:
            ids.append(lang_map[lang])
            mask.append(0)
        for i, (role, block) in enumerate(turns):
            seg = block if i == 0 else "\n" + block  # restore the newline join
            try:
                piece = self.tokenizer.encode_raw(seg)
            except Exception:
                return [], []
            train = 1 if role in _RESPONSE_ROLES else 0
            ids.extend(piece)
            mask.extend([train] * len(piece))
            if train:  # learn to stop after the answer
                ids.append(EOS_ID)
                mask.append(1)
        return ids, mask

    def _iter_pairs(self) -> Iterator[tuple[int, int]]:
        """Yield (token_id, loss_mask) over the full corpus, record by record."""
        for rec in self._iter_records():
            t_ids, m_ids = self._encode_record(rec)
            if t_ids:
                yield from zip(t_ids, m_ids, strict=False)

    def batches(self):
        """
        Yields a numpy int32 array of shape [batch_size, seq_len+1] (or, when
        `with_mask=True`, a (tokens, loss_mask) pair of two such arrays). Backend-
        neutral — the training step converts to a backend tensor via
        `B.ops.array(batch)`. Loops over the corpus indefinitely.
        """
        tokens_needed = self.batch_size * (self.seq_len + 1)
        shape = (self.batch_size, self.seq_len + 1)
        tbuf: list[int] = []
        mbuf: list[int] = []

        while True:
            epoch_count = 0
            for tok, m in self._iter_pairs():
                tbuf.append(tok)
                mbuf.append(m)
                epoch_count += 1
                if len(tbuf) >= tokens_needed:
                    arr = np.array(tbuf[:tokens_needed], dtype=np.int32).reshape(shape)
                    if self.with_mask:
                        marr = np.array(mbuf[:tokens_needed], dtype=np.int32).reshape(shape)
                        yield arr, marr
                    else:
                        yield arr
                    tbuf = tbuf[tokens_needed:]
                    mbuf = mbuf[tokens_needed:]
            # One full pass over the corpus finished; record its size once so the
            # trainer can detect (and cap) re-cycling of a small corpus.
            self.passes += 1
            if self.epoch_tokens is None:
                self.epoch_tokens = epoch_count
            if epoch_count == 0:
                # The corpus yields NO trainable tokens (empty *.val.jsonl, or a
                # split whose records all mask out). Re-entering the while-loop
                # would spin forever and hang next_batch(); end the stream instead
                # so callers (e.g. _make_val_batches) get StopIteration and fall back.
                return

    def __iter__(self):
        return self.batches()
