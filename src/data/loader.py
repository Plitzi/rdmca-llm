"""
Streaming JSONL / TXT DataLoader for RDMCA curriculum training.
Accepts: .jsonl  {"text": "..."}   or  .txt  (one document per line)
Handles: multi-file directories, shuffling within a shard, resume by token offset.
"""
from __future__ import annotations
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from src.modalities.text import BOS_ID, EOS_ID

# Conversational transcripts are line-prefixed by role ("System:/User:/Assistant:"
# …, see src/data/graded.py). A record whose FIRST line opens with a role marker
# is a dialogue/instruction transcript; prose (stories, wiki) never does. RESPONSE
# roles are the turns the MODEL produces (trained on); every other role is CONTEXT
# the model is GIVEN (loss-masked) so it learns to ANSWER, not to model the whole
# transcript and drift into writing the user's next turn.
_TURN_RE = re.compile(r"^(User|Assistant|System|Tools|Action|Observation):")
_RESPONSE_ROLES = {"Assistant", "Action"}


def _split_turns(text: str) -> Optional[List[Tuple[str, str]]]:
    """Split a transcript into [(role, block), …] when conversational (first line
    opens with a role marker), else None (prose → train on every token). A turn's
    block is its role line plus any continuation lines up to the next marker, so
    multi-line answers (e.g. a `<think>…</think>` scratchpad) stay whole."""
    lines = text.split("\n")
    if not lines or not _TURN_RE.match(lines[0]):
        return None
    turns: List[Tuple[str, str]] = []
    role: Optional[str] = None
    buf: List[str] = []
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

    def __init__(self,
                 data_dir: str,
                 tokenizer,
                 seq_len: int = 2048,
                 batch_size: int = 8,
                 shuffle: bool = True,
                 seed: int = 42,
                 shuffle_buffer: int = 50000,
                 source_weights: Optional[Dict[str, float]] = None,
                 val: bool = False,
                 with_mask: bool = False):
        self.data_dir   = Path(data_dir)
        self.tokenizer  = tokenizer
        self.seq_len    = seq_len
        self.batch_size = batch_size
        self.shuffle    = shuffle
        # val=True loads ONLY the held-out `*.val.jsonl` files; val=False loads the
        # training files and EXCLUDES them — so the two never overlap.
        self.val        = val
        # with_mask=True makes `batches()` yield (tokens, loss_mask) pairs so the
        # trainer can apply COMPLETION-ONLY loss on conversational data (train the
        # assistant's tokens, mask the user/system context). False (default) yields
        # bare token arrays — unchanged behaviour for val / consolidation / prose.
        self.with_mask  = with_mask
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

    def _encode_record(self, rec: dict) -> Tuple[List[int], List[int]]:
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
        lang  = rec.get("lang", "en")
        turns = _split_turns(text)
        if turns is None:                                   # prose → full next-token loss
            try:
                ids = self.tokenizer.encode(text, lang=lang, add_bos=True, add_eos=True)
            except Exception:
                return [], []
            return ids, [1] * len(ids)
        return self._encode_turns(turns, lang)

    def _encode_turns(self, turns: List[Tuple[str, str]], lang: str) -> Tuple[List[int], List[int]]:
        """Completion-only encoding: BOS (+ lang) prefix, then each turn tokenized in
        place; assistant/action turns and a trailing EOS get mask=1, the rest mask=0.
        Records with no response turn carry no training signal and are dropped."""
        if not any(role in _RESPONSE_ROLES for role, _ in turns):
            return [], []
        ids:  List[int] = [BOS_ID]
        mask: List[int] = [0]
        lang_map = getattr(self.tokenizer, "lang_tokens", None) or {}
        if lang in lang_map:
            ids.append(lang_map[lang]); mask.append(0)
        for i, (role, block) in enumerate(turns):
            seg = block if i == 0 else "\n" + block         # restore the newline join
            try:
                piece = self.tokenizer.encode_raw(seg)
            except Exception:
                return [], []
            train = 1 if role in _RESPONSE_ROLES else 0
            ids.extend(piece); mask.extend([train] * len(piece))
            if train:                                       # learn to stop after the answer
                ids.append(EOS_ID); mask.append(1)
        return ids, mask

    def _iter_pairs(self) -> Iterator[Tuple[int, int]]:
        """Yield (token_id, loss_mask) over the full corpus, record by record."""
        for rec in self._iter_records():
            t_ids, m_ids = self._encode_record(rec)
            if t_ids:
                yield from zip(t_ids, m_ids)

    def batches(self):
        """
        Yields a numpy int32 array of shape [batch_size, seq_len+1] (or, when
        `with_mask=True`, a (tokens, loss_mask) pair of two such arrays). Backend-
        neutral — the training step converts to a backend tensor via
        `B.ops.array(batch)`. Loops over the corpus indefinitely.
        """
        tokens_needed = self.batch_size * (self.seq_len + 1)
        shape = (self.batch_size, self.seq_len + 1)
        tbuf: List[int] = []
        mbuf: List[int] = []

        while True:
            epoch_count = 0
            for tok, m in self._iter_pairs():
                tbuf.append(tok); mbuf.append(m)
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
        replay = replay or []
        self._ds       = dataset
        self._iter     = dataset.batches()
        self.tokens_per_batch = dataset.batch_size * dataset.seq_len
        self._replay_iters = [d.batches() for d in replay]
        # Weight replay selection by corpus SIZE so the largest earlier stage —
        # conversation (stage 1), by far the biggest — DOMINATES rehearsal and is
        # the skill most protected from erosion. Uniform selection gave a 200M-token
        # conversation corpus the SAME refresh weight as a 4K-token arithmetic one,
        # so a late stage's frozen core forgot how to converse (it "went crazy").
        self._replay_weights = [self._corpus_bytes(d) for d in replay] or None
        self._replay_fraction = float(replay_fraction) if self._replay_iters else 0.0
        self._rng = random.Random(seed)

    @staticmethod
    def _corpus_bytes(ds: TextDataset) -> float:
        """Total on-disk bytes of a dataset's files — a cheap proxy for token count,
        used to weight rehearsal toward the larger (more important) earlier stages."""
        try:
            return float(max(sum(f.stat().st_size for f in ds._files), 1))
        except Exception:
            return 1.0

    def next_batch(self) -> np.ndarray:
        if self._replay_iters and self._rng.random() < self._replay_fraction:
            it = self._rng.choices(self._replay_iters, weights=self._replay_weights, k=1)[0]
            return next(it)
        return next(self._iter)

    def skip(self, n_batches: int) -> int:
        """Fast-forward the stream by `n_batches`, discarding them — used on --resume
        so training continues from where it stopped instead of re-reading the corpus
        from the start (issue C3). Because the dataset AND the replay-vs-primary draw
        are fully seeded, replaying the same number of batches reproduces the exact
        stream position (same primary/replay interleaving). Returns how many were
        actually skipped (< n_batches only if the stream ended, which is rare for the
        cycling corpus). Cost is one-time re-tokenization of the skipped span."""
        skipped = 0
        for _ in range(max(0, int(n_batches))):
            try:
                self.next_batch()
            except StopIteration:
                break
            skipped += 1
        return skipped

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
                    replay_fraction: float = 0.0, val: bool = False,
                    with_mask: bool = False) -> "DataLoader":
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
                               source_weights=source_weights, val=val,
                               with_mask=(with_mask and not val))

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
