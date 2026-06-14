"""
Rehearsal + resume-aware DataLoader for RDMCA curriculum training.

Wraps TextDataset (src/data/dataset.py) with token counting, optional REHEARSAL of
earlier stages' corpora, and a --resume fast-skip backed by a persisted per-record
token-length index. TextDataset + _split_turns are re-exported here for the many
call sites that import them from this module.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

from src.core.data.dataset import (
    _SKIP_FILL,
    TextDataset,
    _split_turns,  # noqa: F401  (re-exported for callers/tests)
)


class DataLoader:
    """Wraps TextDataset with token counting; yields fixed-shape batches.

    Optional REHEARSAL: a list of `replay` datasets (earlier stages' corpora) and
    a `replay_fraction` ∈ [0,1]. That fraction of batches is drawn from a random
    replay dataset instead of the primary one, so training a later cognitive stage
    keeps refreshing earlier skills (esp. conversation) and the frozen core does
    not forget them before the freeze. Pass/token telemetry tracks the PRIMARY
    dataset only (replay is supplementary)."""

    def __init__(
        self,
        dataset: TextDataset,
        replay: list[TextDataset] | None = None,
        replay_fraction: float = 0.0,
        seed: int = 1234,
    ):
        replay = replay or []
        self._ds = dataset
        self._replay_datasets = replay
        self._iter = dataset.batches()
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
        self._has_skip_index = False  # a length index was loaded → skip() can fast-path
        self._pushback: np.ndarray | None = None  # first clean batch after a fast-skip
        self.last_was_replay = False  # was the LAST next_batch() a rehearsal (replay) draw?
        # lets the trainer tag metrics so the bimodal
        # narrow-skill vs conversation-rehearsal loss can be
        # charted as two clean curves (not one spiky mess).

    @staticmethod
    def _corpus_bytes(ds: TextDataset) -> float:
        """Total on-disk bytes of a dataset's files — a cheap proxy for token count,
        used to weight rehearsal toward the larger (more important) earlier stages."""
        try:
            return float(max(sum(f.stat().st_size for f in ds._files), 1))
        except Exception:
            return 1.0

    def next_batch(self) -> np.ndarray:
        if self._pushback is not None:  # hand back the clean batch a fast-skip held
            b, self._pushback = self._pushback, None
            return b
        if self._replay_iters and self._rng.random() < self._replay_fraction:
            it = self._rng.choices(self._replay_iters, weights=self._replay_weights, k=1)[0]
            self.last_was_replay = True
            return next(it)
        self.last_was_replay = False
        return next(self._iter)

    def skip(self, n_batches: int) -> int:
        """Fast-forward the stream by `n_batches`, discarding them — used on --resume
        so training continues from where it stopped instead of re-reading the corpus
        from the start (issue C3). Because the dataset AND the replay-vs-primary draw
        are fully seeded, replaying the same number of batches reproduces the exact
        stream position (same primary/replay interleaving). Returns how many were
        actually skipped (< n_batches only if the stream ended, which is rare for the
        cycling corpus).

        When a token-length index has been loaded (`load_skip_index`, written at every
        checkpoint), the skipped span is reproduced from cached record lengths instead
        of re-tokenizing it — turning a multi-minute fast-forward into seconds. Since a
        resume only ever skips data the interrupted run already consumed (= already
        cached), hits are ~100%; any miss falls back to a real encode (still correct).

        The fast path emits placeholder ids whose only meaning is their COUNT, so the
        final batch holds a partial carry-over of placeholders. We then read forward in
        live mode, discarding placeholder-tainted batches and HANDING BACK the first
        clean one (via `_pushback`), so the resumed stream is bit-exact from the very
        next call. The cost is over-skipping < 1 batch of real data — negligible."""
        self._set_skip_mode(True)
        skipped = 0
        try:
            for _ in range(max(0, int(n_batches))):
                try:
                    self.next_batch()
                except StopIteration:
                    break
                skipped += 1
        finally:
            self._set_skip_mode(False)
        # Flush the placeholder carry-over so no resumed batch contains sentinels.
        if self._has_skip_index and skipped:
            while True:
                try:
                    b = self.next_batch()
                except StopIteration:
                    break
                if not bool((b == _SKIP_FILL).any()):
                    self._pushback = b  # clean → return it on the next call
                    break
        return skipped

    def _set_skip_mode(self, on: bool) -> None:
        self._ds._skip_mode = on
        for d in self._replay_datasets:
            d._skip_mode = on

    # ── resume skip index (per-record token lengths) ──────────────────────────
    def _tokenizer_sig(self) -> str:
        """A signature that changes when the tokenizer changes, so a stale length
        cache (built under a different/ retrained tokenizer) is ignored on load."""
        tok = self._ds.tokenizer
        try:
            st = Path(tok.model_path).stat()
            return f"{st.st_mtime_ns}:{st.st_size}:{getattr(tok, 'text_vocab_size', '')}"
        except Exception:
            return f"-:-:{getattr(tok, 'text_vocab_size', '')}"

    def save_skip_index(self, path) -> None:
        """Persist per-record token lengths gathered so far (primary + replay) next to
        the checkpoint, so a later --resume can fast-forward without re-tokenizing.
        Keyed `p::<file>` for the primary corpus, `r{i}::<file>` for replay i; `__sig__`
        ties the cache to the tokenizer that built it."""
        out = {"__sig__": np.array(self._tokenizer_sig())}
        for stem, a in self._ds.len_cache_arrays().items():
            out[f"p::{stem}"] = a
        for i, d in enumerate(self._replay_datasets):
            for stem, a in d.len_cache_arrays().items():
                out[f"r{i}::{stem}"] = a
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(p.name + ".tmp")
            with open(tmp, "wb") as f:  # explicit handle: np.savez won't append .npz
                np.savez(f, **out)
            os.replace(tmp, p)
        except OSError:
            pass

    def load_skip_index(self, path) -> bool:
        """Load a length index written by `save_skip_index`. Returns True when a
        usable, tokenizer-matched cache was loaded (so `skip` runs fast); False on a
        missing/stale/corrupt index (so `skip` falls back to live tokenization)."""
        p = Path(path)
        if not p.exists():
            return False
        try:
            data = np.load(p, allow_pickle=False)
        except Exception:
            return False
        files = set(data.files)
        if "__sig__" not in files or str(data["__sig__"]) != self._tokenizer_sig():
            return False  # tokenizer changed → cache stale
        pri: dict[str, np.ndarray] = {}
        rep: dict[int, dict[str, np.ndarray]] = {}
        for k in files:
            if k == "__sig__" or "::" not in k:
                continue
            tag, stem = k.split("::", 1)
            if tag == "p":
                pri[stem] = data[k]
            elif tag.startswith("r") and tag[1:].isdigit():
                rep.setdefault(int(tag[1:]), {})[stem] = data[k]
        self._ds.load_len_cache(pri)
        for i, d in enumerate(self._replay_datasets):
            if i in rep:
                d.load_len_cache(rep[i])
        self._has_skip_index = True
        return True

    @property
    def passes(self) -> int:
        """Number of complete passes over the corpus so far."""
        return self._ds.passes

    @property
    def epoch_tokens(self) -> int | None:
        """Tokens in one full pass of the corpus (None until the first completes)."""
        return self._ds.epoch_tokens

    @classmethod
    def from_config(
        cls,
        stage: int,
        cfg: dict,
        tokenizer,
        replay_dirs: list[str] | None = None,
        replay_fraction: float = 0.0,
        val: bool = False,
        with_mask: bool = False,
    ) -> DataLoader:
        """Build a DataLoader directly from the YAML config + stage number.

        `replay_dirs` (earlier stages' corpora) + `replay_fraction` enable
        rehearsal: that fraction of batches comes from those dirs so a later
        cognitive stage keeps the earlier skills fresh. Missing replay dirs are
        skipped silently. With `val=True` it loads only the held-out `*.val.jsonl`
        split (no replay, no oversampling) for honest generalization measurement."""
        mcfg = cfg["model"]
        tcfg = cfg["training"]
        from src.stages import stage_data_dir

        stage_cfg = cfg["curriculum"][f"stage{stage}"]  # key-based (levels may omit stages)
        # Each stage owns its data folder inside its package; a config data_dir wins.
        data_dir = stage_data_dir(stage, cfg)
        # Optional per-source oversampling, e.g. data.source_weights:{dialogue:3.0}
        # to push the conversational share of the training mixture up.
        source_weights = None if val else (stage_cfg.get("data", {}) or {}).get("source_weights")

        def _ds(path: str) -> TextDataset:
            # val batches may carry the completion mask too, so the gate can measure
            # response-only perplexity (matching training); see validation_perplexity.
            return TextDataset(
                data_dir=path,
                tokenizer=tokenizer,
                seq_len=mcfg["context_len"],
                batch_size=tcfg["batch_size"],
                shuffle=True,
                source_weights=source_weights,
                val=val,
                with_mask=with_mask,
            )

        if val:
            return cls(_ds(data_dir))  # held-out split, no replay

        ds = _ds(data_dir)
        replay: list[TextDataset] = []
        for d in replay_dirs or []:
            try:
                replay.append(_ds(d))
            except FileNotFoundError:
                continue  # earlier stage not prepared → skip
        return cls(ds, replay=replay, replay_fraction=replay_fraction)
