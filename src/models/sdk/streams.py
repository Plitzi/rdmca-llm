"""Stream-combining helpers: blend a finite real corpus with an unlimited synthetic
one, round-robin interleave several streams, and cycle a bounded record set."""

from __future__ import annotations

import random
from collections.abc import Iterator


def blend(real_it: Iterator[dict], synth_it: Iterator[dict], n_examples: int) -> Iterator[dict]:
    """Interleave a finite REAL corpus with an unlimited SYNTHETIC one (≈1:1 while
    the real seed lasts, then pure synthetic) up to `n_examples` — so real records
    are spread THROUGHOUT the file (not front-loaded as one block, which the loader
    would only locally reshuffle). Stops early if synthetic runs out. Robust when
    the real stream is empty (offline / dataset load failed): becomes pure synthetic."""
    produced = 0
    real_done = False
    while produced < n_examples:
        if not real_done:
            rec = next(real_it, None)
            if rec is None:
                real_done = True
            elif rec.get("text", "").strip():
                yield rec
                produced += 1
                if produced >= n_examples:
                    return
        rec = next(synth_it, None)
        if rec is None:
            return
        if rec.get("text", "").strip():
            yield rec
            produced += 1


def interleave(*streams: Iterator[dict]) -> Iterator[dict]:
    """Round-robin across live generators until all exhaust, so no single corpus (or
    mood) forms a front-loaded block in the output — the same anti-forgetting mixing
    the training loader does across files, applied here across dialogue sources."""
    live = [s for s in streams if s is not None]
    while live:
        still_live = []
        for stream in live:
            try:
                rec = next(stream)
            except StopIteration:
                continue
            still_live.append(stream)
            yield rec
        live = still_live


def cycle_records(records: list[dict], n: int, seed: int = 1) -> Iterator[dict]:
    """Yield up to `n` records by CYCLING a small bounded set, reshuffling each pass.
    For CLEAN canonical data (definitions, basic_chat) controlled repetition is
    desirable anchoring — the model should see 'A car is …' / 'hi'→'hi!' many times —
    while a per-source token budget (prepare_data) keeps the volume controlled."""
    if not records:
        return
    rng = random.Random(seed)
    produced = 0
    while produced < n:
        rng.shuffle(records)
        for rec in records:
            yield rec
            produced += 1
            if produced >= n:
                return
