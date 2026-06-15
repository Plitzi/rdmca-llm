"""JSONL corpus writer + completeness validator for prepared training data.

`write_jsonl` streams records to `{source}.jsonl` (with a held-out `.val.jsonl`
split) until the token budget is reached, normalizing every record through the
ingestion gate. `validate_jsonl` decides whether an existing file is complete enough
to skip re-downloading on a re-run, preferring the exact `.meta.json` sidecar over a
size estimate.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from src.data.textnorm import clean_record_text  # ingestion gate

# Empirically ~3.91 chars/token for prose (tinystories/dialogue) based on actual
# tokenizer measurements (3.5 under-counted tokens by ~12%). Budgets remain
# approximate for structured stages — the runtime corpus cap uses real token counts.
CHARS_PER_TOKEN = 3.9


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN)


def validate_jsonl(path: Path, token_budget_m: int) -> tuple[bool, str]:
    """
    Check whether an existing JSONL file is complete enough to skip re-downloading.
    Returns (ok, reason_string).

    Rules:
      - Size 0 or missing       → invalid (re-download)
      - Can't parse first line  → invalid (corrupted)
      - tokens < 10% of budget  → invalid (too incomplete, re-download)
      - tokens >= 10% of budget → valid   (skip; shows % complete)
    """
    if not path.exists() or path.stat().st_size == 0:
        return False, "file is empty or missing"

    # Prefer the completeness sidecar written by write_jsonl: a source that was
    # EXHAUSTED is complete even if smaller than the budget (so we don't endlessly
    # re-download e.g. dialogue, which simply has fewer tokens than its budget).
    meta = path.with_suffix(".meta.json")
    if meta.exists():
        try:
            m = json.loads(meta.read_text())
            toks_m = m.get("tokens", 0) / 1e6
            if m.get("exhausted") or m.get("tokens", 0) >= token_budget_m * 1e6 * 0.9:
                kind = "exhausted" if m.get("exhausted") else "complete"
                return True, f"~{toks_m:.0f}M tokens ({kind})"
            return False, (f"partial ~{toks_m:.0f}M (< 90% of {token_budget_m}M) — re-downloading")
        except (json.JSONDecodeError, OSError):
            pass  # fall back to the size heuristic below

    # Validate first line is parseable JSONL
    try:
        with open(path, encoding="utf-8") as f:
            first = f.readline().strip()
        if not first:
            return False, "file has no content"
        rec = json.loads(first)
        if "text" not in rec:
            return False, "missing 'text' key — wrong format"
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return False, f"corrupted JSONL ({e})"

    # Fallback (no sidecar): estimate tokens from file size. INACCURATE — assumes
    # CHARS_PER_TOKEN uniformly, but prose is ~3.8 and structured data (arithmetic,
    # JSON) ~1.2 chars/token, so this can mis-judge completeness by 2-3×. Only used
    # for legacy files written before .meta.json existed; the sidecar above is exact.
    size_bytes = path.stat().st_size
    est_tokens_m = size_bytes / (CHARS_PER_TOKEN * 1_000_000)
    target_m = token_budget_m
    pct = est_tokens_m / target_m * 100 if target_m else 100

    if est_tokens_m < target_m * 0.10:
        return False, (
            f"only ~{est_tokens_m:.0f}M tokens ({pct:.1f}% of {target_m}M target) — too incomplete"
        )

    return True, f"~{est_tokens_m:.0f}M tokens ({pct:.0f}% of {target_m}M)"


def write_jsonl(
    records,
    out_path: Path,
    token_budget_m: int,
    verbose: bool = True,
    val_fraction: float = 0.02,
) -> tuple[int, bool]:
    """Write records to JSONL until the token budget is reached. Returns
    (tokens_written, exhausted) — `exhausted` is True when the source ran out
    BEFORE the budget (so a smaller-than-budget file is complete, not partial).

    A small `val_fraction` of records is routed to a sibling `{stem}.val.jsonl`
    held-out file (deterministic 1-in-K), so the gate can measure generalization on
    data the model never trains on. The budget counts TRAINING tokens only."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    val_path = out_path.with_suffix(".val.jsonl")
    every_k = round(1 / val_fraction) if val_fraction > 0 else 0  # 1-in-K → val
    tokens_written = 0
    target = token_budget_m * 1_000_000
    n = 0
    t0 = time.time()
    last_emit = 0.0
    exhausted = True  # set False if we stop on the budget

    def _emit(final: bool = False) -> None:
        # Single live line, refreshed ~2×/s, so streaming shows continuous progress
        # (the old code only printed every 10k docs — long silences on slow streams).
        if not verbose:
            return
        elapsed = max(time.time() - t0, 1e-6)
        pct = min(tokens_written / target * 100, 100) if target else 0.0
        rate = n / elapsed
        msg = (
            f"    {out_path.stem:<18} {tokens_written / 1e6:5.1f}M / {token_budget_m}M tok "
            f"({pct:4.1f}%) · {n:,} docs · {rate:,.0f} docs/s · {elapsed:.0f}s"
        )
        # \r keeps it on one line; pad to clear any leftover from a longer prior line.
        print("\r" + msg.ljust(78), end=("\n" if final else ""), flush=True)

    if verbose:
        print(
            f"    {out_path.stem}: connecting / streaming… (first shards can take a moment)",
            flush=True,
        )

    fval = open(val_path, "w", encoding="utf-8") if every_k else None  # noqa: SIM115 (closed in finally)
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                # Single normalization + garbage gate for EVERY source (current and
                # future): consistent surface form in, clearly-broken lines dropped.
                text = clean_record_text(rec.get("text", ""))
                if not text:
                    continue
                row = (
                    json.dumps({"text": text, "lang": rec.get("lang", "en")}, ensure_ascii=False)
                    + "\n"
                )
                n += 1
                if fval and n % every_k == 0:  # held-out — not counted in budget
                    fval.write(row)
                    continue
                f.write(row)
                tokens_written += estimate_tokens(text)
                now = time.time()
                if now - last_emit >= 0.5:  # time-based → lively even on slow streams
                    last_emit = now
                    _emit()
                if tokens_written >= target:
                    exhausted = False
                    break
    finally:
        if fval:
            fval.close()
        _emit(final=True)  # final newline + last numbers

    return tokens_written, exhausted
