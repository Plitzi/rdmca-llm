#!/usr/bin/env python3
"""
RDMCA Data Preparation Script — config-driven, per level + stage
================================================================
Writes the training corpus for each curriculum stage of a LEVEL, using the
sources, complexity filter and token budget declared in that level's config
(`configs/levels/levelN.yaml`). Output:
  data/level{L}/stage{N}/{source}.jsonl   {"text": "...", "lang": "<code>"}
plus a {source}.meta.json sidecar recording token count + whether the source
was exhausted (used to decide if a re-run can skip it).

Where the data comes from is per-source, resolved in `src/data/graded.py`:
  - Lower levels use simple/graded sources (tinystories, dialogue, arithmetic,
    analogies, agentic/MCP/skills, reasoning) — small, conversational/structured.
  - Higher levels add the FULL corpora (Wikipedia per configured language, ARC,
    GSM8K, MATH, ethics), with Wikipedia routed to a stage by category keywords
    (STAGE_KEYWORDS) and prose readability-graded (Flesch-Kincaid).

Token budgets come from each stage's `n_tokens` in the config (NOT hardcoded
here), split across the stage's sources.

Usage:
  python scripts/prepare_data.py --level 1 --stage all
  python scripts/prepare_data.py --level 1 --stage 1
  python scripts/prepare_data.py --level 1 --stage 1 --limit 100  # 100MB test
  python scripts/prepare_data.py --level 1 --stage 1 --lang en    # English only
"""
# Re-exec into the project venv BEFORE importing third-party deps, so the script
# works when launched with a bare `python` outside the venv. (Kept after the
# docstring so `__doc__` / argparse epilog / pydoc see the help text.)
from __future__ import annotations
import sys, os
_venv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv", "bin", "python")
if os.path.exists(_venv) and os.path.abspath(sys.executable) != os.path.abspath(_venv):
    os.execv(_venv, [_venv] + sys.argv)

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterator, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# HuggingFace token — optional but raises rate limits significantly.
# Supplied via the HF_TOKEN environment variable (set it in .env — see
# .env.example). Never prompted: missing just means slower downloads.
_HF_TOKEN: Optional[str] = None


def _setup_hf_token() -> None:
    """Read the HF token from the environment (loaded from .env). Always optional;
    never prompts. Set HF_TOKEN in .env to raise download rate limits."""
    global _HF_TOKEN
    from src.env import load_env
    load_env()                                  # ensure .env is loaded before reading
    _HF_TOKEN = os.environ.get("HF_TOKEN", "").strip() or None
    if _HF_TOKEN:
        print("  HF token: found in environment (rate limits enabled)")
    else:
        print("  HF token: not set — continuing without (slower). "
              "Set HF_TOKEN in .env to enable.")

def _validate_jsonl(path: Path, token_budget_m: int) -> tuple[bool, str]:
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
            return False, (f"partial ~{toks_m:.0f}M (< 90% of {token_budget_m}M) "
                           "— re-downloading")
        except (json.JSONDecodeError, OSError):
            pass   # fall back to the size heuristic below

    # Validate first line is parseable JSONL
    try:
        with open(path, "r", encoding="utf-8") as f:
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
    size_bytes   = path.stat().st_size
    est_tokens_m = size_bytes / (CHARS_PER_TOKEN * 1_000_000)
    target_m     = token_budget_m
    pct          = est_tokens_m / target_m * 100 if target_m else 100

    if est_tokens_m < target_m * 0.10:
        return False, (f"only ~{est_tokens_m:.0f}M tokens "
                       f"({pct:.1f}% of {target_m}M target) — too incomplete")

    return True, f"~{est_tokens_m:.0f}M tokens ({pct:.0f}% of {target_m}M)"


# Keywords that trigger inclusion — works for both EN and ES text
# (Spanish Wikipedia articles often contain EN loan words and vice-versa)
STAGE_KEYWORDS = {
    1: None,
    2: ["science", "ciencia", "mathematics", "matemática", "logic", "lógica",
        "analogy", "analogía", "pattern", "patrón", "similarity", "similitud",
        "perception", "percepción", "cognitive", "cognitivo"],
    3: ["mathematics", "matemática", "algebra", "álgebra", "calculus", "cálculo",
        "logic", "lógica", "proof", "demostración", "theorem", "teorema",
        "algorithm", "algoritmo", "formal", "symbolic", "simbólico", "abstract"],
    4: ["cause", "causa", "causal", "effect", "efecto", "mechanism", "mecanismo",
        "engineering", "ingeniería", "medicine", "medicina", "physics", "física",
        "chemistry", "química", "procedure", "procedimiento", "process", "proceso"],
    # 5 = Reasoning (chain-of-thought) — sourced from GSM8K, not Wikipedia, so no
    # keyword gate (article_matches_stage returns True for stages absent here).
    6: ["ethics", "ética", "moral", "philosophy", "filosofía", "rights", "derechos",
        "justice", "justicia", "law", "derecho", "harm", "daño", "value", "valor",
        "norm", "norma", "social"],
}

# Empirically ~3.91 chars/token for prose (tinystories/dialogue) based on actual
# tokenizer measurements (3.5 under-counted tokens by ~12%). Budgets remain
# approximate for structured stages — the runtime corpus cap uses real token counts.
CHARS_PER_TOKEN = 3.9


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN)


def article_matches_stage(title: str, text: str, stage: int) -> bool:
    keywords = STAGE_KEYWORDS.get(stage)
    if keywords is None:
        return True
    combined = (title + " " + text[:500]).lower()
    return any(kw in combined for kw in keywords)


def _load_wikipedia_with_retry(lang: str, max_retries: int = 5):
    """Load Wikipedia dataset with exponential backoff on network errors."""
    from datasets import load_dataset

    dump = f"20231101.{lang}"
    wait = 5
    for attempt in range(1, max_retries + 1):
        try:
            return load_dataset(
                "wikimedia/wikipedia", dump,
                split="train", streaming=True,
                trust_remote_code=False,
                token=_HF_TOKEN,
            )
        except Exception as e:
            if attempt == max_retries:
                raise
            print(f"  [retry {attempt}/{max_retries}] Network error: {e}")
            print(f"  Waiting {wait}s before retry…")
            time.sleep(wait)
            wait = min(wait * 2, 120)


def stream_wikipedia(lang: str = "en",
                     limit_mb: Optional[int] = None) -> Iterator[dict]:
    """Stream Wikipedia articles for a given language code."""
    try:
        from datasets import load_dataset  # noqa: F401 — just checking import
    except ImportError:
        print("ERROR: run pip install datasets first")
        sys.exit(1)

    dump = f"20231101.{lang}"
    print(f"  Loading Wikipedia {lang.upper()} ({dump}, streaming)…")
    ds = _load_wikipedia_with_retry(lang)
    bytes_seen = 0
    limit_bytes = (limit_mb * 1024 * 1024) if limit_mb else None
    retries_left = 3

    while True:
        try:
            for article in ds:
                text = article.get("text", "")
                if len(text) < 200:
                    continue
                yield {"title": article.get("title", ""), "text": text, "lang": lang}
                bytes_seen += len(text.encode())
                if limit_bytes and bytes_seen >= limit_bytes:
                    print(f"    [limit] {lang.upper()} reached {limit_mb}MB — stopping")
                    return
            return  # exhausted normally
        except Exception as e:
            if retries_left <= 0:
                print(f"  [error] Stream interrupted and retries exhausted: {e}")
                print(f"  Run the script again — it will resume from where it left off.")
                return
            retries_left -= 1
            print(f"  [stream error] Connection dropped — reconnecting in 10s… ({e})")
            time.sleep(10)
            ds = _load_wikipedia_with_retry(lang)


def stream_gsm8k_bilingual(split: str = "train") -> Iterator[dict]:
    """GSM8K (EN) + mGSM Spanish — Stage 3."""
    try:
        from datasets import load_dataset

        # English GSM8K
        try:
            ds = load_dataset("openai/gsm8k", "main", split=split)
            for ex in ds:
                yield {"text": f"Problem: {ex['question']}\nSolution: {ex['answer']}",
                       "lang": "en"}
        except Exception as e:
            print(f"    [gsm8k-en] {e}")

        # Spanish mGSM
        try:
            ds = load_dataset("juletxara/mgsm", "es", split=split)
            for ex in ds:
                yield {"text": f"Problema: {ex['question']}\nSolución: {ex['answer_number']}",
                       "lang": "es"}
        except Exception as e:
            print(f"    [mgsm-es] {e}")

    except Exception as e:
        print(f"    [gsm8k] {e}")


def stream_math(split: str = "train") -> Iterator[dict]:
    """Hendrycks competition math — Stage 3."""
    try:
        from datasets import load_dataset
        for level in ("algebra", "counting_and_probability", "number_theory"):
            try:
                ds = load_dataset("EleutherAI/hendrycks_math", level, split=split)
                for ex in ds:
                    yield {"text": f"Problem: {ex['problem']}\nSolution: {ex['solution']}",
                           "lang": "en"}
            except Exception:
                pass
    except Exception as e:
        print(f"    [math] {e}")


def stream_ethics_bilingual() -> Iterator[dict]:
    """Public-domain ethics snippets — bilingual — ethics/BCF stage (stage 6)."""
    snippets = [
        ("The categorical imperative: Act only according to that maxim by which you can at the same time will that it should become a universal law.", "en"),
        ("El imperativo categórico: actúa solo según aquella máxima que puedas querer que se convierta en ley universal.", "es"),
        ("Utilitarianism holds that the right action is the one that produces the greatest good for the greatest number.", "en"),
        ("El utilitarismo sostiene que la acción correcta es aquella que produce el mayor bien para el mayor número de personas.", "es"),
        ("Virtue ethics focuses on the character of the moral agent rather than rules or consequences.", "en"),
        ("La ética de la virtud se centra en el carácter del agente moral más que en las reglas o las consecuencias.", "es"),
        ("Harm principle: power can only be exercised over a member of a community, against his will, to prevent harm to others.", "en"),
        ("Principio del daño: el poder solo puede ejercerse contra la voluntad de alguien para prevenir daño a terceros.", "es"),
        ("Justice as fairness: principles of justice are those rational persons would accept in an initial position of equality.", "en"),
        ("La justicia como equidad: los principios de justicia son aquellos que personas racionales aceptarían en una posición inicial de igualdad.", "es"),
        ("Non-maleficence: refrain from actions that cause harm. Beneficence: take positive steps to help others.", "en"),
        ("No maleficencia: abstenerse de acciones dañinas. Beneficencia: tomar medidas positivas para ayudar a otros.", "es"),
    ]
    for text, lang in snippets:
        yield {"text": text, "lang": lang}


def write_jsonl(records: Iterator[dict], out_path: Path,
                token_budget_m: int, verbose: bool = True,
                val_fraction: float = 0.02) -> tuple[int, bool]:
    """Write records to JSONL until the token budget is reached. Returns
    (tokens_written, exhausted) — `exhausted` is True when the source ran out
    BEFORE the budget (so a smaller-than-budget file is complete, not partial).

    A small `val_fraction` of records is routed to a sibling `{stem}.val.jsonl`
    held-out file (deterministic 1-in-K), so the gate can measure generalization on
    data the model never trains on. The budget counts TRAINING tokens only."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    val_path  = out_path.with_suffix(".val.jsonl")
    every_k   = int(round(1 / val_fraction)) if val_fraction > 0 else 0   # 1-in-K → val
    tokens_written = 0
    target = token_budget_m * 1_000_000
    n = 0
    t0 = time.time()
    last_emit = 0.0
    exhausted = True                          # set False if we stop on the budget

    def _emit(final: bool = False) -> None:
        # Single live line, refreshed ~2×/s, so streaming shows continuous progress
        # (the old code only printed every 10k docs — long silences on slow streams).
        if not verbose:
            return
        elapsed = max(time.time() - t0, 1e-6)
        pct  = min(tokens_written / target * 100, 100) if target else 0.0
        rate = n / elapsed
        msg = (f"    {out_path.stem:<18} {tokens_written/1e6:5.1f}M / {token_budget_m}M tok "
               f"({pct:4.1f}%) · {n:,} docs · {rate:,.0f} docs/s · {elapsed:.0f}s")
        # \r keeps it on one line; pad to clear any leftover from a longer prior line.
        print("\r" + msg.ljust(78), end=("\n" if final else ""), flush=True)

    if verbose:
        print(f"    {out_path.stem}: connecting / streaming… "
              "(first shards can take a moment)", flush=True)

    fval = open(val_path, "w", encoding="utf-8") if every_k else None
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                text = rec.get("text", "")
                if not text.strip():
                    continue
                row = json.dumps({"text": text, "lang": rec.get("lang", "en")},
                                 ensure_ascii=False) + "\n"
                n += 1
                if fval and n % every_k == 0:        # held-out — not counted in budget
                    fval.write(row)
                    continue
                f.write(row)
                tokens_written += estimate_tokens(text)
                now = time.time()
                if now - last_emit >= 0.5:           # time-based → lively even on slow streams
                    last_emit = now
                    _emit()
                if tokens_written >= target:
                    exhausted = False
                    break
    finally:
        if fval:
            fval.close()
        _emit(final=True)                            # final newline + last numbers

    return tokens_written, exhausted


def _arc_subset(subset: str):
    """Full-corpus streamer for one ARC subset (ARC-Easy | ARC-Challenge)."""
    def gen():
        try:
            from datasets import load_dataset
            ds = load_dataset("allenai/ai2_arc", subset, split="train")
            for ex in ds:
                q       = ex["question"]
                labels  = ex["choices"]["label"]
                texts   = ex["choices"]["text"]
                answer  = ex.get("answerKey", "")
                options = "  ".join(f"{l}: {t}" for l, t in zip(labels, texts))
                correct = next((t for l, t in zip(labels, texts) if l == answer), "")
                yield {"text": f"Question: {q}\nOptions: {options}\nAnswer: {correct}",
                       "lang": "en"}
        except Exception as e:
            print(f"    [arc {subset}] {e}")
    return gen


def _full_corpus_streamers(stage: int, langs: List[str],
                           limit_mb: Optional[int]) -> dict:
    """Streamers for the FULL (unfiltered) corpora, keyed by source name. These
    back the higher levels; `graded.stream_source` handles the simple/synthetic
    sources for the lower levels."""
    def wikipedia():
        for lang in langs:
            for art in stream_wikipedia(lang, limit_mb=limit_mb):
                if article_matches_stage(art["title"], art["text"], stage):
                    yield art
    return {
        "wikipedia":     wikipedia,
        "arc_easy":      _arc_subset("ARC-Easy"),
        "arc_challenge": _arc_subset("ARC-Challenge"),
        "gsm8k":         stream_gsm8k_bilingual,
        "math":          stream_math,
        "ethics":        stream_ethics_bilingual,
    }


# Sources that are free prose and SHOULD be readability-graded. Everything else
# (dialogue, tool/skill/MCP JSON, arithmetic, analogies, causal) is conversational
# or structured and must NOT be grade-filtered — see the gate below.
_READABILITY_FILTERED = {"tinystories", "simple_wikipedia", "wikipedia"}


def prepare_stage_for_level(level: int, stage: int, cfg: dict,
                            langs: List[str], limit_mb: Optional[int] = None) -> None:
    """Prepare graded data for one (level, stage), reading the level config's
    curriculum entry: which sources, the complexity filter, the token budget and
    the output dir. Skips stages whose entry_level is above this level."""
    from src.data import graded

    cur = cfg.get("curriculum", {}) or {}
    skey = f"stage{stage}"
    if skey not in cur:
        print(f"  Stage {stage}: not active at level {level} — skipping.")
        return
    sc    = cur[skey]
    entry = int(sc.get("entry_level", 1))
    if entry > level:
        print(f"  Stage {stage}: enters at level {entry} (> {level}) — skipping.")
        return

    data    = sc.get("data", {}) or {}
    sources = data.get("sources", []) or []
    flt     = data.get("filter")                       # None at level 5
    arith   = (flt or {}).get("arithmetic_level", level) if isinstance(flt, dict) else level
    out_dir = Path(sc["data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    budget_m     = max(int(sc.get("n_tokens", 100_000_000) // 1_000_000), 1)
    per_source_m = max(budget_m // max(len(sources), 1), 1)
    extra        = _full_corpus_streamers(stage, langs, limit_mb)

    print(f"\n{'='*60}")
    print(f"Level {level} · Stage {stage}: {sc.get('name','')}")
    print(f"  sources={sources}  filter={flt}  budget~{budget_m}M tokens → {out_dir}/")
    print(f"{'='*60}")

    for src in sources:
        out_path = out_dir / f"{src}.jsonl"
        ok, reason = _validate_jsonl(out_path, per_source_m)
        if ok:
            print(f"  OK (valid): {out_path.name}  —  {reason}")
            continue
        it = graded.stream_source(
            src, langs=langs, n_tokens=per_source_m * 1_000_000,
            arithmetic_level=arith, limit_mb=limit_mb, extra_streamers=extra)
        if it is None:
            print(f"  [skip] unknown source '{src}'")
            continue
        # Readability (Flesch-Kincaid grade) gating only makes sense for free
        # PROSE. Applying grade≤2 to real human dialogue decimated it (it nearly
        # always scores above a preschool grade), starving the model of the
        # conversational `User:/Assistant:` format — so only prose sources are
        # graded; conversational/structured ones pass through.
        if flt and src in _READABILITY_FILTERED:       # None / non-prose ⇒ keep all
            it = (rec for rec in it if graded.passes_filter(rec.get("text", ""), flt))
        tokens, exhausted = write_jsonl(it, out_path, per_source_m)
        # Sidecar: records completeness so a re-run can tell "source exhausted"
        # (complete, smaller than budget) from "download interrupted" (partial).
        out_path.with_suffix(".meta.json").write_text(json.dumps(
            {"tokens": tokens, "budget_m": per_source_m, "exhausted": exhausted}))
        note = "exhausted" if exhausted else "budget reached"
        print(f"  {src}: {tokens/1e6:.1f}M tokens ({note}) → {out_path.name}")

    print(f"  Stage {stage} ready in {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="RDMCA curriculum data preparation")
    parser.add_argument("--level", type=int, default=None,
                        help="Educational level 1-5 (preescolar..universidad). "
                             "Determines the graded data sources + complexity.")
    parser.add_argument("--stage", default="all",
                        help="Stage number (1-5) or 'all'")
    parser.add_argument("--config", default=None,
                        help="Explicit config path (overrides --level)")
    parser.add_argument("--lang", default=None,
                        help="Comma-separated override of config languages")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit each Wikipedia stream to N MB (testing)")
    args = parser.parse_args()

    _setup_hf_token()

    from src.config import resolve_config_path, load_config, get_languages, get_level, MAX_LEVEL
    cfg_path = resolve_config_path(args.config, args.level)
    cfg      = load_config(cfg_path)
    level    = get_level(cfg)                       # NB: level 0 is valid → use `is None`
    if level is None:                               # custom config w/o a level → least filtering
        level = args.level if args.level is not None else MAX_LEVEL
    # Languages: --lang override > config(model.languages) > ['en']
    langs = ([l.strip() for l in args.lang.split(",")] if args.lang
             else get_languages(cfg))
    # "all" → every stage declared in this level's curriculum (data-driven, so
    # new stages like agentic/MCP are picked up automatically).
    if args.stage == "all":
        stages = sorted(int(k.replace("stage", "")) for k in cfg.get("curriculum", {}))
    else:
        stages = [int(args.stage)]

    print(f"Level {level} ({cfg.get('name','custom')}) | languages: {langs} | config: {cfg_path}")
    _NETWORK_ERRORS = (
        "RemoteProtocolError", "ConnectError", "ReadTimeout",
        "ConnectionError", "ServerDisconnected", "TimeoutError",
    )

    try:
        for s in stages:
            prepare_stage_for_level(level, s, cfg, langs=langs, limit_mb=args.limit)
    except KeyboardInterrupt:
        print("\n\nInterrupted. Run the same command again to resume.")
        sys.exit(0)
    except Exception as e:
        if type(e).__name__ in _NETWORK_ERRORS or "disconnected" in str(e).lower():
            print(f"\nNetwork error: {e}")
            print("Run the same command again to resume — files already written are kept.")
            sys.exit(1)
        raise   # anything else: show full traceback

    print(f"\nDone. Next: python scripts/train_tokenizer.py --level {level}")
    sys.stdout.flush()
    sys.stderr.flush()
    # The HuggingFace datasets streaming iterators leave multiprocessing
    # SemLock objects dangling when a stream is closed early (e.g. on the MB
    # limit). Force a GC pass so their finalizers run and unregister from the
    # resource_tracker — otherwise the forced os._exit() below skips that
    # cleanup and the tracker prints a spurious "leaked semaphore" warning.
    import gc
    gc.collect()
    # os._exit() below skips normal cleanup, so any SemLock still registered
    # with the resource_tracker triggers a spurious "leaked semaphore" warning
    # when the tracker process detects our pipe closing. The tracker's resource
    # registry lives in *its* subprocess (not reachable from here), so the only
    # way to stop the warning is to kill that subprocess before it runs its
    # end-of-life check. The OS reclaims the leftover semaphores on exit anyway.
    try:
        import signal
        from multiprocessing import resource_tracker
        pid = getattr(resource_tracker._resource_tracker, "_pid", None)
        if pid is not None:
            os.kill(pid, signal.SIGKILL)
    except Exception:
        pass
    # Force exit — the HuggingFace datasets library leaves background
    # threads running after streaming ends, which blocks normal exit.
    os._exit(0)


if __name__ == "__main__":
    main()
