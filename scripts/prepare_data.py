#!/usr/bin/env python3
from __future__ import annotations
import sys, os
_venv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv", "bin", "python")
if os.path.exists(_venv) and os.path.abspath(sys.executable) != os.path.abspath(_venv):
    os.execv(_venv, [_venv] + sys.argv)

"""
RDMCA Data Preparation Script — config-driven language set
===========================================================
Downloads and processes all 5 curriculum stage datasets for the languages in
the config (`model.languages`), or a `--lang` override.
Output: data/stage{N}_*/  in .jsonl format  {"text": "...", "lang": "<code>"}

Strategy
--------
Wikipedia (one dump per configured language) is the backbone for all stages.
Each article is tagged with its language and routed to the correct stage
by category keywords (bilingual — same keywords trigger in both languages).
Small task-specific datasets are mixed in for domain-specific stages.

Stage → Wikipedia categories:
  1  Language    — everything (general language baseline)
  2  Patterns    — ciencia/science, lógica/logic, analogía/analogy
  3  Abstraction — matemáticas/mathematics, lógica/logic, álgebra/algebra
  4  Causal      — ingeniería/engineering, medicina/medicine, física/physics
  5  Ethics      — ética/ethics, filosofía/philosophy, derecho/law

Token targets (total EN+ES):
  Stage 1: 1.5B   Stage 2: 500M   Stage 3: 1B   Stage 4: 1B   Stage 5: 500M

Usage:
  python scripts/prepare_data.py --stage all
  python scripts/prepare_data.py --stage 1
  python scripts/prepare_data.py --stage 1 --limit 100   # 100MB test run
  python scripts/prepare_data.py --stage 1 --lang en     # English only
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterator, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TOKEN_BUDGET_M = {1: 1500, 2: 500, 3: 1000, 4: 1000, 5: 500}

# HuggingFace token — optional but raises rate limits significantly.
# Set via env var or prompted once at startup.
_HF_TOKEN: Optional[str] = None


def _setup_hf_token() -> None:
    """Check env var, prompt once if not set. Token is always optional."""
    global _HF_TOKEN
    _HF_TOKEN = os.environ.get("HF_TOKEN", "").strip() or None
    if _HF_TOKEN:
        print(f"  HF token: found in environment (rate limits enabled)")
        return

    print()
    print("  HuggingFace token (optional — press Enter to skip)")
    print("  ───────────────────────────────────────────────────")
    print("  How to get one (free):")
    print("    1. Create a free account at huggingface.co")
    print("    2. Go to huggingface.co/settings/tokens")
    print("    3. New token → type Read → copy it here")
    print("  Benefit: higher rate limits and faster downloads.")
    print("  Without a token the download still works, just slower.")
    print()
    try:
        token = input("  HF_TOKEN: ").strip()
    except (EOFError, KeyboardInterrupt):
        token = ""

    if token:
        _HF_TOKEN = token
        os.environ["HF_TOKEN"] = token
        print("  Token set for this session.")
    else:
        print("  Continuing without token.\n")

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

    # Estimate token count from file size
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
    5: ["ethics", "ética", "moral", "philosophy", "filosofía", "rights", "derechos",
        "justice", "justicia", "law", "derecho", "harm", "daño", "value", "valor",
        "norm", "norma", "social"],
}

CHARS_PER_TOKEN = 4.5


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


def stream_arc(split: str = "train") -> Iterator[dict]:
    """ARC Easy + Challenge (EN) — Stage 2."""
    try:
        from datasets import load_dataset
    except ImportError:
        return
    for subset in ("ARC-Easy", "ARC-Challenge"):
        try:
            ds = load_dataset("allenai/ai2_arc", subset, split=split)
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
            print(f"    [arc] {subset}: {e}")


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
    """Public-domain ethics snippets — bilingual — Stage 5."""
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
                token_budget_m: int, verbose: bool = True) -> int:
    """Write records to JSONL until token budget is reached. Returns tokens written."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokens_written = 0
    target = token_budget_m * 1_000_000
    n = 0
    t0 = time.time()

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            text = rec.get("text", "")
            if not text.strip():
                continue
            f.write(json.dumps({"text": text, "lang": rec.get("lang", "en")},
                               ensure_ascii=False) + "\n")
            tokens_written += estimate_tokens(text)
            n += 1
            if verbose and n % 10_000 == 0:
                pct = min(tokens_written / target * 100, 100)
                elapsed = time.time() - t0
                print(f"    {tokens_written/1e6:.0f}M / {token_budget_m}M tokens "
                      f"({pct:.1f}%)  {n:,} docs  {elapsed:.0f}s")
            if tokens_written >= target:
                break

    return tokens_written


def prepare_stage(stage: int, langs: List[str],
                  limit_mb: Optional[int] = None) -> None:
    budget = TOKEN_BUDGET_M[stage]
    lang_str = "+".join(l.upper() for l in langs)
    print(f"\n{'='*60}")
    print(f"Stage {stage} [{lang_str}] — target: {budget}M tokens")
    print(f"{'='*60}")

    dir_map = {
        1: Path("data/stage1_language"),
        2: Path("data/stage2_patterns"),
        3: Path("data/stage3_abstraction"),
        4: Path("data/stage4_causal"),
        5: Path("data/stage5_ethics"),
    }
    out_dir = dir_map[stage]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Budget split: equitable across the configured languages.
    per_lang = budget // max(len(langs), 1)

    # Wikipedia per language
    for lang in langs:
        wiki_path   = out_dir / f"wikipedia_{lang}.jsonl"
        lang_budget = per_lang

        ok, reason = _validate_jsonl(wiki_path, lang_budget)
        if ok:
            print(f"  OK (valid): {wiki_path.name}  —  {reason}")
            continue
        elif wiki_path.exists():
            print(f"  Invalid file detected: {wiki_path.name}  —  {reason}")
            print(f"  Removing and re-downloading…")
            wiki_path.unlink()

        print(f"  Writing {wiki_path} …")

        def filtered_wiki(l=lang):
            for article in stream_wikipedia(l, limit_mb=limit_mb):
                if article_matches_stage(article["title"], article["text"], stage):
                    yield article

        tokens = write_jsonl(filtered_wiki(), wiki_path, lang_budget)
        print(f"  Wikipedia {lang.upper()}: {tokens/1e6:.0f}M tokens → {wiki_path}")

    # Stage-specific task datasets (only the ones matching configured languages).
    if stage == 2 and "en" in langs:
        arc_path = out_dir / "arc_en.jsonl"
        if not arc_path.exists():
            tokens = write_jsonl(stream_arc(), arc_path, 10)
            print(f"  ARC (EN): {tokens/1e6:.2f}M tokens → {arc_path}")

    if stage == 3 and ("en" in langs or "es" in langs):
        gsm_path = out_dir / "gsm8k_tasks.jsonl"
        if not gsm_path.exists():
            tokens = write_jsonl(stream_gsm8k_bilingual(), gsm_path, 10)
            print(f"  GSM8K tasks: {tokens/1e6:.2f}M tokens → {gsm_path}")

        if "en" in langs:
            math_path = out_dir / "competition_math_en.jsonl"
            if not math_path.exists():
                tokens = write_jsonl(stream_math(), math_path, 20)
                print(f"  MATH (EN): {tokens/1e6:.2f}M tokens → {math_path}")

    if stage == 5 and ("en" in langs or "es" in langs):
        ethics_path = out_dir / "ethics_tasks.jsonl"
        if not ethics_path.exists():
            tokens = write_jsonl(stream_ethics_bilingual(), ethics_path, 1)
            print(f"  Ethics tasks: {tokens/1e6:.2f}M tokens → {ethics_path}")

    print(f"  Stage {stage} data ready in {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="RDMCA curriculum data preparation")
    parser.add_argument("--stage", default="all",
                        help="Stage number (1-5) or 'all'")
    parser.add_argument("--config", default=None,
                        help="Config path (languages source of truth)")
    parser.add_argument("--profile", default=None,
                        help="Hardware profile: nano | m2max | test | …")
    parser.add_argument("--lang", default=None,
                        help="Comma-separated override of config languages")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit each Wikipedia stream to N MB (testing)")
    args = parser.parse_args()

    _setup_hf_token()

    # Languages: --lang override > config(model.languages) > ['en']
    from src.config import resolve_config_path, load_config, get_languages
    if args.lang:
        langs = [l.strip() for l in args.lang.split(",")]
    else:
        langs = get_languages(load_config(resolve_config_path(args.config, args.profile)))
    stages = list(range(1, 6)) if args.stage == "all" else [int(args.stage)]

    print(f"Languages: {langs}")
    _NETWORK_ERRORS = (
        "RemoteProtocolError", "ConnectError", "ReadTimeout",
        "ConnectionError", "ServerDisconnected", "TimeoutError",
    )

    try:
        for s in stages:
            prepare_stage(s, langs=langs, limit_mb=args.limit)
    except KeyboardInterrupt:
        print("\n\nInterrupted. Run the same command again to resume.")
        sys.exit(0)
    except Exception as e:
        if type(e).__name__ in _NETWORK_ERRORS or "disconnected" in str(e).lower():
            print(f"\nNetwork error: {e}")
            print("Run the same command again to resume — files already written are kept.")
            sys.exit(1)
        raise   # anything else: show full traceback

    print("\nDone. Next: python scripts/train_tokenizer.py")
    sys.stdout.flush()
    sys.stderr.flush()
    # The HuggingFace datasets streaming iterators leave multiprocessing
    # SemLock objects dangling when a stream is closed early (e.g. on the MB
    # limit). Force a GC pass so their finalizers run and unregister from the
    # resource_tracker — otherwise the forced os._exit() below skips that
    # cleanup and the tracker prints a spurious "leaked semaphore" warning.
    import gc
    gc.collect()
    # Force exit — the HuggingFace datasets library leaves background
    # threads running after streaming ends, which blocks normal exit.
    os._exit(0)


if __name__ == "__main__":
    main()
