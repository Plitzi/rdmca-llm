"""Full (unfiltered) external corpus streamers — Wikipedia, GSM8K, MATH, ethics, ARC.

These back the HIGHER levels (the lower levels use the per-stage synthetic/graded
sources in each stage plugin's sources.py). prepare_data passes them to
`stages.stream_source` as `extra_streamers`, so a stage can draw from a big external
corpus without the streamer living in the CLI. Wikipedia is routed to a stage by
category keywords (STAGE_KEYWORDS).
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterator

# HuggingFace token — optional but raises rate limits significantly. Supplied via the
# HF_TOKEN environment variable (set it in .env — see .env.example). Never prompted:
# missing just means slower downloads.
_HF_TOKEN: str | None = None


def setup_hf_token() -> None:
    """Read the HF token from the environment (loaded from .env). Always optional;
    never prompts. Set HF_TOKEN in .env to raise download rate limits."""
    global _HF_TOKEN
    from src.core.env import load_env

    load_env()  # ensure .env is loaded before reading
    _HF_TOKEN = os.environ.get("HF_TOKEN", "").strip() or None
    if _HF_TOKEN:
        print("  HF token: found in environment (rate limits enabled)")
    else:
        print("  HF token: not set — continuing without (slower). Set HF_TOKEN in .env to enable.")


# Keywords that trigger inclusion — works for both EN and ES text (Spanish Wikipedia
# articles often contain EN loan words and vice-versa).
STAGE_KEYWORDS = {
    1: None,
    2: [
        "science",
        "ciencia",
        "mathematics",
        "matemática",
        "logic",
        "lógica",
        "analogy",
        "analogía",
        "pattern",
        "patrón",
        "similarity",
        "similitud",
        "perception",
        "percepción",
        "cognitive",
        "cognitivo",
    ],
    3: [
        "mathematics",
        "matemática",
        "algebra",
        "álgebra",
        "calculus",
        "cálculo",
        "logic",
        "lógica",
        "proof",
        "demostración",
        "theorem",
        "teorema",
        "algorithm",
        "algoritmo",
        "formal",
        "symbolic",
        "simbólico",
        "abstract",
    ],
    4: [
        "cause",
        "causa",
        "causal",
        "effect",
        "efecto",
        "mechanism",
        "mecanismo",
        "engineering",
        "ingeniería",
        "medicine",
        "medicina",
        "physics",
        "física",
        "chemistry",
        "química",
        "procedure",
        "procedimiento",
        "process",
        "proceso",
    ],
    # 5 = Reasoning (chain-of-thought) — sourced from GSM8K, not Wikipedia, so no
    # keyword gate (article_matches_stage returns True for stages absent here).
    6: [
        "ethics",
        "ética",
        "moral",
        "philosophy",
        "filosofía",
        "rights",
        "derechos",
        "justice",
        "justicia",
        "law",
        "derecho",
        "harm",
        "daño",
        "value",
        "valor",
        "norm",
        "norma",
        "social",
    ],
}


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
                "wikimedia/wikipedia",
                dump,
                split="train",
                streaming=True,
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


def stream_wikipedia(lang: str = "en", limit_mb: int | None = None) -> Iterator[dict]:
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
                print("  Run the script again — it will resume from where it left off.")
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
                yield {"text": f"Problem: {ex['question']}\nSolution: {ex['answer']}", "lang": "en"}
        except Exception as e:
            print(f"    [gsm8k-en] {e}")

        # Spanish mGSM
        try:
            ds = load_dataset("juletxara/mgsm", "es", split=split)
            for ex in ds:
                yield {
                    "text": f"Problema: {ex['question']}\nSolución: {ex['answer_number']}",
                    "lang": "es",
                }
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
                    yield {
                        "text": f"Problem: {ex['problem']}\nSolution: {ex['solution']}",
                        "lang": "en",
                    }
            except Exception:
                pass
    except Exception as e:
        print(f"    [math] {e}")


def stream_ethics_bilingual() -> Iterator[dict]:
    """Public-domain ethics snippets — bilingual — ethics/BCF stage (stage 6)."""
    snippets = [
        (
            "The categorical imperative: Act only according to that maxim by which you can at the same time will that it should become a universal law.",
            "en",
        ),
        (
            "El imperativo categórico: actúa solo según aquella máxima que puedas querer que se convierta en ley universal.",
            "es",
        ),
        (
            "Utilitarianism holds that the right action is the one that produces the greatest good for the greatest number.",
            "en",
        ),
        (
            "El utilitarismo sostiene que la acción correcta es aquella que produce el mayor bien para el mayor número de personas.",
            "es",
        ),
        (
            "Virtue ethics focuses on the character of the moral agent rather than rules or consequences.",
            "en",
        ),
        (
            "La ética de la virtud se centra en el carácter del agente moral más que en las reglas o las consecuencias.",
            "es",
        ),
        (
            "Harm principle: power can only be exercised over a member of a community, against his will, to prevent harm to others.",
            "en",
        ),
        (
            "Principio del daño: el poder solo puede ejercerse contra la voluntad de alguien para prevenir daño a terceros.",
            "es",
        ),
        (
            "Justice as fairness: principles of justice are those rational persons would accept in an initial position of equality.",
            "en",
        ),
        (
            "La justicia como equidad: los principios de justicia son aquellos que personas racionales aceptarían en una posición inicial de igualdad.",
            "es",
        ),
        (
            "Non-maleficence: refrain from actions that cause harm. Beneficence: take positive steps to help others.",
            "en",
        ),
        (
            "No maleficencia: abstenerse de acciones dañinas. Beneficencia: tomar medidas positivas para ayudar a otros.",
            "es",
        ),
    ]
    for text, lang in snippets:
        yield {"text": text, "lang": lang}


def _arc_subset(subset: str):
    """Full-corpus streamer for one ARC subset (ARC-Easy | ARC-Challenge)."""

    def gen():
        try:
            from datasets import load_dataset

            ds = load_dataset("allenai/ai2_arc", subset, split="train")
            for ex in ds:
                q = ex["question"]
                labels = ex["choices"]["label"]
                texts = ex["choices"]["text"]
                answer = ex.get("answerKey", "")
                options = "  ".join(f"{l}: {t}" for l, t in zip(labels, texts, strict=False))
                correct = next((t for l, t in zip(labels, texts, strict=False) if l == answer), "")
                yield {
                    "text": f"Question: {q}\nOptions: {options}\nAnswer: {correct}",
                    "lang": "en",
                }
        except Exception as e:
            print(f"    [arc {subset}] {e}")

    return gen


def full_corpus_streamers(stage: int, langs: list[str], limit_mb: int | None) -> dict:
    """Streamers for the FULL (unfiltered) corpora, keyed by source name. These back
    the higher levels; the stage plugins' sources handle the simple/synthetic sources
    for the lower levels."""

    def wikipedia():
        for lang in langs:
            for art in stream_wikipedia(lang, limit_mb=limit_mb):
                if article_matches_stage(art["title"], art["text"], stage):
                    yield art

    return {
        "wikipedia": wikipedia,
        "arc_easy": _arc_subset("ARC-Easy"),
        "arc_challenge": _arc_subset("ARC-Challenge"),
        "gsm8k": stream_gsm8k_bilingual,
        "math": stream_math,
        "ethics": stream_ethics_bilingual,
    }
