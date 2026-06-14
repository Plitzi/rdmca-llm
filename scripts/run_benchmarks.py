#!/usr/bin/env python3
"""
RDMCA Benchmark Harness
=======================
Standard external benchmarks, run against a trained checkpoint so the model's
EVOLUTION across levels/stages is measurable on the same yardsticks the field uses.
A level-1 (~11M, preschool) model scores near chance on most of these — that's
expected; the POINT is the trend (L1 → L2 → … should climb), not the absolute.

Benchmarks (each best-effort — skipped with a note if its dataset is offline):
  • wikitext   — language-model perplexity on wikitext-2 test (lower = better)
  • lambada    — long-range last-word prediction accuracy
  • mmlu       — 4-way multiple-choice accuracy (length-normalized logprob), chance 0.25
  • gsm8k      — grade-school math, exact-match on the final number (generation)
  • mt_bench   — multi-turn chat quality; needs a strong external JUDGE model, so it is
                 only scored when --judge-cmd is given (otherwise reported as skipped)

Results are written to dist/benchmarks/level{L}_stage{N}.json and appended to
dist/benchmarks/history.csv so the evolution can be plotted/diffed over time.

Usage:
  .venv/bin/python scripts/run_benchmarks.py --level 1 --stage 5
  .venv/bin/python scripts/run_benchmarks.py --level 1 --stage 5 --benchmarks wikitext lambada
  .venv/bin/python scripts/run_benchmarks.py --checkpoint dist/checkpoints/level1/stage5/best.npz --level 1
"""

from __future__ import annotations

import os
import sys

_venv = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv", "bin", "python"
)
if os.path.exists(_venv) and os.path.abspath(sys.executable) != os.path.abspath(_venv):
    os.execv(_venv, [_venv, *sys.argv])

import argparse
import csv
import json
import re
import time
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ALL = ["wikitext", "lambada", "mmlu", "gsm8k", "mt_bench"]


# ── scoring primitives ─────────────────────────────────────────────────────────


def _logits_np(model, ids):
    """Full-sequence logits [S, V] as numpy for a single token sequence. Uses the
    backend's to_numpy (logits are bf16 at inference; numpy has no bfloat16)."""
    import src.core.backend as backend

    ops = backend.current().ops
    out = model.logits(ops.array(np.asarray([ids], dtype=np.int64)))
    return ops.to_numpy(out)[0]


def _logsoftmax_row(row):
    m = row.max()
    z = row - m
    return z - np.log(np.exp(z).sum())


def _continuation_logprob(model, prompt_ids, cont_ids) -> float:
    """Sum log P(cont | prompt) — the model's score for `cont_ids` following
    `prompt_ids`. Used to rank multiple-choice options and last-word candidates."""
    ids = list(prompt_ids) + list(cont_ids)
    if len(ids) < 2:
        return float("-inf")
    lg = _logits_np(model, ids)
    total = 0.0
    for i, tid in enumerate(cont_ids):
        pos = len(prompt_ids) + i - 1  # position whose logits predict tid
        total += float(_logsoftmax_row(lg[pos])[tid])
    return total


def _try_dataset(load, log):
    """Run a dataset loader, returning None (with a skip note) on any failure."""
    try:
        return load()
    except Exception as e:
        log(f"  [skip] dataset unavailable ({type(e).__name__}: {e})")
        return None


# ── benchmarks ──────────────────────────────────────────────────────────────────


def bench_wikitext(model, tok, limit, log) -> dict:
    """Token-level perplexity on wikitext-2 test (sliding 256-token windows)."""
    from datasets import load_dataset

    ds = _try_dataset(
        lambda: load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test"), log
    ) or _try_dataset(lambda: load_dataset("wikitext", "wikitext-2-raw-v1", split="test"), log)
    if ds is None:
        return {"skipped": True}
    import src.core.backend as backend

    ops, engine = backend.current().ops, backend.current().engine
    text = "\n".join(r["text"] for r in ds if r["text"].strip())
    ids = tok.encode(text, add_bos=False, add_eos=False)
    win = 256
    n = min(len(ids), (limit or 50) * win)
    losses = []
    engine.set_eval(model)
    for s in range(0, n - 1, win):
        chunk = ids[s : s + win + 1]
        if len(chunk) < 2:
            continue
        loss = model.eval_ce(ops.array(np.asarray([chunk], dtype=np.int64)))
        engine.eval(loss)
        losses.append(engine.item(loss))
    engine.set_train(model)
    ppl = float(np.exp(np.mean(losses))) if losses else float("nan")
    log(f"  wikitext: ppl={ppl:.2f} over {len(losses)} windows")
    return {"ppl": ppl, "windows": len(losses)}


def bench_lambada(model, tok, limit, log) -> dict:
    """Last-word prediction accuracy: greedily continue the context and check the
    predicted word matches the held-out final word."""
    from datasets import load_dataset

    ds = _try_dataset(
        lambda: load_dataset("EleutherAI/lambada_openai", "en", split="test"), log
    ) or _try_dataset(lambda: load_dataset("lambada", split="test"), log)
    if ds is None:
        return {"skipped": True}
    correct = total = 0
    for r in ds.select(range(min(limit or 200, len(ds)))):
        text = r["text"].strip()
        if " " not in text:
            continue
        ctx, _, last = text.rpartition(" ")
        target = re.sub(r"[^\w]", "", last).lower()
        if not target:
            continue
        prompt = tok.encode(ctx + " ", add_bos=True, add_eos=False)
        cont = tok.encode(last, add_bos=False, add_eos=False)
        # greedy-decode len(cont) tokens and compare the decoded word
        gen = []
        ids = list(prompt)
        for _ in range(max(1, len(cont)) + 2):
            row = _logits_np(model, ids)[-1]
            nxt = int(row.argmax())
            gen.append(nxt)
            ids.append(nxt)
        pred = re.sub(
            r"[^\w]", "", tok.decode(gen).split()[0] if tok.decode(gen).split() else ""
        ).lower()
        correct += int(pred == target)
        total += 1
    acc = correct / total if total else float("nan")
    log(f"  lambada: acc={acc:.3f} ({correct}/{total})")
    return {"acc": acc, "n": total}


def bench_mmlu(model, tok, limit, log) -> dict:
    """4-way multiple choice: pick the option with the highest length-normalized
    continuation logprob. Chance = 0.25."""
    from datasets import load_dataset

    ds = _try_dataset(lambda: load_dataset("cais/mmlu", "all", split="test"), log)
    if ds is None:
        return {"skipped": True}
    correct = total = 0
    for r in ds.select(range(min(limit or 200, len(ds)))):
        q, choices, ans = r["question"], r["choices"], int(r["answer"])
        prompt = tok.encode(f"Question: {q}\nAnswer:", add_bos=True, add_eos=False)
        scores = []
        for ch in choices:
            cont = tok.encode(" " + str(ch), add_bos=False, add_eos=False)
            lp = _continuation_logprob(model, prompt, cont)
            scores.append(lp / max(1, len(cont)))  # length-normalized
        correct += int(int(np.argmax(scores)) == ans)
        total += 1
    acc = correct / total if total else float("nan")
    log(f"  mmlu: acc={acc:.3f} ({correct}/{total}) — chance 0.25")
    return {"acc": acc, "n": total}


def bench_gsm8k(model, tok, limit, log, generate) -> dict:
    """Grade-school math: generate a solution and exact-match the final integer."""
    from datasets import load_dataset

    ds = _try_dataset(lambda: load_dataset("gsm8k", "main", split="test"), log)
    if ds is None:
        return {"skipped": True}
    correct = total = 0
    last_num = re.compile(r"-?\d[\d,]*")
    for r in ds.select(range(min(limit or 100, len(ds)))):
        gold = last_num.findall(r["answer"].split("####")[-1])
        if not gold:
            continue
        gold_n = gold[-1].replace(",", "")
        prompt = tok.encode(
            f"\nUser: {r['question']}\nAssistant:", lang="en", add_bos=True, add_eos=False
        )
        ids, _ = generate(
            model,
            list(prompt),
            max_new_tokens=160,
            temperature=0.0,
            top_p=1.0,
            vocab_size=model.cfg.vocab_size,
            context_len=model.cfg.context_len,
            stream=False,
            decode_fn=tok.decode,
            top_k=1,
            rep_penalty=1.0,
        )
        nums = last_num.findall(tok.decode(ids))
        pred = nums[-1].replace(",", "") if nums else None
        correct += int(pred == gold_n)
        total += 1
    acc = correct / total if total else float("nan")
    log(f"  gsm8k: acc={acc:.3f} ({correct}/{total})")
    return {"acc": acc, "n": total}


def bench_mt_bench(model, tok, limit, log, judge_cmd) -> dict:
    """MT-Bench needs a strong external judge model to grade open-ended multi-turn
    answers. Without one there is no automatable score, so we skip with a clear note
    rather than emit a misleading number. Provide --judge-cmd to enable (the command
    receives the generated answers as JSON on stdin and must return a 1–10 score)."""
    if not judge_cmd:
        log("  [skip] mt_bench needs an external judge — pass --judge-cmd to enable")
        return {"skipped": True, "reason": "no judge configured"}
    log("  [skip] mt_bench judge integration is a stub — wire --judge-cmd to your judge")
    return {"skipped": True, "reason": "judge stub"}


# ── driver ────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="Run external benchmarks on a checkpoint.")
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--stage", type=int, default=None)
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--benchmarks", nargs="*", default=ALL, choices=ALL)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap examples per benchmark (speed; default per-bench)",
    )
    ap.add_argument("--judge-cmd", type=str, default=None, help="external judge for mt_bench")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not args.checkpoint and args.stage is None:
        print("Specify --stage N or --checkpoint PATH")
        sys.exit(1)

    from src.core.config import resolve_config_path
    from src.core.modalities.text import TextTokenizer
    from uses.chat.run_chat import generate, load_model

    la = Namespace(
        config=resolve_config_path(None, args.level),
        level=args.level,
        stage=args.stage,
        checkpoint=args.checkpoint,
        dummy=False,
        quant="none",
        force=args.force,
    )
    print("Loading model…")
    model, _ = load_model(la)
    tok = TextTokenizer()
    if not tok.ready:
        print("Tokenizer not trained — aborting.")
        sys.exit(1)

    results, t0 = {}, time.time()
    for name in args.benchmarks:
        print(f"[bench] {name} …")
        try:
            if name == "wikitext":
                results[name] = bench_wikitext(model, tok, args.limit, print)
            elif name == "lambada":
                results[name] = bench_lambada(model, tok, args.limit, print)
            elif name == "mmlu":
                results[name] = bench_mmlu(model, tok, args.limit, print)
            elif name == "gsm8k":
                results[name] = bench_gsm8k(model, tok, args.limit, print, generate)
            elif name == "mt_bench":
                results[name] = bench_mt_bench(model, tok, args.limit, print, args.judge_cmd)
        except Exception as e:
            print(f"  [error] {name}: {type(e).__name__}: {e}")
            results[name] = {"error": f"{type(e).__name__}: {e}"}

    record = {
        "level": args.level,
        "stage": args.stage,
        "checkpoint": args.checkpoint,
        "timestamp": time.time(),
        "elapsed_s": round(time.time() - t0, 1),
        "results": results,
    }

    out_dir = ROOT / "dist" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else out_dir / f"level{args.level}_stage{args.stage}.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"\nSaved → {out}")

    # Append a flat row to history.csv so evolution across levels/stages is plottable.
    hist = out_dir / "history.csv"
    cols = ["timestamp", "level", "stage", "wikitext_ppl", "lambada_acc", "mmlu_acc", "gsm8k_acc"]
    new = not hist.exists()
    with open(hist, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(cols)

        def g(b, k):
            return results.get(b, {}).get(k, "")

        w.writerow(
            [
                round(record["timestamp"]),
                args.level,
                args.stage,
                g("wikitext", "ppl"),
                g("lambada", "acc"),
                g("mmlu", "acc"),
                g("gsm8k", "acc"),
            ]
        )
    print(f"History → {hist}")


if __name__ == "__main__":
    main()
