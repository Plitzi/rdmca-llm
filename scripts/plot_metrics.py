#!/usr/bin/env python3
"""
RDMCA Training Metrics Plotter
==============================
Charts the training curves a stage emits to `metrics.csv` (written next to each
stage's train.log by the dashboard): training loss + per-token perplexity over
tokens, and the gate's validation perplexity + running best over steps.

The CSV has two row kinds:
  train,step,tokens_m,loss,ppl,lr,tps,grad_norm,,,
  gate,step,tokens_m,,,,,,val_ppl,best_val_ppl,passed

Usage:
  python scripts/plot_metrics.py --level 1                # every stage with metrics
  python scripts/plot_metrics.py --level 1 --stage 1      # one stage
  python scripts/plot_metrics.py --level 1 --stage 1 3 5  # a few
  python scripts/plot_metrics.py --csv path/to/metrics.csv

Output: a PNG next to each metrics.csv (stageN/metrics.png), or --out PATH.
Needs matplotlib (`pip install matplotlib`); without it the script explains how
to install and exits cleanly.
"""
from __future__ import annotations
import sys, os
_venv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv", "bin", "python")
if os.path.exists(_venv) and os.path.abspath(sys.executable) != os.path.abspath(_venv):
    os.execv(_venv, [_venv] + sys.argv)

import argparse
import csv
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent


def _read_metrics(csv_path: Path):
    """Parse a metrics.csv into (train_rows, gate_rows), each a list of float dicts."""
    train, gate = [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            def num(k):
                v = (row.get(k) or "").strip()
                try:
                    return float(v)
                except ValueError:
                    return None
            if row.get("kind") == "train":
                train.append({"step": num("step"), "tokens_m": num("tokens_m"),
                              "loss": num("loss"), "ppl": num("ppl"),
                              "lr": num("lr"), "tps": num("tps"),
                              "grad_norm": num("grad_norm"),
                              "replay": num("replay")})
            elif row.get("kind") == "gate":
                gate.append({"step": num("step"), "val_ppl": num("val_ppl"),
                             "best": num("best_val_ppl"), "passed": num("passed")})
    return train, gate


def _series(rows, x, y):
    xs, ys = [], []
    for r in rows:
        if r.get(x) is not None and r.get(y) is not None:
            xs.append(r[x]); ys.append(r[y])
    return xs, ys


def plot_stage(csv_path: Path, out: Optional[Path], label: str, plt) -> Optional[Path]:
    train, gate = _read_metrics(csv_path)
    if not train and not gate:
        print(f"  [skip] {csv_path} has no data yet")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"RDMCA training metrics — {label}", fontsize=13, fontweight="bold")

    # (0,0) training loss vs tokens — SPLIT by batch type. With rehearsal the per-step
    # loss is bimodal (narrow-skill batches near 0, interleaved conversation-replay much
    # higher); plotting them together looks like wild "spikes". Split into two clean
    # curves: the skill loss should fall to ~0; the rehearsal (conversation) loss should
    # stay flat/fall — if it CLIMBS, conversation is being forgotten.
    ax = axes[0][0]
    has_replay = any(r.get("replay") is not None for r in train)
    if has_replay:
        skill   = [r for r in train if (r.get("replay") or 0) < 0.5]
        rehears = [r for r in train if (r.get("replay") or 0) >= 0.5]
        sx, sy = _series(skill, "tokens_m", "loss")
        rx, ry = _series(rehears, "tokens_m", "loss")
        if sx:
            ax.plot(sx, sy, color="#1f77b4", lw=1.0, alpha=0.8, label="new-skill batches")
        if rx:
            ax.plot(rx, ry, color="#d62728", lw=1.0, alpha=0.8,
                    label="conversation rehearsal")
        if sx or rx:
            ax.legend(fontsize=8)
        ax.set_title("Training loss (split: skill vs rehearsal — the 'spikes')")
    else:
        xs, ys = _series(train, "tokens_m", "loss")
        if xs:
            ax.plot(xs, ys, color="#1f77b4", lw=1.3)
        ax.set_title("Training loss")
    ax.set_xlabel("tokens (M)"); ax.set_ylabel("loss")
    ax.grid(alpha=0.3)

    # (0,1) live per-token perplexity vs tokens (log scale — it spans orders of magnitude)
    ax = axes[0][1]
    xs, ys = _series(train, "tokens_m", "ppl")
    if xs:
        ax.plot(xs, ys, color="#ff7f0e", lw=1.3)
        ax.set_yscale("log")
    ax.set_title("Live per-token perplexity"); ax.set_xlabel("tokens (M)"); ax.set_ylabel("ppl (log)")
    ax.grid(alpha=0.3, which="both")

    # (1,0) gate: validation perplexity + running best vs step (the authoritative metric)
    ax = axes[1][0]
    xs, ys = _series(gate, "step", "val_ppl")
    if xs:
        ax.plot(xs, ys, color="#2ca02c", lw=1.3, marker="o", ms=3, label="val ppl")
    bxs, bys = _series(gate, "step", "best")
    if bxs:
        ax.plot(bxs, bys, color="#d62728", lw=1.5, ls="--", label="best (ratchet bar)")
    # mark the checkpoints the gate accepted as a new best
    pxs, pys = [], []
    for r in gate:
        if r.get("passed") and r.get("step") is not None and r.get("val_ppl") is not None:
            pxs.append(r["step"]); pys.append(r["val_ppl"])
    if pxs:
        ax.scatter(pxs, pys, color="#d62728", s=40, zorder=5, label="new best")
    ax.set_title("Gate: validation perplexity"); ax.set_xlabel("step"); ax.set_ylabel("val ppl")
    if xs or bxs:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (1,1) learning rate vs step
    ax = axes[1][1]
    xs, ys = _series(train, "step", "lr")
    if xs:
        ax.plot(xs, ys, color="#9467bd", lw=1.3)
    ax.set_title("Learning rate"); ax.set_xlabel("step"); ax.set_ylabel("lr")
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = out or csv_path.with_name("metrics.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  ✓ {label}: {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Plot RDMCA training metrics.")
    ap.add_argument("--level", type=int, default=1, help="level (default 1)")
    ap.add_argument("--stage", type=int, nargs="*", default=None,
                    help="stage(s) to plot; omit for every stage with a metrics.csv")
    ap.add_argument("--csv", type=str, default=None, help="plot a specific metrics.csv")
    ap.add_argument("--out", type=str, default=None, help="output PNG (single-target only)")
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")                       # headless: write a file, no display
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed. Install it with:\n"
              "  .venv/bin/python -m pip install matplotlib\n"
              "(metrics.csv is still written every run, so you can re-plot any time.)")
        sys.exit(1)

    targets: list[tuple[Path, str]] = []
    if args.csv:
        targets.append((Path(args.csv), Path(args.csv).parent.name))
    else:
        base = ROOT / "dist" / "checkpoints" / f"level{args.level}"
        stages = args.stage or sorted(
            int(p.name.replace("stage", "")) for p in base.glob("stage*")
            if (p / "metrics.csv").exists())
        for s in stages:
            csv_path = base / f"stage{s}" / "metrics.csv"
            if csv_path.exists():
                targets.append((csv_path, f"level {args.level} · stage {s}"))
            else:
                print(f"  [skip] no metrics.csv for stage {s} (train it first)")

    if not targets:
        print("No metrics.csv found. Run training first (it writes one per stage).")
        sys.exit(1)
    out = Path(args.out) if args.out and len(targets) == 1 else None
    for csv_path, label in targets:
        plot_stage(csv_path, out, label, plt)


if __name__ == "__main__":
    main()
