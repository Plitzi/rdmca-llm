#!/usr/bin/env python3
"""
RDMCA Training Metrics Plotter
==============================
Charts the training curves a stage emits to `metrics.csv` (written next to each
stage's train.log by the dashboard): training loss + per-token perplexity over
tokens, and the gate's validation perplexity + running best over steps.

The CSV has two row kinds:
  train,step,tokens_m,loss,ppl,lr,tps,grad_norm,,,replay,
  gate,step,tokens_m,,,,,,val_ppl,best_val_ppl,passed,,entry_ppl

`entry_ppl` is the stage's inherited-baseline perplexity (measured before training):
the gate plot draws it as a reference line so improvement reads as the gap below it.

Usage:
  python scripts/plot_metrics.py --level 1                # every stage with metrics
  python scripts/plot_metrics.py --level 1 --stage 1      # one stage
  python scripts/plot_metrics.py --level 1 --stage 1 3 5  # a few
  python scripts/plot_metrics.py --level 1 --overview     # whole-curriculum panorama
  python scripts/plot_metrics.py --csv path/to/metrics.csv

Output: a PNG next to each metrics.csv (stageN/metrics.png), the overview at
level{L}/overview.png, or --out PATH.
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
                             "best": num("best_val_ppl"), "passed": num("passed"),
                             "entry_ppl": num("entry_ppl")})
    return train, gate


def _entry_ppl(gate) -> Optional[float]:
    """The stage ENTRY (inherited-baseline) perplexity, if recorded on any gate row."""
    for r in gate:
        if r.get("entry_ppl") is not None:
            return r["entry_ppl"]
    return None


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
    # Entry baseline (inherited PP): progress is the gap BELOW this line. Above it the
    # stage has regressed its own starting point (offset-corrected reading).
    entry = _entry_ppl(gate)
    if entry is not None:
        ax.axhline(entry, color="#7f7f7f", lw=1.2, ls=":", label=f"entry PP {entry:.1f}")
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


def plot_overview(level: int, out: Optional[Path], plt) -> Optional[Path]:
    """The WHOLE-CURRICULUM panorama: one figure across every trained stage so you
    can see the model's evolution end-to-end. Top — each stage's ENTRY PP vs its final
    best PP (grouped bars + Δ%), the offset-corrected 'did this stage improve its own
    starting point?' view. Bottom — the gate validation-PP timeline concatenated across
    stages on a global step axis, with stage boundaries marked."""
    import json
    base = ROOT / "dist" / "checkpoints" / f"level{level}"
    stages = sorted(int(p.name.replace("stage", "")) for p in base.glob("stage*")
                    if (p / "metrics.csv").exists())
    if not stages:
        print("No trained stages found for an overview.")
        return None

    per = []                                   # (stage, entry, best, gate_rows)
    for s in stages:
        _, gate = _read_metrics(base / f"stage{s}" / "metrics.csv")
        entry = _entry_ppl(gate)
        best = None
        sc = base / f"stage{s}" / "stage_complete.json"
        if sc.exists():
            try:
                d = json.loads(sc.read_text())
                best = d.get("gate_score") or d.get("best_val_ppl")
                entry = d.get("entry_ppl", entry)
            except (OSError, ValueError):
                pass
        if best is None:
            bxs, bys = _series(gate, "step", "best")
            best = bys[-1] if bys else None
        per.append((s, entry, best, gate))

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(13, 9))
    fig.suptitle(f"RDMCA — full curriculum panorama (level {level})",
                 fontsize=14, fontweight="bold")

    # ── Top: entry vs best PP per stage (offset-corrected improvement) ──
    xs = list(range(len(per)))
    entries = [(e if e is not None else float("nan")) for _, e, _, _ in per]
    bests   = [(b if b is not None else float("nan")) for _, _, b, _ in per]
    w = 0.38
    ax0.bar([x - w/2 for x in xs], entries, width=w, color="#7f7f7f", label="entry PP (inherited)")
    ax0.bar([x + w/2 for x in xs], bests,   width=w, color="#2ca02c", label="best PP (stage end)")
    for x, (s, e, b, _) in zip(xs, per):
        if e and b and e > 0:
            ax0.annotate(f"{(b-e)/e*100:+.0f}%", (x, max(e, b)),
                         ha="center", va="bottom", fontsize=8,
                         color=("#2ca02c" if b <= e else "#d62728"))
    ax0.set_xticks(xs); ax0.set_xticklabels([f"S{s}" for s, *_ in per])
    ax0.set_ylabel("val perplexity"); ax0.set_title("Per-stage entry vs final best PP (Δ% = offset-corrected change)")
    ax0.legend(fontsize=8); ax0.grid(alpha=0.3, axis="y")

    # ── Bottom: gate val-PP timeline concatenated across stages ──
    offset, ticks, tlabels = 0, [], []
    for s, _, _, gate in per:
        gx, gy = _series(gate, "step", "val_ppl")
        if not gx:
            continue
        xx = [offset + g for g in gx]
        ax1.plot(xx, gy, lw=1.2, marker="o", ms=2, label=f"S{s}")
        ax1.axvline(offset, color="#cccccc", lw=0.8, ls=":")
        ticks.append(offset + (gx[-1] / 2 if gx else 0)); tlabels.append(f"S{s}")
        offset += (gx[-1] if gx else 0) + max(1, (gx[-1] if gx else 0) * 0.04)
    ax1.set_xticks(ticks); ax1.set_xticklabels(tlabels)
    ax1.set_ylabel("gate val ppl"); ax1.set_title("Gate validation perplexity across the curriculum")
    ax1.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = out or (base / "overview.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  ✓ overview: {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Plot RDMCA training metrics.")
    ap.add_argument("--level", type=int, default=1, help="level (default 1)")
    ap.add_argument("--stage", type=int, nargs="*", default=None,
                    help="stage(s) to plot; omit for every stage with a metrics.csv")
    ap.add_argument("--csv", type=str, default=None, help="plot a specific metrics.csv")
    ap.add_argument("--out", type=str, default=None, help="output PNG (single-target only)")
    ap.add_argument("--overview", action="store_true",
                    help="one whole-curriculum panorama across all trained stages")
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

    if args.overview:
        plot_overview(args.level, Path(args.out) if args.out else None, plt)
        return

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
