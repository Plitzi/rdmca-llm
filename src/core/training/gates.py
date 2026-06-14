"""Graduation gates — masked validation perplexity, the inherited-baseline (entry)
PP, the ratcheting decision, and the absolute per-stage check (plus the BCF probe)."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

import src.core.backend as backend
from src.core.training.curriculum import stage_name
from src.plugins import bcf_stage, get_stage

BCF_STAGE = bcf_stage()


def validation_perplexity(model, val_batches) -> float:
    """Mean validation perplexity over a FIXED set of batches (the same arrays on
    every call), so the gate metric is comparable across evals. Batches are (tokens,
    mask) pairs; the mask makes the CE COMPLETION-ONLY, matching the training objective
    (bare token arrays without a mask are still accepted for back-compat)."""
    B = backend.current()
    losses = []
    for batch in val_batches:
        if isinstance(batch, tuple):
            toks, mask = batch
            loss = model.eval_ce(B.ops.array(toks), mask=B.ops.array(mask))
        else:
            loss = model.eval_ce(B.ops.array(batch))
        B.engine.eval(loss)
        losses.append(B.engine.item(loss))
    return float(np.exp(np.mean(losses)))


def stage_entry_ppl(model, ckpt_dir: Path, val_batches, resume: bool, log=print) -> float:
    """The stage's ENTRY perplexity: the inherited checkpoint's val PP on THIS stage's
    gate set, measured ONCE before any training. The gate's absolute PP carries an
    offset from the previous stage plus the rehearsal mix, so the meaningful signal is
    the delta from this baseline. Persisted to entry.json and reused on --resume."""
    path = Path(ckpt_dir) / "entry.json"
    if resume and path.exists():
        try:
            return float(json.loads(path.read_text())["entry_ppl"])
        except Exception:
            pass
    B = backend.current()
    B.engine.set_eval(model)
    ppl = validation_perplexity(model, val_batches)
    B.engine.set_train(model)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"entry_ppl": ppl, "timestamp": time.time()}, indent=2))
    except OSError:
        pass
    log(
        f"[gate] starting ppl = {ppl:.2f} (inherited from the previous stage) — the gate "
        f"shows ↓/↑ % vs this, so you see if THIS stage improved its own starting point"
    )
    return ppl


# Proxy perplexity gates per stage until task-specific benchmarks (BLiMP / ARC /
# GSM8K / COPA / BCF probes) are wired in. Overridable via cfg["gate"]["max_perplexity"].
# These are STARTING POINTS (viability floors), not the quality target: the gate
# RATCHETS (beat-your-own-best), so the floor stays lenient and never soft-locks a
# working model. 6 = memory, 7 = ethics/BCF.
DEFAULT_GATE_PPL = {1: 50.0, 2: 45.0, 3: 40.0, 4: 38.0, 5: 36.0, 6: 36.0, 7: 35.0}


def gate_threshold(stage: int, cfg: dict | None = None) -> float:
    """The stage's STARTING-POINT floor (cfg.gate.max_perplexity > default): the bar the
    first pass must clear and the minimum the best must clear to graduate. The in-loop
    gate then RATCHETS below it (beat-your-own-best); see gate_decision."""
    gate_cfg = (cfg or {}).get("gate", {})
    return gate_cfg.get("max_perplexity", {}).get(stage, DEFAULT_GATE_PPL.get(stage, 40.0))


def gate_decision(
    score: float, best_score: float, threshold: float, min_delta: float = 0.002
) -> tuple:
    """Ratcheting graduation gate. Returns (is_candidate, is_new_best, is_meaningful):
    • is_candidate  — `score` clears the absolute floor (`threshold`), so it is
      ELIGIBLE to be a best at all. An above-floor point is not viable and can NEVER
      become the best.
    • is_new_best   — is_candidate AND STRICTLY beats the running best. ANY genuine
      improvement is a new best and is SAVED (the ratchet bar moves down).
    • is_meaningful — is_new_best AND the gain exceeds `min_delta` (relative). Governs
      PLATEAU/early-stop only; it does NOT gate saving."""
    is_candidate = math.isfinite(score) and score <= threshold
    is_new_best = is_candidate and score < best_score
    is_meaningful = is_new_best and score < best_score * (1.0 - min_delta)
    return is_candidate, is_new_best, is_meaningful


def evaluate_gate(
    model, stage: int, val_batches=None, cfg: dict | None = None, log=print, step=None
) -> tuple:
    """Absolute graduation check on masked validation perplexity (the proxy that
    actually measures the model). The ethics/BCF stage additionally checks BCF probe
    accuracy when a probe set is available. Returns (score, meets_bar)."""
    # Stages without a benchmark gate fall back to the perplexity-only proxy with a
    # generic label (taken from the stage registry / config).
    gate = get_stage(stage).gate
    desc = gate.label if gate else stage_name(stage, cfg)
    max_ppl = gate_threshold(stage, cfg)

    # Measure with dropout OFF (eval mode), then restore training mode — the gate is
    # called mid-training, so validation must not see dropout noise.
    B = backend.current()
    B.engine.set_eval(model)
    ppl = validation_perplexity(model, val_batches)
    B.engine.set_train(model)
    passed = ppl <= max_ppl
    tag = f"step={step:,} | " if step is not None else ""
    log(
        f"[gate] {tag}val perplexity={ppl:.2f} <= {max_ppl:.1f} -> {'PASS' if passed else 'fail'}  ({desc})"
    )

    if stage == BCF_STAGE:
        passed = passed and _bcf_gate(model, cfg)
    return ppl, passed


def _bcf_gate(model, cfg: dict) -> bool:
    """Ethics/BCF probe-accuracy gate (>= 0.90) when probes are available."""
    probe_path = Path("data/benchmarks/bcf_probes.jsonl")
    if not probe_path.exists():
        print("  [gate] BCF probes not found — skipping BCF accuracy check")
        return True
    from src.core.modalities.text import TextTokenizer
    from src.core.model.bcf import BCFHead, bcf_accuracy

    probes = []
    with open(probe_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            probes.append((rec["text"], int(rec["label"])))
    head = getattr(model, "bcf_head", None) or BCFHead(model.cfg.d_model)
    acc = bcf_accuracy(model, TextTokenizer(), head, probes)
    print(f"  [gate] BCF probe accuracy={acc:.3f} | threshold>= 0.90")
    return acc >= 0.90
