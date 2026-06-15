"""Auxiliary heads + the stage-completion seam: the BCF head, the conversation mood
head, and the side effects when a stage finishes (persist sector / freeze the core)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import src.core.backend as backend
from src.core.training.checkpoint import freeze_model
from src.core.training.curriculum import BCF_STAGE, is_behavioral_stage, last_cognitive_stage
from src.models import get_stage


def train_bcf_head(
    model, ckpt_dir: Path, precision: str = "fp32", epochs: int = 30, batch: int = 16
) -> None:
    """Train the Behavioral Constraint head on the probe set over frozen-core features
    (§15.3). Runs only if data/benchmarks/bcf_probes.jsonl exists; the trained head is
    stored on model.bcf_head and saved beside the stage."""
    probe_path = Path("data/benchmarks/bcf_probes.jsonl")
    if not probe_path.exists():
        print(
            "  [bcf] No probe set — skipping BCF head training (expected data/benchmarks/bcf_probes.jsonl)"
        )
        return
    from src.core.modalities.text import TextTokenizer
    from src.core.model.bcf import BCFHead, bcf_accuracy, bcf_train_step

    probes = []
    with open(probe_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                probes.append((rec["text"], int(rec["label"])))

    head = BCFHead(model.cfg.d_model)
    backend.current().engine.set_precision(head, precision)  # match model device/dtype
    model.bcf_head = head  # attach for gate + pipeline use
    opt = backend.current().engine.make_optimizer(head, lr=1e-3, weight_decay=0.0)
    tok = TextTokenizer()
    print(f"  [bcf] Training BCF head on {len(probes)} probes, {epochs} epochs")
    for _epoch in range(epochs):
        np.random.shuffle(probes)
        for i in range(0, len(probes), batch):
            bcf_train_step(model, tok, head, probes[i : i + batch], opt)
    acc = bcf_accuracy(model, tok, head, probes)
    print(f"  [bcf] final probe accuracy={acc:.3f}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    backend.current().engine.save_weights(head, str(ckpt_dir / "bcf_head.npz"))


def maybe_train_mood_head(model, stage: int, cfg: dict, ckpt_dir: Path, precision: str) -> None:
    """Train + save the conversation mood head beside this stage's checkpoint. Best-
    effort and OFF the critical path: gated by `training.mood_head` (default on) and
    silently skipped if the labeled data is unavailable — it must never fail a stage."""
    if not cfg.get("training", {}).get("mood_head", True):
        return
    # Mood is a CONVERSATIONAL faculty — only (re)train it on conversational stages
    # (stage 1 + the frozen-core BCF stage). The stage plugin declares this.
    if not get_stage(stage).trains_mood:
        print(
            f"  [mood] stage {stage} is not conversational — keeping the existing "
            "head (chat falls back to the nearest earlier head + lexicon)"
        )
        return
    try:
        from src.core.modalities.text import TextTokenizer
        from src.core.model.mood import train_mood_head as _train_mood

        _train_mood(
            model,
            TextTokenizer(),
            ckpt_dir,
            level=cfg.get("level"),
            stage=stage,
            precision=precision,
        )
    except Exception as e:
        print(f"  [mood] skipped ({type(e).__name__}: {e})")


def on_stage_complete(
    model, stage: int, cfg: dict, root: Path, ckpt_dir: Path, precision: str, adapter=None
) -> None:
    """Side effects when a stage finishes: a behavioral stage persists its trained
    sector; the last cognitive stage trains the BCF head (if ethics is active) and
    freezes the foundational core. This is the single freeze/sector seam."""
    from src.core.model import sector_io

    if is_behavioral_stage(stage):
        if adapter is not None:
            print(f"  Behavioral sector saved: {sector_io.save_sector(adapter, root, stage)}")
        return
    # Cognitive stage finished: train the conversation mood head on this checkpoint's
    # core so the stage is chat-ready with mood tracking — no separate script needed.
    maybe_train_mood_head(model, stage, cfg, ckpt_dir, precision)
    if stage == last_cognitive_stage(cfg):
        if stage == BCF_STAGE:
            train_bcf_head(model, ckpt_dir, precision)
        freeze_model(model, root / "foundational")
