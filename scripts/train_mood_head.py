#!/usr/bin/env python3
from __future__ import annotations
# Re-exec into the project venv BEFORE importing third-party deps.
import os, sys
_venv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv", "bin", "python")
if os.path.exists(_venv) and os.path.abspath(sys.executable) != os.path.abspath(_venv):
    os.execv(_venv, [_venv] + sys.argv)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root

"""
Train the conversation MOOD HEAD on top of the FROZEN foundational core.

This is a cheap probe (minutes, not the ~1h LM retrain): the LM weights are read
frozen, only the small MoodHead learns. It lets the chat track the conversation's
mood (neutral by default) on the EXISTING checkpoint — no LM retrain required.

  python scripts/train_mood_head.py --level 1 --stage 1

Labels come from EmpatheticDialogues' emotion field (mapped onto the mood palette,
src/modalities/moods.py); NEUTRAL examples come from the already-prepared local
instruct / wikipedia / tinystories files (factual/narrative ⇒ no active mood).
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np


def _latest_ckpt(level: int, stage: int) -> str:
    d = Path("dist/checkpoints") / f"level{level}" / f"stage{stage}"
    latest = d / "latest.json"
    if latest.exists():
        p = json.loads(latest.read_text()).get("checkpoint")
        if p and Path(p).exists():
            return p
    npz = sorted(d.glob("step_*.npz"))
    if npz:
        return str(npz[-1])
    raise FileNotFoundError(f"no checkpoint in {d}")


def _local_neutral(level: int, stage: int, n: int) -> list:
    """Sample NEUTRAL texts from the already-prepared factual/narrative files."""
    base = Path("data") / f"level{level}" / f"stage{stage}"
    out = []
    for stem in ("instruct", "simple_wikipedia", "tinystories"):
        f = base / f"{stem}.jsonl"
        if not f.exists():
            continue
        with open(f) as fh:
            for line in fh:
                try:
                    t = json.loads(line).get("text", "")
                except json.JSONDecodeError:
                    continue
                # Train on the RAW user-side utterance (what the chat classifies at
                # inference): pull the `User:` line out of instruct transcripts, use
                # prose as-is. Factual/narrative ⇒ no active mood ⇒ neutral.
                if "User:" in t:
                    seg = t.split("User:", 1)[1].split("\nAssistant:", 1)[0].strip()
                else:
                    seg = t.strip()
                if seg:
                    out.append((seg[:300], "neutral"))
                if len(out) >= n:
                    break
        if len(out) >= n:
            break
    return out


def _emotional(per_emotion: int) -> list:
    """Stream EmpatheticDialogues, map each dialogue's emotion onto a mood. Uses the
    RAW transcript (no `(mood: …)` annotation) so the classifier can't cheat."""
    from collections import Counter
    from datasets import load_dataset
    from src.modalities.moods import emotion_to_mood
    out, counts = [], Counter()
    try:
        ds = load_dataset("Estwld/empathetic_dialogues_llm", split="train", streaming=True)
    except Exception as e:
        print(f"  [emotional] {e}")
        return out
    for ex in ds:
        emo = (ex.get("emotion") or "").strip().lower()
        mood = emotion_to_mood(emo)
        if mood == "neutral" or counts[mood] >= per_emotion:
            continue
        # Train on the speaker's RAW emotional utterance (the first user turn),
        # matching what the chat classifies at inference (a single user message).
        turns = [(c.get("role"), c.get("content")) for c in (ex.get("conversations") or [])]
        utt = next((c for r, c in turns if c and (r or "").lower() in
                    ("user", "human", "speaker", "0")), None)
        if not utt and turns:
            utt = turns[0][1]
        if utt and utt.strip():
            counts[mood] += 1
            out.append((utt.strip()[:300], mood))
        if all(counts[m] >= per_emotion for m in
               ("happy", "sad", "angry", "afraid", "surprised", "caring")):
            break
    return out


def main():
    ap = argparse.ArgumentParser(description="Train the mood head on the frozen core")
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--per-mood", type=int, default=300, help="emotional examples per mood")
    ap.add_argument("--neutral", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    import mlx.core as mx
    import src.backend as backend
    from src.config import resolve_config_path, load_config
    from src.model.transformer import RDMCAFoundational
    from src.model.config import ModelConfig
    from src.modalities.text import TextTokenizer
    from src.model.mood import (MoodHead, mood_loss, _pooled_states,
                                MOODS, MOOD_INDEX)
    B = backend.current()

    cfg = load_config(resolve_config_path(None, args.level))
    mc = cfg["model"]
    mcfg = ModelConfig(d_model=mc["d_model"], n_heads=mc["n_heads"], ffn_dim=mc["ffn_dim"],
                       context_len=mc["context_len"], vocab_size=mc["vocab_size"],
                       mrl_dims=mc["mrl_dims"], dropout=0.0)
    model = RDMCAFoundational(mcfg)
    ckpt = args.checkpoint or _latest_ckpt(args.level, args.stage)
    B.engine.load_weights(model, ckpt)
    B.engine.set_eval(model)
    print(f"  loaded core: {ckpt}")
    tok = TextTokenizer()

    print("  building labeled set…")
    data = _emotional(args.per_mood) + _local_neutral(args.level, args.stage, args.neutral)
    random.shuffle(data)
    if len(data) < 50:
        print("  ERROR: too few labeled examples (need HF EmpatheticDialogues).")
        sys.exit(1)
    from collections import Counter
    print(f"  {len(data)} examples · per-mood: {dict(Counter(m for _, m in data))}")

    # Precompute frozen pooled features ONCE (the LM doesn't change), then train the
    # small head directly on them — fast, no forward pass in the training loop.
    print("  extracting frozen features…")
    feats = np.array(_pooled_states(model, tok, [t for t, _ in data]))   # [N, d]
    labels = np.array([MOOD_INDEX[m] for _, m in data], dtype=np.float32)
    n_val = max(8, len(data) // 10)
    Xv, yv = mx.array(feats[:n_val]), labels[:n_val]
    Xt, yt = feats[n_val:], labels[n_val:]

    head = MoodHead(mcfg.d_model)
    opt  = B.engine.make_optimizer(head, args.lr, 0.0)

    def _acc(X, y):
        preds = np.array(mx.argmax(head(mx.array(X)), axis=-1))
        return float((preds == y).mean())

    idx = np.arange(len(Xt))
    for ep in range(args.epochs):
        np.random.shuffle(idx)
        losses = []
        for i in range(0, len(idx), args.batch):
            j = idx[i:i + args.batch]
            Hb, yb = mx.array(Xt[j]), mx.array(yt[j])
            def loss_fn(hd, H=Hb, Y=yb):
                return mood_loss(hd(H), Y)
            loss, grads = B.engine.value_and_grad(head, loss_fn)(head)
            B.engine.optimizer_step(opt, head, grads)
            losses.append(float(loss.item()))
        print(f"  epoch {ep+1}/{args.epochs}: loss={np.mean(losses):.3f} "
              f"train_acc={_acc(Xt, yt):.2f} val_acc={_acc(np.array(Xv), yv):.2f}")

    out = Path(ckpt).parent / "mood_head.npz"
    B.engine.save_weights(head, str(out))
    print(f"  saved → {out}")


if __name__ == "__main__":
    main()
