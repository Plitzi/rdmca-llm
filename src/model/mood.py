"""
Mood head — a lightweight emotional-state classifier on top of the foundational
hidden states (same probe pattern as the BCF safety head, see src/model/bcf.py).

Goal (RDMCA conversational layer): give the model a conversation-driven mood that
defaults to NEUTRAL and only shifts when the dialogue clearly carries an emotion —
the way a calm assistant stays neutral until something in the exchange warrants a
reaction. The mood is read from the frozen core (the LM is not retrained to host
it) and surfaced as a plain-text annotation on the SYSTEM channel
(`System: … (mood: happy)`), so it needs NO new tokenizer symbols and works with
the existing checkpoint.

Two pieces:
  • MoodHead   — multiclass classifier  h ∈ R^d_model → P(mood)
  • a NEUTRAL bias + confidence gate so "nothing emotional happening" stays neutral.

Backend-neutral (written against `src.backend.current()`).
"""

from __future__ import annotations

import src.backend as backend
from src.modalities.moods import (  # shared light taxonomy (no backend)
    MOOD_INDEX,
    MOOD_MARGIN,
    MOODS,
    NEUTRAL,
    emotion_to_mood,
    lexicon_mood,
)

B = backend.current()
nn = B.nn
ops = B.ops


# ── Mood head ────────────────────────────────────────────────────────────────


class MoodHead(nn.Module):
    """Multiclass mood classifier over the mean-pooled foundational hidden state.
    Mean pooling (vs. BCF's last-token) reads the whole exchange's tone rather
    than just the final token."""

    def __init__(self, d_model: int, n_moods: int = len(MOODS), hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, n_moods)

    def __call__(self, h):
        """h: [..., d_model] → [..., n_moods] logits."""
        return self.fc2(ops.relu(self.fc1(h)))

    def probs(self, h):
        return ops.softmax(self(h), axis=-1)


def mood_loss(logits, labels):
    """Standard multiclass cross-entropy over mood logits."""
    return ops.cross_entropy(logits, labels, reduction="mean")


def load_mood_head(d_model: int, level=None, stage=None, checkpoint=None):
    """Load the trained mood head that sits beside a stage's checkpoint
    (`<ckpt_dir>/mood_head.npz`), or None if there is none. Shared by EVERY
    inference surface (chat, agent, future serving) so mood behaves identically
    everywhere — it is trained automatically at each cognitive stage's completion
    by train_stage (via `train_mood_head` below)."""
    from pathlib import Path

    candidates = []
    if checkpoint:
        candidates.append(Path(checkpoint).parent / "mood_head.npz")
    elif stage is not None:
        root = (
            Path("dist/checkpoints")
            if level is None
            else Path("dist/checkpoints") / f"level{level}"
        )
        # Mood is only trained at conversational stages (1 + BCF). At any other stage
        # fall back to the NEAREST earlier head (stage, stage-1, …, 1) so chat still
        # has a mood head; the lexicon works regardless.
        candidates += [root / f"stage{s}" / "mood_head.npz" for s in range(stage, 0, -1)]
    else:
        return None
    for path in candidates:
        if not path.exists():
            continue
        head = MoodHead(d_model)
        try:
            B.engine.load_weights(head, str(path))
            return head
        except Exception:
            continue
    return None


def _pooled_states(model, tokenizer, texts, seq_len: int = 128):
    """Mean-pooled foundational hidden state for each text (frozen core only)."""
    if hasattr(model, "set_active_sectors"):
        model.set_active_sectors([])  # read the frozen core, no sectors
    rows = []
    for t in texts:
        try:
            ids = tokenizer.encode(t, add_eos=True)
        except TypeError:
            ids = tokenizer.encode(t)
        ids = (ids or [0])[:seq_len]
        toks = ops.array(ids)[None]
        h = model(toks)  # [1, S, d_model]
        rows.append(ops.mean(h, axis=1))  # [1, d_model] mean over sequence
    return ops.concatenate(rows, axis=0)  # [N, d_model]


def mood_train_step(model, tokenizer, head: MoodHead, batch, optimizer) -> float:
    """One supervised step on the mood head over a batch of (text, mood_label).
    Only the head trains — the foundational features are read frozen (stop_gradient).
    `batch` items: (text, mood_name | mood_index)."""
    texts = [b[0] for b in batch]
    labels = ops.array(
        [float(MOOD_INDEX.get(b[1], b[1]) if isinstance(b[1], str) else b[1]) for b in batch]
    )
    h = ops.stop_gradient(_pooled_states(model, tokenizer, texts))

    def loss_fn(hd):
        return mood_loss(hd(h), labels)

    grad_fn = B.engine.value_and_grad(head, loss_fn)
    loss, grads = grad_fn(head)
    B.engine.optimizer_step(optimizer, head, grads)
    return B.engine.item(loss)


def mood_accuracy(model, tokenizer, head: MoodHead, probes) -> float:
    """Classification accuracy on a (text, mood) probe set."""
    if not probes:
        return 1.0
    texts = [p[0] for p in probes]
    labels = [MOOD_INDEX.get(p[1], p[1]) if isinstance(p[1], str) else p[1] for p in probes]
    h = _pooled_states(model, tokenizer, texts)
    preds = ops.argmax(head(h), axis=-1)
    correct = sum(int(int(B.engine.item(preds[i])) == labels[i]) for i in range(len(labels)))
    return correct / len(labels)


def mood_probs(model, tokenizer, head: MoodHead, text: str, seq_len: int = 128):
    """Raw per-text mood distribution as a plain list[float] over MOODS."""
    h = _pooled_states(model, tokenizer, [text], seq_len=seq_len)
    p = head.probs(h)[0]  # [n_moods]
    return [float(B.engine.item(p[i])) for i in range(len(MOODS))]


# ── Training data + end-to-end training ───────────────────────────────────────
# Used by the normal training pipeline (train_stage._on_stage_complete). The mood
# head is a cheap probe over the frozen core, so it is trained at each cognitive
# stage's completion — no separate manual step (or script) needed.


def _neutral_examples(level, stage, n: int, log=print) -> list:
    """Sample NEUTRAL (text, "neutral") pairs from the conversational corpus. These
    live in STAGE 1 (the conversational stage) — factual/narrative/chit-chat carries
    no active mood — so we always read from there regardless of the current stage
    (a narrow stage like arithmetic has no neutral conversational data of its own,
    which is exactly what made its mood head near-random)."""
    import json
    from pathlib import Path

    base = Path("data") / f"level{level}" / "stage1"
    if not base.exists():
        base = Path("data") / f"level{level}" / f"stage{stage}"
    out: list = []
    for stem in ("instruct", "simple_wikipedia", "tinystories", "basic_chat", "definitions"):
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
                # inference): pull the `User:` line out of transcripts, prose as-is.
                if "User:" in t:
                    seg = t.split("User:", 1)[1].split("\nAssistant:", 1)[0].strip()
                else:
                    seg = t.strip()
                if seg:
                    out.append((seg[:300], "neutral"))
                if len(out) >= n:
                    return out
    return out


def _emotional_examples(per_mood: int, log=print) -> list:
    """Stream EmpatheticDialogues, map each dialogue's emotion onto a mood, and take
    the speaker's RAW first utterance (matching what the chat classifies). Best-effort:
    returns [] if the dataset can't be loaded (offline) instead of raising."""
    from collections import Counter

    try:
        from datasets import load_dataset
    except Exception as e:  # datasets not installed
        log(f"  [mood] datasets unavailable: {e}")
        return []
    out, counts = [], Counter()
    targets = ("happy", "sad", "angry", "afraid", "surprised", "caring")
    try:
        ds = load_dataset("Estwld/empathetic_dialogues_llm", split="train", streaming=True)
    except Exception as e:
        log(f"  [mood] EmpatheticDialogues unavailable: {e}")
        return out
    for ex in ds:
        emo = (ex.get("emotion") or "").strip().lower()
        mood = emotion_to_mood(emo)
        if mood == "neutral" or counts[mood] >= per_mood:
            continue
        turns = [(c.get("role"), c.get("content")) for c in (ex.get("conversations") or [])]
        utt = next(
            (c for r, c in turns if c and (r or "").lower() in ("user", "human", "speaker", "0")),
            None,
        )
        if not utt and turns:
            utt = turns[0][1]
        if utt and utt.strip():
            counts[mood] += 1
            out.append((utt.strip()[:300], mood))
        if all(counts[m] >= per_mood for m in targets):
            break
    return out


def build_mood_examples(
    level, stage, per_mood: int = 300, neutral: int = 1500, seed: int = 0, log=print
) -> list:
    """Assemble the labeled (text, mood) set: emotional turns (EmpatheticDialogues)
    + neutral turns (local factual/narrative files), shuffled. May be empty/small
    when offline — callers decide whether that is enough to train."""
    import random

    data = _emotional_examples(per_mood, log=log) + _neutral_examples(
        level, stage, neutral, log=log
    )
    random.Random(seed).shuffle(data)
    return data


def train_mood_head(
    model,
    tokenizer,
    ckpt_dir,
    *,
    level,
    stage,
    per_mood: int = 300,
    neutral: int = 1500,
    epochs: int = 8,
    batch: int = 32,
    lr: float = 2e-3,
    seed: int = 0,
    precision: str = "fp32",
    min_examples: int = 50,
    log=print,
):
    """Train the MoodHead on the frozen core and save it to `<ckpt_dir>/mood_head.npz`.

    Cheap (head-only over precomputed frozen features). Returns a metrics dict on
    success, or None when there is too little labeled data (e.g. EmpatheticDialogues
    is offline) — in which case nothing is written and the caller carries on. This is
    the single implementation behind both the training pipeline and the CLI script."""
    from collections import Counter
    from pathlib import Path

    import numpy as np

    data = build_mood_examples(level, stage, per_mood=per_mood, neutral=neutral, seed=seed, log=log)
    if len(data) < min_examples:
        log(
            f"  [mood] only {len(data)} labeled examples (< {min_examples}) — skipping "
            "(need HF EmpatheticDialogues; offline?)"
        )
        return None
    log(f"  [mood] {len(data)} examples · per-mood: {dict(Counter(m for _, m in data))}")

    # Precompute frozen pooled features ONCE (the LM does not change), then train the
    # small head directly on them — fast, no model forward pass in the training loop.
    feats = np.array(_pooled_states(model, tokenizer, [t for t, _ in data]))  # [N, d]
    labels = np.array([MOOD_INDEX[m] for _, m in data], dtype=np.float32)
    n_val = max(8, len(data) // 10)
    Xv, yv = feats[:n_val], labels[:n_val]
    Xt, yt = feats[n_val:], labels[n_val:]

    head = MoodHead(model.cfg.d_model if hasattr(model, "cfg") else feats.shape[1])
    B.engine.set_precision(head, precision)
    opt = B.engine.make_optimizer(head, lr, 0.0)

    def _acc(X, y) -> float:
        preds = np.array(ops.argmax(head(ops.array(X)), axis=-1))
        return float((preds == y).mean()) if len(y) else 1.0

    idx = np.arange(len(Xt))
    last = {}
    for ep in range(epochs):
        np.random.shuffle(idx)
        losses = []
        for i in range(0, len(idx), batch):
            j = idx[i : i + batch]
            Hb, yb = ops.array(Xt[j]), ops.array(yt[j])

            def loss_fn(hd, H=Hb, Y=yb):
                return mood_loss(hd(H), Y)

            loss, grads = B.engine.value_and_grad(head, loss_fn)(head)
            B.engine.optimizer_step(opt, head, grads)
            losses.append(float(B.engine.item(loss)))
        last = {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "train_acc": _acc(Xt, yt),
            "val_acc": _acc(Xv, yv),
        }
        log(
            f"  [mood] epoch {ep + 1}/{epochs}: loss={last['loss']:.3f} "
            f"train_acc={last['train_acc']:.2f} val_acc={last['val_acc']:.2f}"
        )

    out = Path(ckpt_dir) / "mood_head.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    B.engine.save_weights(head, str(out))
    log(f"  [mood] saved → {out}")
    return {"examples": len(data), "path": str(out), **last}


def _pick_mood(state) -> tuple[str, float]:
    """NEUTRAL unless another mood beats it by MOOD_MARGIN (keeps the default calm)."""
    top = max(range(len(MOODS)), key=lambda i: state[i])
    if top == NEUTRAL or (state[top] - state[NEUTRAL]) < MOOD_MARGIN:
        return "neutral", state[NEUTRAL]
    return MOODS[top], state[top]


def _lexicon_distribution(text: str) -> list:
    """A mood distribution from the reliable lexicon detector. Concentrates mass on
    the detected mood (scaled by confidence); neutral otherwise. This is the floor
    signal that behaves correctly at any model size (the learned head is near-chance
    on the 11M core), used by both the stateless and the running classifiers."""
    mood, conf = lexicon_mood(text)
    dist = [0.0] * len(MOODS)
    if mood == "neutral":
        dist[NEUTRAL] = 1.0
    else:
        dist[MOOD_INDEX[mood]] = conf
        dist[NEUTRAL] = 1.0 - conf
    return dist


# Only let the LEARNED head override a lexicon-neutral reading when it is THIS
# confident — the 11M head is weak, so it must clear a high bar to speak up.
_HEAD_OVERRIDE_MIN = 0.55


def classify_mood(
    model, tokenizer, head: MoodHead, text: str, seq_len: int = 128
) -> tuple[str, float]:
    """Stateless single-text mood (neutral default). The LEXICON is the primary,
    reliable signal; the learned head only refines when the lexicon finds nothing
    AND the head is highly confident. For a running, conversation-aware mood use
    MoodTracker — emotions build over the whole exchange, not one line."""
    if not text.strip():
        return "neutral", 1.0
    mood, conf = lexicon_mood(text)
    if mood != "neutral":
        return mood, conf
    if head is None:
        return "neutral", 1.0
    hmood, hconf = _pick_mood(mood_probs(model, tokenizer, head, text, seq_len=seq_len))
    return (
        (hmood, hconf) if (hmood != "neutral" and hconf >= _HEAD_OVERRIDE_MIN) else ("neutral", 1.0)
    )


class MoodTracker:
    """Conversation-aware mood with memory. Emotions are carried by the WHOLE
    exchange, not a single message: each turn the current message's distribution is
    blended into a running state by exponential smoothing (`alpha` = how much the
    latest message moves it). So a happy run stays happy through a neutral "ok", and
    the state decays back toward neutral when nothing sustains a mood — the default.

      state ← alpha · P(current message) + (1 − alpha) · state
    """

    def __init__(
        self,
        head: MoodHead | None,
        alpha: float = 0.4,
        margin: float = MOOD_MARGIN,
        context_chars: int = 240,
    ):
        self.head = head
        self.alpha = alpha
        self.margin = margin
        self.context_chars = context_chars
        self.reset()

    def reset(self) -> None:
        self.state = [0.0] * len(MOODS)
        self.state[NEUTRAL] = 1.0

    def update(self, model, tokenizer, message: str, context: str = "") -> str:
        """Fold one new user message into the running mood and return the current
        mood. The LEXICON is the signal that moves the state (reliable at any model
        size); the learned head only nudges when the lexicon is neutral AND the head
        is confident. Works with head=None (lexicon-only)."""
        if not message.strip():
            return self.current()
        lex_mood, _ = lexicon_mood(message)
        if lex_mood != "neutral" or self.head is None:
            p = _lexicon_distribution(message)
        else:
            # lexicon found nothing emotional → consult the head, but only let a
            # confident non-neutral reading move the state; otherwise decay to neutral.
            text = (context[-self.context_chars :] + "\n" + message) if context else message
            hp = mood_probs(model, tokenizer, self.head, text)
            hmood, hconf = _pick_mood(hp)
            p = (
                hp
                if (hmood != "neutral" and hconf >= _HEAD_OVERRIDE_MIN)
                else _lexicon_distribution("")
            )  # neutral one-hot
        self.state = [
            self.alpha * p[i] + (1 - self.alpha) * self.state[i] for i in range(len(MOODS))
        ]
        return self.current()

    def current(self) -> str:
        return _pick_mood(self.state)[0]

    def distribution(self) -> dict:
        """Running mood state as a {mood: prob} dict (for context/mood stats)."""
        return {MOODS[i]: round(self.state[i], 3) for i in range(len(MOODS))}
