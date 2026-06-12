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
from typing import Optional, Tuple

import src.backend as backend
from src.modalities.moods import (              # shared light taxonomy (no backend)
    MOODS, MOOD_INDEX, NEUTRAL, MOOD_MARGIN,
    emotion_to_mood, mood_system_phrase,        # noqa: F401  (re-exported for callers)
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
    everywhere — train it with scripts/train_mood_head.py."""
    from pathlib import Path
    if checkpoint:
        path = Path(checkpoint).parent / "mood_head.npz"
    elif stage is not None:
        root = Path("dist/checkpoints") if level is None else Path("dist/checkpoints") / f"level{level}"
        path = root / f"stage{stage}" / "mood_head.npz"
    else:
        return None
    if not path.exists():
        return None
    head = MoodHead(d_model)
    try:
        B.engine.load_weights(head, str(path))
        return head
    except Exception:
        return None


def _pooled_states(model, tokenizer, texts, seq_len: int = 128):
    """Mean-pooled foundational hidden state for each text (frozen core only)."""
    if hasattr(model, "set_active_sectors"):
        model.set_active_sectors([])              # read the frozen core, no sectors
    rows = []
    for t in texts:
        try:
            ids = tokenizer.encode(t, add_eos=True)
        except TypeError:
            ids = tokenizer.encode(t)
        ids = (ids or [0])[:seq_len]
        toks = ops.array(ids)[None]
        h = model(toks)                            # [1, S, d_model]
        rows.append(ops.mean(h, axis=1))           # [1, d_model] mean over sequence
    return ops.concatenate(rows, axis=0)           # [N, d_model]


def mood_train_step(model, tokenizer, head: MoodHead, batch, optimizer) -> float:
    """One supervised step on the mood head over a batch of (text, mood_label).
    Only the head trains — the foundational features are read frozen (stop_gradient).
    `batch` items: (text, mood_name | mood_index)."""
    texts  = [b[0] for b in batch]
    labels = ops.array([float(MOOD_INDEX.get(b[1], b[1]) if isinstance(b[1], str) else b[1])
                        for b in batch])
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
    texts  = [p[0] for p in probes]
    labels = [MOOD_INDEX.get(p[1], p[1]) if isinstance(p[1], str) else p[1] for p in probes]
    h      = _pooled_states(model, tokenizer, texts)
    preds  = ops.argmax(head(h), axis=-1)
    correct = sum(int(int(B.engine.item(preds[i])) == labels[i]) for i in range(len(labels)))
    return correct / len(labels)


def mood_probs(model, tokenizer, head: MoodHead, text: str, seq_len: int = 128):
    """Raw per-text mood distribution as a plain list[float] over MOODS."""
    h = _pooled_states(model, tokenizer, [text], seq_len=seq_len)
    p = head.probs(h)[0]                              # [n_moods]
    return [float(B.engine.item(p[i])) for i in range(len(MOODS))]


def _pick_mood(state) -> Tuple[str, float]:
    """NEUTRAL unless another mood beats it by MOOD_MARGIN (keeps the default calm)."""
    top = max(range(len(MOODS)), key=lambda i: state[i])
    if top == NEUTRAL or (state[top] - state[NEUTRAL]) < MOOD_MARGIN:
        return "neutral", state[NEUTRAL]
    return MOODS[top], state[top]


def classify_mood(model, tokenizer, head: MoodHead, text: str,
                  seq_len: int = 128) -> Tuple[str, float]:
    """Stateless single-text mood (neutral default). For a running, conversation-
    aware mood use MoodTracker — emotions build over the whole exchange, not one line."""
    if head is None or not text.strip():
        return "neutral", 1.0
    return _pick_mood(mood_probs(model, tokenizer, head, text, seq_len=seq_len))


class MoodTracker:
    """Conversation-aware mood with memory. Emotions are carried by the WHOLE
    exchange, not a single message: each turn the current message's distribution is
    blended into a running state by exponential smoothing (`alpha` = how much the
    latest message moves it). So a happy run stays happy through a neutral "ok", and
    the state decays back toward neutral when nothing sustains a mood — the default.

      state ← alpha · P(current message) + (1 − alpha) · state
    """

    def __init__(self, head: Optional[MoodHead], alpha: float = 0.4,
                 margin: float = MOOD_MARGIN, context_chars: int = 240):
        self.head = head
        self.alpha = alpha
        self.margin = margin
        self.context_chars = context_chars
        self.reset()

    def reset(self) -> None:
        self.state = [0.0] * len(MOODS)
        self.state[NEUTRAL] = 1.0

    def update(self, model, tokenizer, message: str, context: str = "") -> str:
        """Fold one new user message (optionally with a little recent context, so a
        short/ambiguous line is read in light of the conversation) into the running
        mood, and return the current mood name."""
        if self.head is None or not message.strip():
            return self.current()
        text = (context[-self.context_chars:] + "\n" + message) if context else message
        p = mood_probs(model, tokenizer, self.head, text)
        self.state = [self.alpha * p[i] + (1 - self.alpha) * self.state[i]
                      for i in range(len(MOODS))]
        return self.current()

    def current(self) -> str:
        return _pick_mood(self.state)[0]

    def distribution(self) -> dict:
        """Running mood state as a {mood: prob} dict (for context/mood stats)."""
        return {MOODS[i]: round(self.state[i], 3) for i in range(len(MOODS))}
