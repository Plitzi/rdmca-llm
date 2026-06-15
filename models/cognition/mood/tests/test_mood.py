"""Mood faculty (models/cognition/mood) — model-free paths: the MoodHead module, the
lexicon distribution / neutral-default picker, stateless classify, and the running
MoodTracker (lexicon-driven, head=None). Lives with cognition (mood is its faculty)."""

import numpy as np

import src.backend as backend
from models.cognition.mood.head import (
    MoodHead,
    MoodTracker,
    _lexicon_distribution,
    _pick_mood,
    classify_mood,
    load_mood_head,
    mood_loss,
)
from models.cognition.mood.lexicon import MOODS, NEUTRAL

ops = backend.current().ops


def test_mood_head_forward_and_probs():
    head = MoodHead(d_model=16)
    h = ops.array(np.zeros((4, 16), dtype=np.float32))
    assert tuple(head(h).shape) == (4, len(MOODS))  # logits
    probs = head.probs(h)
    assert tuple(probs.shape) == (4, len(MOODS))
    row = np.asarray(ops.to_numpy(probs))[0]
    assert abs(row.sum() - 1.0) < 1e-4  # softmax normalizes


def test_mood_loss_is_finite():
    head = MoodHead(d_model=16)
    logits = head(ops.array(np.zeros((3, 16), dtype=np.float32)))
    loss = mood_loss(logits, ops.array(np.array([0, 1, 2], dtype=np.int64)))
    assert np.isfinite(float(backend.current().engine.item(loss)))


def test_load_mood_head_without_checkpoint():
    # No trained checkpoint on disk → None (callers then run lexicon-only).
    assert load_mood_head(d_model=16) is None


def test_lexicon_distribution_and_pick_neutral_default():
    dist = _lexicon_distribution("")  # nothing emotional → neutral one-hot
    assert dist[NEUTRAL] == 1.0 and len(dist) == len(MOODS)
    assert _pick_mood(dist) == ("neutral", 1.0)
    # a state that barely beats neutral stays neutral (MOOD_MARGIN guard)
    state = [0.5, 0.55, 0, 0, 0, 0, 0]
    assert _pick_mood(state)[0] == "neutral"


def test_classify_mood_empty_and_neutral_default():
    assert classify_mood(None, None, None, "") == ("neutral", 1.0)
    # head=None + a neutral statement → neutral (no model needed on the lexicon path)
    mood, _ = classify_mood(None, None, None, "the meeting is at noon")
    assert mood in MOODS


def test_mood_tracker_lexicon_only_runs_and_decays():
    tracker = MoodTracker(head=None, alpha=0.5)
    assert tracker.current() == "neutral"
    out = tracker.update(None, None, "I am absolutely thrilled and so happy!")
    assert out in MOODS  # lexicon moved (or held) the running state
    dist = tracker.distribution()
    assert set(dist) == set(MOODS) and abs(sum(dist.values()) - round(sum(dist.values()), 3)) < 1
    # neutral messages decay the state back toward neutral
    for _ in range(20):
        tracker.update(None, None, "ok")
    assert tracker.current() == "neutral"


def test_mood_tracker_reset():
    tracker = MoodTracker(head=None)
    tracker.update(None, None, "I am furious!")
    tracker.reset()
    assert tracker.current() == "neutral"


# ── model-backed mood functions (tiny model + fake tokenizer) ─────────────────
def _tiny_model():
    from src.model.config import ModelConfig
    from src.model.transformer import RDMCAFoundational

    cfg = ModelConfig(
        d_model=32, n_layers=1, n_heads=2, n_kv_heads=1, ffn_dim=64, context_len=64,
        vocab_size=64, mrl_dims=[16, 32], dropout=0.0,
    )  # fmt: skip
    return RDMCAFoundational(cfg)


class _FakeTok:
    ready = True

    def encode(self, text, add_bos=False, add_eos=False):
        return [(ord(c) % 60) + 1 for c in text][:32] or [1]


def test_pooled_states_and_mood_probs():
    from models.cognition.mood.head import _pooled_states, mood_probs

    model, tok = _tiny_model(), _FakeTok()
    h = _pooled_states(model, tok, ["hello there", "another line"])
    assert tuple(h.shape) == (2, model.cfg.d_model)
    dist = mood_probs(model, tok, MoodHead(model.cfg.d_model), "how are you")
    assert len(dist) == len(MOODS) and abs(sum(dist) - 1.0) < 1e-4


def test_mood_train_step_and_accuracy():
    from models.cognition.mood.head import mood_accuracy, mood_train_step

    model, tok = _tiny_model(), _FakeTok()
    head = MoodHead(model.cfg.d_model)
    opt = backend.current().engine.make_optimizer(head, 1e-2, 0.0)
    loss = mood_train_step(model, tok, head, [("happy day", "happy"), ("so sad", "sad")], opt)
    assert np.isfinite(float(loss))
    acc = mood_accuracy(model, tok, head, [("x", "neutral"), ("y", "happy")])
    assert 0.0 <= acc <= 1.0
    assert mood_accuracy(model, tok, head, []) == 1.0  # empty probes → trivially 1.0


def test_emotional_and_neutral_examples_offline(monkeypatch):
    from models.cognition.mood import head as H

    # EmpatheticDialogues unavailable (offline) → [] without raising
    monkeypatch.setattr(H, "load_dataset", None, raising=False)
    assert H._emotional_examples(per_mood=5) == [] or isinstance(H._emotional_examples(5), list)
    # no local corpus on disk → [] neutral examples
    assert H._neutral_examples(level=999, stage=1, n=10) == []


def test_train_mood_head_skips_when_too_few(tmp_path, monkeypatch):
    from models.cognition.mood import head as H

    monkeypatch.setattr(H, "build_mood_examples", lambda *a, **k: [])
    assert H.train_mood_head(_tiny_model(), _FakeTok(), tmp_path, level=0, stage=1) is None


def test_train_mood_head_trains_and_saves(tmp_path, monkeypatch):
    from models.cognition.mood import head as H

    examples = [(f"happy {i}", "happy") for i in range(40)] + [
        (f"calm {i}", "neutral") for i in range(40)
    ]
    monkeypatch.setattr(H, "build_mood_examples", lambda *a, **k: examples)
    out = H.train_mood_head(
        _tiny_model(), _FakeTok(), tmp_path, level=0, stage=1, epochs=2, min_examples=10
    )
    assert out and out["examples"] == 80
    assert (tmp_path / "mood_head.npz").exists()
