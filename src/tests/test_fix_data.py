"""Regression tests — data: experience log, relevance feedback, dialogue mixing,
the confidence validator, the streaming loader (interleave / weights / held-out val),
and completion-only loss masking. (Split from the old test_fixes.py.)"""

import json
import tempfile
from pathlib import Path

import numpy as np
from fixes_common import FakeTok, write_corpus

# ─────────────────────────── experience log + relevance ──────────────────────


def test_experience_log_only_saves_signal_bearing_turns():
    """A turn with no feedback is NOT saved (no benefit); a corrected turn learns the
    CORRECTION, not the model's wrong answer."""
    from src.memory.experience_log import detect_correction, load_experiences, log_experience

    with tempfile.TemporaryDirectory() as td:
        p = str(Path(td) / "e.jsonl")
        assert log_experience("hi", "hello", feedback="neutral", path=p) is False
        assert log_experience("2+2?", "4", feedback="accepted", path=p) is True
        assert (
            log_experience(
                "cap of France?", "London", feedback="corrected", correction="Paris", path=p
            )
            is True
        )
        recs = load_experiences(p)
        assert len(recs) == 2  # neutral was dropped
        corr = next(r for r in recs if r["feedback"] == "corrected")
        assert "Paris" in corr["text"] and "London" not in corr["text"]
    # implicit correction detection (EN + ES), no false positives on a new topic
    assert detect_correction("no, it is Paris") and detect_correction("eso está mal")
    assert not detect_correction("what about Spain?")


def test_relevance_feedback_overrides_utility():
    """A `corrected` experience must score higher R⁺ than the same content unlabeled —
    feedback is the ground-truth Utility (error-driven learning gets the boost)."""
    from src.memory.episodic_buffer import Experience
    from src.relevance.engine import RelevanceEngine

    re = RelevanceEngine(ltss=None)
    re.update_state(np.zeros(64, dtype=np.float32))
    emb = np.random.randn(64).astype(np.float32)
    neutral = Experience(text="x", embedding=emb, feedback="neutral")
    neutral.episodic_context = []
    corrected = Experience(text="x", embedding=emb, feedback="corrected")
    corrected.episodic_context = []
    assert re.score(corrected) > re.score(neutral)


# ─────────────────────────── confidence validator ────────────────────────────


def test_confidence_validator_routes_by_knowledge():
    """The confidence-gated validator: human-labelled or high-coherence → self-approve;
    mid → defer (no external source); very low → escalate to human."""
    from src.consolidation.validation import (
        ExperienceValidator,
        HumanReviewSource,
    )

    class _Exp:
        def __init__(self, feedback="neutral"):
            import uuid as _u

            self.feedback = feedback
            self.uid = str(_u.uuid4())
            self.text = "x"

    class _FakeQueue:
        def __init__(self):
            self.queued = []

        def queue_for_review(self, exp, score, rationale=""):
            self.queued.append(exp.uid)

    q = _FakeQueue()
    v = ExperienceValidator(human_source=HumanReviewSource(q))  # no external sources

    # human-corrected → authoritative → consolidate (coherence irrelevant)
    assert v.decide(_Exp("corrected"), coherence=0.0).fate == "consolidate"
    # neutral + high coherence (consistent with prior knowledge) → consolidate
    assert v.decide(_Exp("neutral"), coherence=0.9).fate == "consolidate"
    # neutral + mid coherence, nothing external available → defer (retry)
    assert v.decide(_Exp("neutral"), coherence=0.5).fate == "defer"
    # neutral + very low coherence → escalate to human queue
    d = v.decide(_Exp("neutral"), coherence=0.1)
    assert d.fate == "queue" and d.source == "human" and len(q.queued) == 1


def test_validator_external_stubs_are_inert_until_configured():
    """Unconfigured peer-model / web-research sources are skipped, not crash."""
    from src.consolidation.validation import (
        PeerModelSource,
        WebResearchSource,
        default_validator,
    )

    assert PeerModelSource().available() is False
    assert WebResearchSource().available() is False
    v = default_validator(ambiguity_handler=None)  # no human either

    class _Exp:
        feedback = "neutral"
        uid = "u"
        text = "x"

    # mid confidence, all external inert, no human → defer (no crash)
    assert v.decide(_Exp(), coherence=0.5).fate == "defer"


# ─────────────────────────── data loader: interleave + weights ───────────────


def test_loader_interleaves_sources_no_block():
    """Records from both files must be mixed throughout — not one whole file then
    the other (which caused catastrophic forgetting)."""
    from src.data.loader import TextDataset

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        write_corpus(d)
        ds = TextDataset(
            str(d), FakeTok(), seq_len=16, batch_size=2, shuffle=True, shuffle_buffer=50
        )
        seen = []
        for i, rec in enumerate(ds._iter_records()):
            seen.append("DIALOG" if rec["text"].startswith("DIALOG") else "STORY")
            if i >= 600:
                break
        # both sources appear within the first window (no pure-story prefix block)
        assert "DIALOG" in seen[:200] and "STORY" in seen[:200]


def test_loader_source_weights_oversample():
    """A small file with a high source weight should contribute a much larger
    share than its size alone would give."""
    from src.data.loader import TextDataset

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        write_corpus(d)

        def dialog_share(weights):
            ds = TextDataset(
                str(d),
                FakeTok(),
                seq_len=16,
                batch_size=2,
                shuffle=True,
                shuffle_buffer=50,
                source_weights=weights,
            )
            n_d = n_t = 0
            for i, rec in enumerate(ds._iter_records()):
                if rec["text"].startswith("DIALOG"):
                    n_d += 1
                else:
                    n_t += 1
                if i >= 1500:
                    break
            return n_d / (n_d + n_t)

        base = dialog_share(None)
        boosted = dialog_share({"dialogue": 5.0})
        assert boosted > base + 0.1  # oversampling clearly lifts the share


# ─────────────────────────── held-out val split (H7) ─────────────────────────


def test_loader_excludes_val_files_from_training():
    from src.data.loader import TextDataset

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        with open(d / "story.jsonl", "w") as f:
            for _ in range(50):
                f.write(json.dumps({"text": "TRAIN sample text here"}) + "\n")
        with open(d / "story.val.jsonl", "w") as f:
            for _ in range(20):
                f.write(json.dumps({"text": "VAL sample text here"}) + "\n")
        train = TextDataset(str(d), FakeTok(), seq_len=8, batch_size=1, shuffle=False)
        val = TextDataset(str(d), FakeTok(), seq_len=8, batch_size=1, shuffle=False, val=True)
        train_txt = {r["text"].split()[0] for r in train._iter_records()}
        val_txt = {r["text"].split()[0] for r in val._iter_records()}
        assert train_txt == {"TRAIN"}  # training never sees the held-out file
        assert val_txt == {"VAL"}  # val loads only the held-out file


def test_write_jsonl_routes_holdout():
    from src.data.jsonl_writer import write_jsonl

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "src.jsonl"
        recs = ({"text": f"document number {i} with enough words to count"} for i in range(500))
        write_jsonl(recs, out, token_budget_m=999, verbose=False, val_fraction=0.1)
        val = out.with_suffix(".val.jsonl")
        assert val.exists()
        with open(out) as fh:
            n_train = sum(1 for _ in fh)
        with open(val) as fh:
            n_val = sum(1 for _ in fh)
        assert n_val > 0 and n_train > 0
        assert 0.05 < n_val / (n_train + n_val) < 0.15  # ~10% held out


# ───────────────── completion-only loss masking (conversational SFT) ─────────


def test_split_turns_detects_conversation_vs_prose():
    from src.data.loader import _split_turns

    convo = "System: be nice.\nUser: hi there\nAssistant: hello!"
    roles = [r for r, _ in _split_turns(convo)]
    assert roles == ["System", "User", "Assistant"]
    # prose never opens with a role marker → None (train on every token)
    assert _split_turns("Once upon a time there was a cat.") is None


def test_completion_mask_trains_only_response_turns():
    """A transcript trains the Assistant turn (+ a turn-final EOS) and masks the
    System/User context, so the loss teaches answering, not transcript modelling."""
    from src.data.loader import TextDataset
    from src.modalities.text import EOS_ID

    ds = TextDataset.__new__(TextDataset)
    ds.tokenizer = FakeTok()
    ids, mask = ds._encode_record({"text": "System: s\nUser: q\nAssistant: a", "lang": "en"})
    assert len(ids) == len(mask) and sum(mask) > 0
    assert sum(mask) < len(mask)  # context is masked out
    assert ids[-1] == EOS_ID and mask[-1] == 1  # learns to stop after the answer
    # every trained position lies inside the assistant turn (after "Assistant")
    raw = FakeTok().encode_raw("\nAssistant: a")
    assert sum(mask) == len(raw) + 1  # assistant turn + its EOS


def test_completion_mask_drops_record_with_no_response():
    from src.data.loader import TextDataset

    ds = TextDataset.__new__(TextDataset)
    ds.tokenizer = FakeTok()
    ids, mask = ds._encode_record({"text": "User: q\nUser: q2", "lang": "en"})
    assert ids == [] and mask == []  # nothing to train on → skipped


def test_loader_with_mask_yields_token_mask_pairs():
    from src.data.loader import TextDataset

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        with open(d / "chat.jsonl", "w") as f:
            for _ in range(200):
                f.write(json.dumps({"text": "User: hi\nAssistant: hello there friend"}) + "\n")
        ds = TextDataset(str(d), FakeTok(), seq_len=16, batch_size=2, shuffle=False, with_mask=True)
        toks, mask = next(iter(ds.batches()))
        assert toks.shape == (2, 17) and mask.shape == (2, 17)
        assert set(np.unique(mask)).issubset({0, 1})
        assert 0 < int(mask.sum()) < mask.size  # some trained, some masked


def test_completion_mask_handles_system_prefixed_record():
    """A System-prefixed transcript still masks System+User and trains only the
    Assistant turn — the enriched data stays compatible with completion masking."""
    from src.data.loader import TextDataset

    ds = TextDataset.__new__(TextDataset)
    ds.tokenizer = FakeTok()
    _ids, mask = ds._encode_record(
        {"text": "System: be kind (mood: happy)\nUser: hi\nAssistant: hello", "lang": "en"}
    )
    assert 0 < sum(mask) < len(mask)
    trained = FakeTok().encode_raw("\nAssistant: hello")
    assert sum(mask) == len(trained) + 1  # assistant turn + EOS only
