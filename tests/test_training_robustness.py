"""
Aggressive robustness tests for the TRAINING pipeline — the foundation everything
else builds on. Covers the failure modes behind the Level-1 stages 3–7 collapse:

  - stage transition / checkpoint-load chain (non-contiguous stages, BCF freeze);
  - catastrophic forgetting: the loader must INTERLEAVE sources, never read one
    corpus fully then the next (the measured PPL-jump-at-file-boundary bug);
  - rehearsal weighting biased toward the largest (conversation) corpus;
  - completion-only loss masking (train assistant + EOS, mask user/system);
  - resume must NOT retrain duplicate data (seeded skip reproduces the stream);
  - val/train disjointness;
  - tiny/empty corpus guards (no hang, sub-batch corpus still yields);
  - cosine LR schedule (warmup ramp, cosine decay, min floor, no div-by-zero).

Self-contained: a fake tokenizer + temp dirs, no trained tokenizer/checkpoints.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

import train_stage as T
from src.data.loader import TextDataset, DataLoader
from src.modalities.text import BOS_ID, EOS_ID


# ─────────────────────────── fakes / helpers ────────────────────────────────
class FakeTok:
    """Deterministic char→id tokenizer (no SentencePiece needed)."""
    ready = True
    lang_tokens = {"en": 5, "es": 6}

    def encode(self, text, lang="en", add_bos=False, add_eos=False):
        ids = [ord(c) % 900 + 100 for c in text]
        if add_bos:
            ids = [BOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]
        return ids

    def encode_raw(self, text):
        return [ord(c) % 900 + 100 for c in text]

    def decode(self, ids):
        return "".join(chr((i - 100) % 900) for i in ids if i not in (BOS_ID, EOS_ID))


def _write_jsonl(path: Path, texts, lang="en"):
    path.write_text("\n".join(json.dumps({"text": t, "lang": lang}) for t in texts) + "\n")


# ════════════════════════ 1. stage transition chain ═════════════════════════
def _cfg(stages):
    return {"curriculum": {f"stage{s}": {"name": f"s{s}"} for s in stages}}


def test_prev_active_stage_contiguous():
    cfg = _cfg([1, 2, 3, 4, 5, 6, 7])
    assert [T.prev_active_stage(s, cfg) for s in range(2, 8)] == [1, 2, 3, 4, 5, 6]
    assert T.prev_active_stage(1, cfg) is None


def test_prev_active_stage_non_contiguous():
    """Stages can skip numbers (e.g. {1,2,3,6}); the predecessor is the real
    highest-below, not stage-minus-one."""
    cfg = _cfg([1, 2, 3, 6])
    assert T.prev_active_stage(6, cfg) == 3
    assert T.prev_active_stage(3, cfg) == 2


def test_behavioral_freeze_boundary():
    """Cognitive stages (≤ BCF_STAGE) train the core; behavioral (> BCF) are LoRA.
    The freeze happens right after the last cognitive stage."""
    assert T.BCF_STAGE == 7
    assert not T.is_behavioral_stage(T.BCF_STAGE)
    assert T.is_behavioral_stage(T.BCF_STAGE + 1)
    # Core freezes after the highest cognitive stage, ignoring behavioral 8/9.
    assert T.last_cognitive_stage(_cfg([1, 2, 3, 6, 7, 8, 9])) == 7
    assert T.last_cognitive_stage(_cfg([1, 2, 3])) == 3


# ════════════════════════ 2. catastrophic forgetting ════════════════════════
def test_loader_interleaves_sources_no_single_source_block(tmp_path):
    """The loader must DRAW from all sources at once, not drain one then the next.
    With two equal-sized sources, the first records must contain BOTH — otherwise
    the model trains a whole distribution then forgets it (the file-boundary bug)."""
    _write_jsonl(tmp_path / "aaa.jsonl", [f"A{i}" for i in range(400)])
    _write_jsonl(tmp_path / "bbb.jsonl", [f"B{i}" for i in range(400)])
    ds = TextDataset(str(tmp_path), FakeTok(), seq_len=8, batch_size=2,
                     shuffle=True, shuffle_buffer=16)
    first = [r["text"][0] for r in (rec for rec, _ in zip(ds._iter_records(), range(80)))]
    assert "A" in first[:40] and "B" in first[:40]          # both appear early
    # neither source forms a long contiguous run at the start
    assert first[:40].count("A") < 38 and first[:40].count("B") < 38


def test_rehearsal_weight_favours_largest_corpus(tmp_path):
    """Replay selection weight is proportional to corpus size, so conversation
    (the largest earlier stage) dominates rehearsal instead of being tied with a
    tiny arithmetic corpus (the uniform-selection forgetting bug)."""
    big, small = tmp_path / "big", tmp_path / "small"
    big.mkdir(); small.mkdir()
    _write_jsonl(big / "d.jsonl", [f"big {i}" for i in range(2000)])
    _write_jsonl(small / "d.jsonl", ["tiny"])
    dbig = TextDataset(str(big), FakeTok(), seq_len=8, batch_size=2)
    dsm  = TextDataset(str(small), FakeTok(), seq_len=8, batch_size=2)
    ld = DataLoader(TextDataset(str(big), FakeTok(), seq_len=8, batch_size=2),
                    replay=[dbig, dsm], replay_fraction=0.5)
    w = ld._replay_weights
    assert w[0] / sum(w) > 0.95                              # big corpus ~all the weight


def test_rehearsal_fraction_controls_replay_mixing(tmp_path):
    """End-to-end: replay_fraction=0 draws ONLY the primary; =1 draws ONLY replay.
    Distinguishable vocab ('1' primary vs '9' replay) proves rehearsal is wired, so
    a later stage actually keeps refreshing earlier skills."""
    prim, rep = tmp_path / "prim", tmp_path / "rep"
    prim.mkdir(); rep.mkdir()
    _write_jsonl(prim / "d.jsonl", ["111111111"] * 200)
    _write_jsonl(rep / "d.jsonl", ["999999999"] * 200)
    rds = TextDataset(str(rep), FakeTok(), seq_len=8, batch_size=2)

    def chars(frac):
        ld = DataLoader(TextDataset(str(prim), FakeTok(), seq_len=8, batch_size=2),
                        replay=[rds], replay_fraction=frac, seed=3)
        seen = set()
        for _ in range(20):
            seen |= set(FakeTok().decode(ld.next_batch().ravel().tolist()))
        return seen

    assert chars(0.0) == {"1"}                               # no rehearsal → primary only
    assert "9" in chars(1.0)                                 # full rehearsal → replay drawn


# ════════════════════════ 3. completion-only masking ════════════════════════
def test_completion_masking_trains_only_assistant(tmp_path):
    """User/System context is masked (loss 0); the assistant turn + a trailing EOS
    are trained (loss 1) — so the model learns to ANSWER, not echo the prompt."""
    _write_jsonl(tmp_path / "c.jsonl", ["User: hi there\nAssistant: hello"])
    ds = TextDataset(str(tmp_path), FakeTok(), seq_len=64, batch_size=1, with_mask=True)
    rec = {"text": "User: hi there\nAssistant: hello", "lang": "en"}
    ids, mask = ds._encode_record(rec)
    assert len(ids) == len(mask) and any(mask)
    # the LAST trained token is EOS (learn to stop after answering)
    assert mask[-1] == 1 and ids[-1] == EOS_ID
    # the user span (before "Assistant") is fully masked
    tok = FakeTok()
    user_len = 1 + 1 + len(tok.encode_raw("User: hi there"))   # BOS + lang + user block
    assert all(m == 0 for m in mask[:user_len])
    # at least one assistant token is trained
    assert any(m == 1 for m in mask[user_len:])


def test_record_with_no_response_is_dropped(tmp_path):
    """A transcript with only a User turn carries no training signal → empty."""
    _write_jsonl(tmp_path / "d.jsonl", ["placeholder"])     # dir must be non-empty
    ds = TextDataset(str(tmp_path), FakeTok(), seq_len=64, batch_size=1, with_mask=True)
    ids, mask = ds._encode_record({"text": "User: just a question?", "lang": "en"})
    assert ids == [] and mask == []


def test_prose_trains_every_token(tmp_path):
    """Non-conversational prose (no role markers) trains on every token (mask all 1)."""
    _write_jsonl(tmp_path / "d.jsonl", ["placeholder"])
    ds = TextDataset(str(tmp_path), FakeTok(), seq_len=64, batch_size=1, with_mask=True)
    ids, mask = ds._encode_record({"text": "the cat sat on the mat", "lang": "en"})
    assert ids and all(m == 1 for m in mask)


# ════════════════════════ 4. resume = no duplicate data ═════════════════════
def test_seeded_skip_reproduces_stream(tmp_path):
    """--resume fast-forwards via skip(); a seeded loader that skips k batches lands
    on the SAME (k+1)-th batch a fresh loader produces — so resume continues instead
    of re-reading (and overfitting) data already seen."""
    _write_jsonl(tmp_path / "d.jsonl", [f"row number {i} content" for i in range(500)])

    def fresh():
        ds = TextDataset(str(tmp_path), FakeTok(), seq_len=8, batch_size=2,
                         shuffle=True, seed=7)
        return DataLoader(ds, seed=99)

    a = fresh()
    seen = [a.next_batch() for _ in range(6)]
    b = fresh()
    assert b.skip(5) == 5
    np.testing.assert_array_equal(b.next_batch(), seen[5])


# ════════════════════════ 5. val / train disjointness ═══════════════════════
def test_val_loader_reads_only_val_files(tmp_path):
    _write_jsonl(tmp_path / "d.jsonl", [f"train {i}" for i in range(50)])
    _write_jsonl(tmp_path / "d.val.jsonl", [f"heldout {i}" for i in range(50)])
    train = TextDataset(str(tmp_path), FakeTok(), val=False)
    val   = TextDataset(str(tmp_path), FakeTok(), val=True)
    assert all(not f.name.endswith(".val.jsonl") for f in train._files)
    assert val._files and all(f.name.endswith(".val.jsonl") for f in val._files)


# ════════════════════════ 6. tiny / empty corpus guards ═════════════════════
def test_empty_corpus_stops_instead_of_hanging(tmp_path):
    (tmp_path / "e.jsonl").write_text("")
    ds = TextDataset(str(tmp_path), FakeTok(), seq_len=8, batch_size=2)
    with pytest.raises(StopIteration):
        DataLoader(ds).next_batch()


def test_sub_batch_corpus_still_yields(tmp_path):
    """A corpus SMALLER than one batch must still produce batches (by cycling),
    not hang — only a truly empty corpus stops."""
    _write_jsonl(tmp_path / "d.jsonl", ["hi"])              # ~few tokens << one batch
    ds = TextDataset(str(tmp_path), FakeTok(), seq_len=8, batch_size=2)
    batch = DataLoader(ds).next_batch()
    assert batch.shape == (2, 9)


# ════════════════════════ 7. cosine LR schedule ═════════════════════════════
def test_cosine_lr_warmup_then_decay_to_floor():
    base, lo, warm, total = 3e-4, 5e-5, 100, 1000
    assert T.cosine_lr(0, base, lo, warm, total) == 0.0           # ramp starts at 0
    assert T.cosine_lr(warm, base, lo, warm, total) == pytest.approx(base)   # peak at warmup end
    mid = T.cosine_lr(550, base, lo, warm, total)
    assert lo < mid < base                                        # decaying through the middle
    end = T.cosine_lr(total, base, lo, warm, total)
    assert end == pytest.approx(lo, abs=1e-9)                     # floor at the end
    # monotonic non-increasing after warmup
    xs = [T.cosine_lr(s, base, lo, warm, total) for s in range(warm, total + 1, 50)]
    assert all(a >= b - 1e-12 for a, b in zip(xs, xs[1:]))


def test_cosine_lr_no_div_by_zero_when_total_le_warmup():
    # total == warmup would divide by zero without the max(...,1) guard
    assert T.cosine_lr(50, 3e-4, 5e-5, 100, 100) == pytest.approx(3e-4 * 50 / 100)
    assert np.isfinite(T.cosine_lr(150, 3e-4, 5e-5, 100, 100))


# ════════════════════════ 8. per-stage audit record ═════════════════════════
def test_write_stage_audit_captures_full_context(tmp_path):
    """Each stage must persist a COMPLETE, auditable context: hyperparameters, data
    provenance (per-source tokens), rehearsal mix + weights, model geometry, env."""
    ddir = tmp_path / "data"; ddir.mkdir()
    (ddir / "src.meta.json").write_text(json.dumps({"tokens": 1234567, "exhausted": False}))

    class FakeCfgObj:
        d_model = 256; n_heads = 4; n_layers = 6; vocab_size = 8192
        context_len = 512; mrl_dims = [128, 256]

    class FakeModel:
        cfg = FakeCfgObj()
        def count_params(self): return 11_000_000

    class FakeLoader:
        replay_dirs = ["data/level1/stage1", "data/level1/stage2"]
        replay_fraction = 0.15
        _replay_weights = [900.0, 100.0]

    cfg = {"level": 1, "backend": "mlx",
           "curriculum": {"stage5": {"name": "Reasoning", "data_dir": str(ddir)}}}
    tcfg = {"lr": 3e-4, "lr_min": 5e-5, "batch_size": 16, "grad_accumulation": 1,
            "warmup_steps": 500, "max_corpus_passes": 4, "clip_grad_norm": 1.0,
            "save_every": 500, "eval_every": 500}
    rec = T.write_stage_audit(
        tmp_path, stage=5, cfg=cfg, model=FakeModel(), model_cfg=FakeCfgObj(), tcfg=tcfg,
        data_loader=FakeLoader(), target=25_000_000, total_steps=3000, precision="bf16",
        seed=1234, hparams_extra={"early_stop_patience": 4, "early_stop_min_delta": 0.005})

    saved = json.loads((tmp_path / "audit.json").read_text())
    assert saved == rec                                          # persisted verbatim
    assert saved["stage"] == 5 and saved["seed"] == 1234
    assert saved["model"]["params"] == 11_000_000
    assert saved["hparams"]["lr"] == 3e-4 and saved["hparams"]["early_stop_patience"] == 4
    assert saved["data"]["sources"][0]["tokens"] == 1234567     # per-source provenance
    assert saved["rehearsal"]["weights_pct"] == [90.0, 10.0]    # size-weighted mix
    assert "command" in saved and "started" in saved
