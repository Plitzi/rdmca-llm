"""
Regression tests for the 2026-06 fix batch (chat, training, architecture, data).
Self-contained: no trained tokenizer / prepared data / checkpoints required — uses
the MLX backend (always present here), fakes, and temp dirs.
"""
import sys, json, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

import src.backend as backend
backend.select("mlx")                      # bind model/loader modules to MLX
B = backend.current()

from src.model.transformer import RDMCAFoundational
from src.model.config import ModelConfig


# ─────────────────────────── architecture: config validation (M5/L8) ─────────

def test_modelconfig_rejects_unsorted_mrl():
    with pytest.raises(ValueError):
        ModelConfig(d_model=256, n_heads=4, mrl_dims=[256, 128])

def test_modelconfig_rejects_duplicate_mrl():
    with pytest.raises(ValueError):
        ModelConfig(d_model=256, n_heads=4, mrl_dims=[128, 128, 256])

def test_modelconfig_rejects_mrl_over_dmodel():
    with pytest.raises(ValueError):
        ModelConfig(d_model=256, n_heads=4, mrl_dims=[128, 512])

def test_modelconfig_rejects_indivisible_heads():
    with pytest.raises(ValueError):
        ModelConfig(d_model=100, n_heads=8, mrl_dims=[100])

def test_attention_rejects_odd_head_dim():
    # d_model 48 / n_heads 16 = head_dim 3 (odd) → RoPE assert in attention init.
    # Match the message so a DIFFERENT assertion firing first can't mask a regression.
    cfg = ModelConfig(d_model=48, n_heads=16, ffn_dim=96, context_len=16,
                      vocab_size=128, mrl_dims=[48])
    with pytest.raises(AssertionError, match="head_dim must be even"):
        RDMCAFoundational(cfg)


def test_experience_log_only_saves_signal_bearing_turns():
    """A turn with no feedback is NOT saved (no benefit); a corrected turn learns the
    CORRECTION, not the model's wrong answer."""
    from src.memory.experience_log import log_experience, load_experiences, detect_correction
    with tempfile.TemporaryDirectory() as td:
        p = str(Path(td) / "e.jsonl")
        assert log_experience("hi", "hello", feedback="neutral", path=p) is False
        assert log_experience("2+2?", "4", feedback="accepted", path=p) is True
        assert log_experience("cap of France?", "London", feedback="corrected",
                              correction="Paris", path=p) is True
        recs = load_experiences(p)
        assert len(recs) == 2                                  # neutral was dropped
        corr = next(r for r in recs if r["feedback"] == "corrected")
        assert "Paris" in corr["text"] and "London" not in corr["text"]
    # implicit correction detection (EN + ES), no false positives on a new topic
    assert detect_correction("no, it is Paris") and detect_correction("eso está mal")
    assert not detect_correction("what about Spain?")


def test_relevance_feedback_overrides_utility():
    """A `corrected` experience must score higher R⁺ than the same content unlabeled —
    feedback is the ground-truth Utility (error-driven learning gets the boost)."""
    from src.relevance.engine import RelevanceEngine
    from src.memory.episodic_buffer import Experience
    re = RelevanceEngine(ltss=None)
    re.update_state(np.zeros(64, dtype=np.float32))
    emb = np.random.randn(64).astype(np.float32)
    neutral   = Experience(text="x", embedding=emb, feedback="neutral");   neutral.episodic_context = []
    corrected = Experience(text="x", embedding=emb, feedback="corrected"); corrected.episodic_context = []
    assert re.score(corrected) > re.score(neutral)


def test_dialogue_interleave_and_emotion_balance():
    """Dialogue mixing: _interleave round-robins all sources (no front-loaded block),
    and empathetic streaming caps per emotion so moods stay balanced."""
    import src.data.graded as g
    # round-robin: drains every source, no loss, interleaved order
    def gen(tag, n):
        for i in range(n):
            yield {"text": f"{tag}{i}"}
    out = [r["text"] for r in g._interleave(gen("A", 3), gen("B", 1), gen("C", 2))]
    assert out[:3] == ["A0", "B0", "C0"]                  # round-robin, not blocks
    assert sorted(out) == ["A0", "A1", "A2", "B0", "C0", "C1"]   # nothing dropped

    # emotion cap balances a skewed source (sad×50, joyful×3 → 5 + 3 = 8 at cap 5)
    import datasets as _d
    orig = _d.load_dataset

    class _Fake:
        def __iter__(self):
            for _ in range(50):
                yield {"emotion": "sad",
                       "conversations": [{"role": "u", "content": "x"}, {"role": "a", "content": "y"}]}
            for _ in range(3):
                yield {"emotion": "joyful",
                       "conversations": [{"role": "u", "content": "x"}, {"role": "a", "content": "y"}]}
    _d.load_dataset = lambda *a, **k: _Fake()
    try:
        n = sum(1 for _ in g._stream_empathetic_balanced(per_emotion_cap=5))
    finally:
        _d.load_dataset = orig
    assert n == 8, f"emotion cap not balancing: got {n}, expected 8 (5 sad + 3 joyful)"


def test_confidence_validator_routes_by_knowledge():
    """The confidence-gated validator: human-labelled or high-coherence → self-approve;
    mid → defer (no external source); very low → escalate to human."""
    from src.consolidation.validation import (ExperienceValidator, HumanReviewSource,
                                              SelfKnowledgeSource)

    class _Exp:
        def __init__(self, feedback="neutral"):
            import uuid as _u
            self.feedback = feedback; self.uid = str(_u.uuid4()); self.text = "x"

    class _FakeQueue:
        def __init__(self): self.queued = []
        def queue_for_review(self, exp, score, rationale=""): self.queued.append(exp.uid)

    q = _FakeQueue()
    v = ExperienceValidator(human_source=HumanReviewSource(q))   # no external sources

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
    from src.consolidation.validation import (default_validator, PeerModelSource,
                                              WebResearchSource)
    assert PeerModelSource().available() is False
    assert WebResearchSource().available() is False
    v = default_validator(ambiguity_handler=None)               # no human either

    class _Exp:
        feedback = "neutral"; uid = "u"; text = "x"
    # mid confidence, all external inert, no human → defer (no crash)
    assert v.decide(_Exp(), coherence=0.5).fate == "defer"


def test_output_head_is_weight_tied_to_embedding():
    """The output projection must REUSE embed.weight (weight tying): no separate
    'head' param in the tree, and head_at_dim(h, d) == h[:, :d] @ embed.weight[:, :d].T."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    m = _tiny_model()
    names = [k for k, _ in tree_flatten(m.parameters())]
    assert not any(".head." in k or k.startswith("head.") for k in names), \
        f"a separate output head leaked into the param tree: {[k for k in names if 'head' in k]}"
    h = mx.array(np.random.randn(2, 5, 64).astype(np.float32))
    for d in (32, 64):
        got = np.array(m.head_at_dim(h, d).tolist())
        want = np.array((h[..., :d] @ m.embed.weight[:, :d].T).tolist())
        assert np.allclose(got, want, atol=1e-5), f"head_at_dim not tied to embed at d={d}"


def test_model_is_causal():
    """Output at position k must NOT depend on tokens after k (causal masking):
    changing a future token leaves all earlier positions' hidden states identical."""
    import mlx.core as mx
    m = _tiny_model(); m.train(False)
    base = np.random.randint(1, 256, (1, 16))
    a = base.copy(); b = base.copy()
    b[0, 10:] = (b[0, 10:] + 7) % 256          # perturb only positions ≥10
    ha = np.array(m(mx.array(a)).tolist())
    hb = np.array(m(mx.array(b)).tolist())
    # positions 0..9 saw identical context → must be bit-close; position ≥10 differs.
    assert np.allclose(ha[:, :10], hb[:, :10], atol=1e-5), "future token leaked into past"
    assert not np.allclose(ha[:, 10:], hb[:, 10:], atol=1e-5), "perturbation had no effect"


# ─────────────────────────── architecture: MRL uniform weights (C1) ──────────

def test_mrl_weights_are_uniform():
    """mrl_loss must equal the simple mean of per-dim cross-entropies (uniform
    weighting), not a 1/d-weighted sum that starves the full head."""
    import mlx.core as mx
    cfg = ModelConfig(d_model=64, n_heads=2, ffn_dim=128, context_len=16,
                      vocab_size=256, mrl_dims=[32, 64], dropout=0.0)  # deterministic
    m = RDMCAFoundational(cfg)
    toks = mx.array(np.random.randint(0, 256, (2, 17)))
    total = m.mrl_loss(toks)
    inputs, targets = toks[:, :-1], toks[:, 1:]
    h = m(inputs)
    per = []
    for d in cfg.mrl_dims:
        lg = m.head_at_dim(h, d)
        Bsz, S, V = lg.shape
        per.append(B.ops.cross_entropy(lg.reshape(Bsz * S, V),
                                       targets.reshape(Bsz * S), reduction="mean"))
    expected = (per[0] + per[1]) / 2.0
    assert abs(float(total.item()) - float(expected.item())) < 1e-3


def test_eval_ce_mask_matches_training_objective():
    """Validation eval_ce with a completion mask must equal the masked mean over the
    unmasked (assistant) targets — the SAME objective training optimizes — and must
    differ from the unmasked full-sequence mean. Otherwise the gate measures context
    tokens the model is never trained to predict and perplexity is inflated ~7×."""
    import mlx.core as mx
    cfg = ModelConfig(d_model=64, n_heads=2, ffn_dim=128, context_len=16,
                      vocab_size=256, mrl_dims=[32, 64], dropout=0.0)
    m = RDMCAFoundational(cfg)
    toks = mx.array(np.random.randint(0, 256, (2, 17)))
    mask = np.zeros((2, 17), dtype=np.int32); mask[:, 9:] = 1   # only the tail trains
    mask_mx = mx.array(mask)

    masked   = float(m.eval_ce(toks, mask=mask_mx).item())
    unmasked = float(m.eval_ce(toks).item())

    # Manual masked-mean reference at full dim.
    inputs, targets = toks[:, :-1], toks[:, 1:]
    lg = m.head_at_dim(m(inputs), cfg.mrl_dims[-1])
    Bsz, S, V = lg.shape
    ce = B.ops.cross_entropy(lg.reshape(Bsz * S, V), targets.reshape(Bsz * S),
                             reduction="none")
    mm = mask_mx[:, 1:].reshape(Bsz * S).astype(ce.dtype)
    ref = float((B.ops.sum(ce * mm) / B.ops.sum(mm)).item())
    assert abs(masked - ref) < 1e-3
    assert abs(masked - unmasked) > 1e-3        # masking actually changes the metric

    # An all-ones mask reduces to the plain mean (prose stages are unaffected).
    allone = float(m.eval_ce(toks, mask=mx.ones((2, 17))).item())
    assert abs(allone - unmasked) < 1e-3


# ─────────────────────────── MoE: top_k restored on grow (M4) ────────────────

def test_moe_top_k_restored_on_grow():
    from src.model.moe import SectorGate
    g = SectorGate(d_model=32, n_experts=1, top_k=2)
    assert g.top_k == 1                     # capped to available experts
    g.grow_experts(5)
    assert g.n_experts == 6
    assert g.top_k == 2                     # restored toward the configured target


# ─────────────────────────── training: grad accumulation (C4) ────────────────

def _tiny_model():
    cfg = ModelConfig(d_model=64, n_heads=2, ffn_dim=128, context_len=32,
                      vocab_size=256, mrl_dims=[64])
    return RDMCAFoundational(cfg)

def test_accumulate_finalize_is_mean_of_microbatches():
    """finalize(accumulate(g1,g2), 1/2) must equal (g1+g2)/2 ELEMENT-WISE — not
    just differ in norm (a test that only checked the norm would pass even if
    finalize returned g1 unchanged)."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    m = _tiny_model()
    lg = B.engine.value_and_grad(m, lambda mm, t: mm.mrl_loss(t))
    b1 = B.ops.array(np.random.randint(0, 256, (3, 33)).astype(np.int64))
    b2 = B.ops.array(np.random.randint(0, 256, (3, 33)).astype(np.int64))
    _, g1 = lg(m, b1)
    _, g2 = lg(m, b2)
    run = B.engine.accumulate_grads(None, g1, m)
    run = B.engine.accumulate_grads(run, g2, m)
    acc = B.engine.finalize_grads(run, 0.5, m)
    # Element-wise: every leaf of `acc` equals the mean of the two micro-batch grads.
    f1, f2, fa = dict(tree_flatten(g1)), dict(tree_flatten(g2)), dict(tree_flatten(acc))
    leaves = [k for k in fa if isinstance(fa[k], mx.array)]
    assert leaves, "no gradient leaves to compare"
    for k in leaves:
        expected = (f1[k].astype(mx.float32) + f2[k].astype(mx.float32)) / 2.0
        d = float(mx.max(mx.abs(fa[k].astype(mx.float32) - expected)).item())
        assert d < 1e-5, f"{k}: accumulated grad is not the mean (max|Δ|={d:.2e})"

def test_clip_grads_caps_norm():
    m = _tiny_model()
    lg = B.engine.value_and_grad(m, lambda mm, t: mm.mrl_loss(t))
    b = B.ops.array(np.random.randint(0, 256, (3, 33)).astype(np.int64))
    _, g = lg(m, b)
    clipped = B.engine.clip_grads(m, g, 0.1)
    assert B.engine.grad_norm(m, clipped) <= 0.1 + 1e-3

def test_clip_grads_noop_when_under_threshold():
    m = _tiny_model()
    lg = B.engine.value_and_grad(m, lambda mm, t: mm.mrl_loss(t))
    b = B.ops.array(np.random.randint(0, 256, (3, 33)).astype(np.int64))
    _, g = lg(m, b)
    n0 = B.engine.grad_norm(m, g)
    same = B.engine.clip_grads(m, g, 1e9)
    assert abs(B.engine.grad_norm(m, same) - n0) < 1e-4


# ─────────────────────────── backend surface completeness ────────────────────

def test_backend_surface_complete_both():
    from src.backend.base import check_surface
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # benign "switching backend" notice
        for name in ("mlx", "torch"):
            try:
                backend.select(name)
            except Exception:
                continue                    # backend not installed in this env
            assert check_surface(backend.current()) == []
        backend.select("mlx")               # restore


# ─────────────────────────── agent: turn-boundary cleanup (chat) ──────────────

def test_clean_answer_cuts_inline_role_tag():
    from src.agent import clean_answer
    leak = "I'm not sure. User: hi Assistant: hello"
    assert clean_answer(leak) == "I'm not sure."

def test_first_stop_index_inline_and_ignores_leading():
    from src.agent import first_stop_index
    assert first_stop_index("ok. User: x") == 4         # inline boundary
    assert first_stop_index("Assistant: hi") is None    # leading primed tag ignored

def test_safe_stream_len_holds_back_role_prefix():
    from src.agent import safe_stream_len
    # trailing "User" could still become "User:" → held back
    assert safe_stream_len("done. User") == len("done. ")
    # plain text fully emittable
    assert safe_stream_len("hello there") == len("hello there")

def test_strip_thinking_removes_scratchpad():
    from src.agent import strip_thinking
    assert strip_thinking("<think>plan</think>answer").strip() == "answer"


# ─────────────────────────── sampling: rep penalty + top_k (L6) ───────────────

def test_rep_penalty_demotes_recent_token():
    from uses.chat.run_chat import sample_top_p
    import mlx.core as mx
    logits = mx.array(np.array([1.0, 5.0, 1.0, 1.0], dtype=np.float32))  # token 1 peaks
    # temperature 0 → argmax; penalizing token 1 hard should move the choice off it.
    out = sample_top_p(logits, temperature=0.0, top_p=1.0,
                       recent_ids=[1], rep_penalty=10.0)
    assert out != 1

def test_top_k_restricts_choices():
    from uses.chat.run_chat import sample_top_p
    import mlx.core as mx
    logits = mx.array(np.array([10.0, 9.0, -50.0, -50.0], dtype=np.float32))
    picks = {sample_top_p(logits, temperature=1.0, top_p=1.0, top_k=2) for _ in range(50)}
    assert picks <= {0, 1}                  # only the top-2 are ever sampled


# ─────────────────────────── tokenizer: central control symbols (#1, C2) ──────

def test_tokenizer_symbols_include_control_and_modality():
    from src.modalities.vocab import tokenizer_symbols, CONTROL_SPECIALS
    syms = tokenizer_symbols(["en", "es"])
    for s in ("<lang:en>", "<lang:es>", "<mod:text>", "<think>", "</think>",
              "<tool_call>", "</tool_call>"):
        assert s in syms
    assert "<think>" in CONTROL_SPECIALS

def test_agent_think_delimiters_match_registry():
    from src.agent import THINK_OPEN, THINK_CLOSE
    from src.modalities.vocab import REASONING_SPECIALS
    assert [THINK_OPEN, THINK_CLOSE] == REASONING_SPECIALS


# ─────────────────────────── data loader: interleave + weights (loader) ───────

class _FakeTok:
    """Minimal tokenizer for the loader: deterministic ids, no special tokens."""
    lang_tokens: dict = {}
    def encode(self, text, lang="en", add_bos=True, add_eos=True):
        ids = [(ord(c) % 250) + 3 for c in text][:40]
        return ids or [3]
    def encode_raw(self, text):
        return [(ord(c) % 250) + 5 for c in text]

def _write_corpus(d: Path):
    # Big "story" file and small "dialogue" file, each tagged so we can identify source.
    with open(d / "story.jsonl", "w") as f:
        for _ in range(2000):
            f.write(json.dumps({"text": "STORY " + "x" * 60}) + "\n")
    with open(d / "dialogue.jsonl", "w") as f:
        for _ in range(400):
            f.write(json.dumps({"text": "DIALOG " + "y" * 60}) + "\n")

def test_loader_interleaves_sources_no_block():
    """Records from both files must be mixed throughout — not one whole file then
    the other (which caused catastrophic forgetting)."""
    from src.data.loader import TextDataset
    with tempfile.TemporaryDirectory() as td:
        d = Path(td); _write_corpus(d)
        ds = TextDataset(str(d), _FakeTok(), seq_len=16, batch_size=2,
                         shuffle=True, shuffle_buffer=50)
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
        d = Path(td); _write_corpus(d)
        def dialog_share(weights):
            ds = TextDataset(str(d), _FakeTok(), seq_len=16, batch_size=2,
                             shuffle=True, shuffle_buffer=50, source_weights=weights)
            n_d = n_t = 0
            for i, rec in enumerate(ds._iter_records()):
                if rec["text"].startswith("DIALOG"): n_d += 1
                else: n_t += 1
                if i >= 1500:
                    break
            return n_d / (n_d + n_t)
        base = dialog_share(None)
        boosted = dialog_share({"dialogue": 5.0})
        assert boosted > base + 0.1         # oversampling clearly lifts the share


# ─────────────────────────── data: held-out val split (H7) ───────────────────

def test_loader_excludes_val_files_from_training():
    from src.data.loader import TextDataset
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        with open(d / "story.jsonl", "w") as f:
            for _ in range(50): f.write(json.dumps({"text": "TRAIN sample text here"}) + "\n")
        with open(d / "story.val.jsonl", "w") as f:
            for _ in range(20): f.write(json.dumps({"text": "VAL sample text here"}) + "\n")
        train = TextDataset(str(d), _FakeTok(), seq_len=8, batch_size=1, shuffle=False)
        val   = TextDataset(str(d), _FakeTok(), seq_len=8, batch_size=1, shuffle=False, val=True)
        train_txt = {r["text"].split()[0] for r in train._iter_records()}
        val_txt   = {r["text"].split()[0] for r in val._iter_records()}
        assert train_txt == {"TRAIN"}       # training never sees the held-out file
        assert val_txt == {"VAL"}           # val loads only the held-out file

def test_write_jsonl_routes_holdout():
    import scripts.prepare_data as pd
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "src.jsonl"
        recs = ({"text": f"document number {i} with enough words to count"} for i in range(500))
        pd.write_jsonl(recs, out, token_budget_m=999, verbose=False, val_fraction=0.1)
        val = out.with_suffix(".val.jsonl")
        assert val.exists()
        n_train = sum(1 for _ in open(out))
        n_val   = sum(1 for _ in open(val))
        assert n_val > 0 and n_train > 0
        assert 0.05 < n_val / (n_train + n_val) < 0.15      # ~10% held out


# ─────────────────────────── training: optimizer state resume (M1) ───────────

def test_optimizer_state_roundtrip():
    """save_optimizer → load_optimizer restores AdamW moments exactly (warm resume)."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    m = _tiny_model()
    opt = B.engine.make_optimizer(m, 5e-4, 0.1)
    lg = B.engine.value_and_grad(m, lambda mm, t: mm.mrl_loss(t))
    for _ in range(4):                          # populate optimizer state
        b = B.ops.array(np.random.randint(0, 256, (3, 33)).astype(np.int64))
        loss, g = lg(m, b); B.engine.optimizer_step(opt, m, g); B.engine.eval(loss)
    with tempfile.TemporaryDirectory() as td:
        p = str(Path(td) / "s.opt")
        B.engine.save_optimizer(opt, p)
        opt2 = B.engine.make_optimizer(m, 5e-4, 0.1)
        assert B.engine.load_optimizer(opt2, p) is True
        s1 = dict(tree_flatten(opt.state)); s2 = dict(tree_flatten(opt2.state))
        keys = [k for k in s1 if isinstance(s1[k], mx.array) and isinstance(s2.get(k), mx.array)]
        assert keys, "optimizer state should have array leaves"
        for k in keys:
            d = float(mx.max(mx.abs(s1[k].astype(mx.float32) - s2[k].astype(mx.float32))).item())
            assert d < 1e-5
        # absent file → graceful False (cold start)
        assert B.engine.load_optimizer(opt2, str(Path(td) / "missing.opt")) is False


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
    ds = TextDataset.__new__(TextDataset); ds.tokenizer = _FakeTok()
    ids, mask = ds._encode_record(
        {"text": "System: s\nUser: q\nAssistant: a", "lang": "en"})
    assert len(ids) == len(mask) and sum(mask) > 0
    assert sum(mask) < len(mask)              # context is masked out
    assert ids[-1] == EOS_ID and mask[-1] == 1  # learns to stop after the answer
    # every trained position lies inside the assistant turn (after "Assistant")
    raw = _FakeTok().encode_raw("\nAssistant: a")
    assert sum(mask) == len(raw) + 1          # assistant turn + its EOS

def test_completion_mask_drops_record_with_no_response():
    from src.data.loader import TextDataset
    ds = TextDataset.__new__(TextDataset); ds.tokenizer = _FakeTok()
    ids, mask = ds._encode_record({"text": "User: q\nUser: q2", "lang": "en"})
    assert ids == [] and mask == []           # nothing to train on → skipped

def test_mrl_loss_all_ones_mask_equals_unmasked():
    """An all-ones mask must reproduce the plain mean cross-entropy exactly."""
    import mlx.core as mx
    cfg = ModelConfig(d_model=64, n_heads=2, ffn_dim=128, context_len=16,
                      vocab_size=256, mrl_dims=[32, 64], dropout=0.0)
    m = RDMCAFoundational(cfg)
    toks = mx.array(np.random.randint(0, 256, (2, 17)))
    plain  = float(m.mrl_loss(toks).item())
    masked = float(m.mrl_loss(toks, mx.ones((2, 17), dtype=mx.int32)).item())
    assert abs(plain - masked) < 1e-4

def test_mrl_loss_mask_equals_manual_restricted_mean():
    """The masked loss must equal the per-dim CE averaged over ONLY the unmasked
    target positions (the completion-only contract), independently recomputed."""
    import mlx.core as mx
    cfg = ModelConfig(d_model=64, n_heads=2, ffn_dim=128, context_len=16,
                      vocab_size=256, mrl_dims=[32, 64], dropout=0.0)
    m = RDMCAFoundational(cfg)
    toks = mx.array(np.random.randint(0, 256, (2, 17)))
    mask = np.zeros((2, 17), dtype=np.int32); mask[:, 9:] = 1   # train only the tail
    got = float(m.mrl_loss(toks, mx.array(mask)).item())
    inputs, targets = toks[:, :-1], toks[:, 1:]
    tmask = mx.array(mask[:, 1:]).reshape(-1).astype(mx.float32)
    h = m(inputs)
    per = []
    for d in cfg.mrl_dims:
        lg = m.head_at_dim(h, d); Bsz, S, V = lg.shape
        ce = B.ops.cross_entropy(lg.reshape(Bsz * S, V),
                                 targets.reshape(Bsz * S), reduction="none")
        per.append(float((mx.sum(ce * tmask) / mx.sum(tmask)).item()))
    expected = (per[0] + per[1]) / 2.0
    assert abs(got - expected) < 1e-3

def test_loader_with_mask_yields_token_mask_pairs():
    from src.data.loader import TextDataset
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        with open(d / "chat.jsonl", "w") as f:
            for _ in range(200):
                f.write(json.dumps({"text": "User: hi\nAssistant: hello there friend"}) + "\n")
        ds = TextDataset(str(d), _FakeTok(), seq_len=16, batch_size=2,
                         shuffle=False, with_mask=True)
        toks, mask = next(iter(ds.batches()))
        assert toks.shape == (2, 17) and mask.shape == (2, 17)
        assert set(np.unique(mask)).issubset({0, 1})
        assert 0 < int(mask.sum()) < mask.size   # some trained, some masked


# ───────────────── system prompt + mood (conversational layer) ───────────────

def test_emotion_maps_to_mood_palette():
    from src.modalities.moods import emotion_to_mood, MOODS
    assert emotion_to_mood("joyful") == "happy"
    assert emotion_to_mood("terrified") == "afraid"
    assert emotion_to_mood("caring") == "caring"
    assert emotion_to_mood("totally-unknown-emotion") == "neutral"   # default
    assert emotion_to_mood(None) == "neutral"
    assert all(emotion_to_mood(e) in MOODS for e in
               ("sad", "angry", "surprised", "proud", "anxious"))

def test_mood_system_phrase_neutral_is_empty():
    from src.modalities.moods import mood_system_phrase
    assert mood_system_phrase("neutral") == ""        # default adds nothing
    assert mood_system_phrase("happy") == "(mood: happy)"
    assert mood_system_phrase("bogus") == ""          # unknown → nothing

def test_system_preamble_framing():
    from src import agent
    assert agent.system_preamble(None, "neutral") == ""             # nothing → no line
    assert agent.system_preamble("Be kind.", "neutral") == "System: Be kind.\n"
    assert agent.system_preamble(None, "sad") == "System: (mood: sad)\n"
    assert agent.system_preamble("Be kind.", "happy") == "System: Be kind. (mood: happy)\n"

def test_agent_prompt_prepends_system_persona():
    from src import agent
    p = agent.build_agent_prompt([], "hello", system="You are terse.")
    assert p.startswith("System: You are terse. ")     # persona ahead of tool spec
    assert "User: hello" in p and p.rstrip().endswith("Assistant:")

def test_data_enrichment_system_and_story():
    """instruct system injection yields a System line; story reframing is a NATURAL
    User→Assistant request with NO system prompt (telling a story needs no persona)."""
    import src.data.graded as g
    sysd = g._prepend_system("User: q\nAssistant: a", "You are kind.", "happy")
    assert sysd.startswith("System: You are kind. (mood: happy)\nUser:")
    # the story-request format the stream emits
    story = f"User: {g._STORY_PROMPTS[0]}\nAssistant: Once upon a time."
    assert not story.startswith("System:")             # no persona gate for stories
    assert "Assistant:" in story

def test_completion_mask_handles_system_prefixed_record():
    """A System-prefixed transcript still masks System+User and trains only the
    Assistant turn — the enriched data stays compatible with completion masking."""
    from src.data.loader import TextDataset
    ds = TextDataset.__new__(TextDataset); ds.tokenizer = _FakeTok()
    ids, mask = ds._encode_record(
        {"text": "System: be kind (mood: happy)\nUser: hi\nAssistant: hello", "lang": "en"})
    assert 0 < sum(mask) < len(mask)
    trained = _FakeTok().encode_raw("\nAssistant: hello")
    assert sum(mask) == len(trained) + 1               # assistant turn + EOS only

def test_classify_mood_defaults_neutral_without_head():
    from src.model.mood import classify_mood
    mood, conf = classify_mood(None, None, None, "anything")
    assert mood == "neutral"

def test_mood_tracker_neutral_without_head():
    from src.model.mood import MoodTracker
    t = MoodTracker(None)
    assert t.update(None, None, "I am so happy!") == "neutral"   # one msg ⇒ inertia


def test_lexicon_mood_fixes_broken_classifications():
    """The learned 11M head was near-random ('im good'→angry, 'my dog died'→caring,
    requests→emotion). The lexicon is the reliable floor: clear cues map correctly,
    requests/questions stay neutral, and negation flips a positive cue to sad."""
    from src.modalities.moods import lexicon_mood
    cases = {
        "im good": "happy", "i am so happy today": "happy", "thanks for your help": "happy",
        "my dog died": "sad", "i feel terrible": "sad", "i am not good": "sad",
        "i hate this": "angry", "im scared of the dark": "afraid",
        "tell me a story": "neutral", "what is 2+2": "neutral",
        "can you help me with math": "neutral", "how are you": "neutral",
    }
    for text, want in cases.items():
        got, _ = lexicon_mood(text)
        assert got == want, f"{text!r}: got {got}, want {want}"


def test_mood_tracker_lexicon_drives_mood_without_a_head():
    """No learned head needed: a sustained emotional tone is detected by the lexicon
    alone (the head is only an optional refinement)."""
    from src.model.mood import MoodTracker
    t = MoodTracker(None, alpha=0.5)
    last = "neutral"
    for _ in range(5):
        last = t.update(None, None, "i am so happy and grateful")
    assert last == "happy"
    for _ in range(8):
        last = t.update(None, None, "tell me about cats")     # neutral request
    assert last == "neutral"                                   # decays back

def test_mood_tracker_builds_and_decays_over_conversation(monkeypatch):
    """Conversation-aware mood: one message isn't enough (inertia), a sustained tone
    takes hold, and it decays back to neutral — emotion is the WHOLE exchange."""
    import src.model.mood as mood
    from src.model.mood import MoodTracker, MOOD_INDEX, MOODS
    happy = [0.0] * len(MOODS); happy[MOOD_INDEX["happy"]] = 1.0
    neutral = [0.0] * len(MOODS); neutral[0] = 1.0
    monkeypatch.setattr(mood, "mood_probs",
                        lambda m, t, h, text, **k: happy if "joy" in text else neutral)
    tr = MoodTracker(head=object(), alpha=0.4)
    assert tr.update(None, None, "joy") == "neutral"        # one message ⇒ inertia
    for _ in range(4):
        m = tr.update(None, None, "joy")
    assert m == "happy"                                     # sustained tone takes hold
    for _ in range(6):
        m = tr.update(None, None, "calm")
    assert m == "neutral"                                   # decays back to default

def test_mood_tracker_reset(monkeypatch):
    import src.model.mood as mood
    from src.model.mood import MoodTracker, MOOD_INDEX, MOODS
    happy = [0.0] * len(MOODS); happy[MOOD_INDEX["happy"]] = 1.0
    monkeypatch.setattr(mood, "mood_probs", lambda *a, **k: happy)
    tr = MoodTracker(head=object(), alpha=0.6)
    for _ in range(5):
        tr.update(None, None, "x")
    assert tr.current() == "happy"
    tr.reset()
    assert tr.current() == "neutral"

def test_mood_head_learns_to_separate_moods():
    """The mood head should fit a tiny separable set (sanity that the classifier
    + train step are wired): loss drops over a few steps on frozen features."""
    import mlx.core as mx
    from src.model.mood import MoodHead, mood_loss, MOOD_INDEX
    head = MoodHead(d_model=32, hidden=16)
    opt  = B.engine.make_optimizer(head, 1e-2, 0.0)
    rng  = np.random.RandomState(0)
    # 3 clusters of features → 3 mood labels; learnable by a small MLP.
    centers = rng.randn(3, 32)
    feats   = np.vstack([centers[i] + 0.05 * rng.randn(20, 32) for i in range(3)]).astype(np.float32)
    labels  = np.array([i for i in range(3) for _ in range(20)], dtype=np.float32)
    h = mx.array(feats); y = mx.array(labels)
    def loss_fn(hd): return mood_loss(hd(h), y)
    lg = B.engine.value_and_grad(head, loss_fn)
    first = float(lg(head)[0].item())
    for _ in range(60):
        loss, grads = lg(head); B.engine.optimizer_step(opt, head, grads)
    assert float(loss.item()) < first - 0.3            # clearly learned


# ───────────────── context / token accounting (observability + billing) ──────

def test_context_report_accounting_and_billing_dict():
    from src.observability import ContextReport
    r = ContextReport(surface="chat", context_len=512, system_tokens=12,
                      history_tokens=300, tokens_in=320, tokens_out=28,
                      tokens_reasoning=15, mood="happy",
                      mood_dist={"happy": 0.42, "neutral": 0.3, "caring": 0.1},
                      memory_files=3, tps=200.0,
                      params={"temp": 0.7, "top_p": 0.9})
    assert r.used == 348 and r.free == 512 - 348
    assert round(r.fill_pct, 1) == round(100 * 348 / 512, 1)
    d = r.to_dict()                                  # billing/telemetry payload
    for k in ("surface", "tokens_in", "tokens_out", "tokens_reasoning",
              "system_tokens", "mood", "memory_files", "used", "free", "fill_pct"):
        assert k in d
    assert d["used"] == 348 and d["mood"] == "happy"
    panel = r.render()
    assert "tokens in" in panel and "window" in panel and "mood" in panel
    assert r.render_compact().startswith("  [in 320 · out 28 · think 15")

def test_count_tokens_is_safe():
    from src.observability import count_tokens
    class _T:
        def encode(self, t, add_bos=True, add_eos=True): return list(range(len(t)))
    assert count_tokens(_T(), "abcd") == 4
    assert count_tokens(_T(), "") == 0
    assert count_tokens(None, "x") == 0              # no tokenizer ⇒ 0, never crashes


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
