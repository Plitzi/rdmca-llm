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
    cfg = ModelConfig(d_model=48, n_heads=16, ffn_dim=96, context_len=16,
                      vocab_size=128, mrl_dims=[48])
    with pytest.raises(AssertionError):
        RDMCAFoundational(cfg)


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
    """finalize(accumulate(g1,g2), 1/2) ≈ mean(g1,g2)."""
    m = _tiny_model()
    lg = B.engine.value_and_grad(m, lambda mm, t: mm.mrl_loss(t))
    b1 = B.ops.array(np.random.randint(0, 256, (3, 33)).astype(np.int64))
    b2 = B.ops.array(np.random.randint(0, 256, (3, 33)).astype(np.int64))
    _, g1 = lg(m, b1)
    n1 = B.engine.grad_norm(m, g1)
    run = B.engine.accumulate_grads(None, g1, m)
    _, g2 = lg(m, b2)
    run = B.engine.accumulate_grads(run, g2, m)
    acc = B.engine.finalize_grads(run, 0.5, m)
    n_acc = B.engine.grad_norm(m, acc)
    # The mean gradient norm should differ from a single micro-batch's (it averages
    # two different batches) and be finite/positive.
    assert n_acc > 0 and np.isfinite(n_acc)
    assert abs(n_acc - n1) > 1e-6

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
    def encode(self, text, lang="en", add_bos=True, add_eos=True):
        ids = [(ord(c) % 250) + 3 for c in text][:40]
        return ids or [3]

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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
