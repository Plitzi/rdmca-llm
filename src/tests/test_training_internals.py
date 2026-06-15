"""Training-subsystem internals that don't need a trained tokenizer or a real corpus:
model build (setup), the ratcheting graduation gate, masked validation perplexity, the
entry-PP baseline, and the data loader's no-tokenizer guard. The full text training loop
is exercised by a real run (see GUIDE); here we cover the pure logic around it."""

import numpy as np
import pytest

import src.backend as backend
from src.plugins import set_active_model
from src.training import gates

_TINY = {
    "model_name": "cognition",
    "level": 0,
    "training": {"precision": "fp32", "seed": 0},
    "model": {
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 2,
        "n_kv_heads": 1,
        "ffn_dim": 128,
        "context_len": 64,
        "vocab_size": 256,
        "mrl_dims": [32, 64],
        "dropout": 0.0,
        "rope_theta": 10000.0,
    },
}


def _build_tiny(tmp_path):
    set_active_model("cognition")
    from src.training.setup import build_stage_model

    model, model_cfg, _adapter, _precision, _seed = build_stage_model(1, _TINY, tmp_path)
    return model, model_cfg


def _fake_val_batches(n=3, b=2, s=16, vocab=256):
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        toks = rng.integers(0, vocab, size=(b, s)).astype(np.int64)
        mask = np.ones((b, s), dtype=np.float32)
        out.append((toks, mask))
    return out


def test_build_stage_model_random_init(tmp_path):
    model, model_cfg = _build_tiny(tmp_path)
    assert model_cfg.d_model == 64 and model_cfg.n_layers == 2
    assert model.count_params() > 0


def test_validation_perplexity_is_finite(tmp_path):
    model, _ = _build_tiny(tmp_path)
    ppl = gates.validation_perplexity(model, _fake_val_batches())
    assert np.isfinite(ppl) and ppl > 1.0


def test_validation_perplexity_accepts_bare_tokens(tmp_path):
    model, _ = _build_tiny(tmp_path)
    bare = [b[0] for b in _fake_val_batches()]  # tokens without a mask (back-compat path)
    assert np.isfinite(gates.validation_perplexity(model, bare))


def test_evaluate_gate_returns_score_and_decision(tmp_path, capsys):
    model, _ = _build_tiny(tmp_path)
    score, passed = gates.evaluate_gate(model, 1, _fake_val_batches(), cfg=_TINY, step=10)
    assert np.isfinite(score) and isinstance(passed, bool)
    assert "[gate]" in capsys.readouterr().out


def test_stage_entry_ppl_persists_and_reuses(tmp_path):
    model, _ = _build_tiny(tmp_path)
    vb = _fake_val_batches()
    first = gates.stage_entry_ppl(model, tmp_path, vb, resume=False)
    assert (tmp_path / "entry.json").exists()
    # On resume the persisted value is reused verbatim (not recomputed).
    reused = gates.stage_entry_ppl(model, tmp_path, vb, resume=True)
    assert reused == first


def test_gate_threshold_default_and_override():
    assert gates.gate_threshold(1) == gates.DEFAULT_GATE_PPL[1]
    cfg = {"gate": {"max_perplexity": {1: 12.5}}}
    assert gates.gate_threshold(1, cfg) == 12.5


def test_gate_decision_ratchet_branches():
    thr = 40.0
    # above the floor → never a candidate
    assert gates.gate_decision(50.0, float("inf"), thr) == (False, False, False)
    # first below-floor score → candidate + new best + meaningful
    assert gates.gate_decision(30.0, float("inf"), thr) == (True, True, True)
    # tiny improvement → new best but NOT meaningful (plateau)
    cand, best, meaningful = gates.gate_decision(29.999, 30.0, thr, min_delta=0.01)
    assert cand and best and not meaningful
    # worse than best but still below floor → candidate, not a new best
    assert gates.gate_decision(35.0, 30.0, thr) == (True, False, False)
    # NaN → never a candidate
    assert gates.gate_decision(float("nan"), float("inf"), thr)[0] is False


def test_build_data_loader_exits_without_tokenizer(monkeypatch):
    """With no trained tokenizer, the loader must abort with guidance, not crash later."""
    set_active_model("cognition")
    from src.modalities.text import TextTokenizer
    from src.training.dataload import build_data_loader

    monkeypatch.setattr(TextTokenizer, "ready", property(lambda self: False), raising=False)
    with pytest.raises(SystemExit):
        build_data_loader(1, _TINY)
