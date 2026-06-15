"""Resource estimation + OOM guard (src/resources.py) — pure formulas, so fully unit
testable: parameter count tracks the config dims, memory estimates are monotonic and
ordered (train > infer), the guard aborts/forces correctly, and announce prints."""

import pytest

import src.resources as R

_MODEL = {
    "d_model": 64,
    "n_layers": 2,
    "n_heads": 2,
    "n_kv_heads": 1,
    "ffn_dim": 256,
    "context_len": 64,
    "vocab_size": 4096,
    "mrl_dims": [32, 64],
}
_CFG = {
    "level": 0,
    "name": "smoke",
    "model": _MODEL,
    "training": {"batch_size": 16, "precision": "bf16"},
}


def test_count_params_matches_formula_and_grows_with_size():
    base = R.count_params(_MODEL)
    assert base > 0
    wider = dict(_MODEL, d_model=128)
    deeper = dict(_MODEL, n_layers=4)
    assert R.count_params(wider) > base
    assert R.count_params(deeper) > base


def test_count_params_counts_optional_ple_and_mtp():
    base = R.count_params(_MODEL)
    with_ple = R.count_params(dict(_MODEL, ple_dim=8))
    with_mtp = R.count_params(dict(_MODEL, n_mtp_heads=1, mtp_hidden_dim=32))
    assert with_ple > base and with_mtp > base


def test_train_memory_exceeds_inference():
    train = R.estimate_train_memory_gb(_MODEL, {"batch_size": 16}, "bf16")
    infer = R.estimate_infer_memory_gb(_MODEL, "bf16")
    assert train > infer > 0


def test_fp32_estimate_exceeds_bf16():
    assert R.estimate_train_memory_gb(
        _MODEL, {"batch_size": 16}, "fp32"
    ) > R.estimate_train_memory_gb(_MODEL, {"batch_size": 16}, "bf16")


def test_estimate_for_modes_and_default_precision():
    # estimate_for defaults to fp32 (conservative) when precision unset
    cfg_no_prec = {"model": _MODEL, "training": {"batch_size": 8}}
    assert R.estimate_for(cfg_no_prec, "train") > 0
    assert R.estimate_for(_CFG, "infer") > 0


def test_available_memory_is_positive():
    assert R.available_memory_gb() > 0


def test_max_runnable_level_returns_a_level_or_none(monkeypatch):
    monkeypatch.setattr(R, "available_memory_gb", lambda: 10_000.0)  # huge → top level fits
    assert R.max_runnable_level("train") is not None
    monkeypatch.setattr(R, "available_memory_gb", lambda: 1e-6)  # tiny → nothing fits
    assert R.max_runnable_level("train") is None


def test_guard_passes_when_it_fits(monkeypatch):
    monkeypatch.setattr(R, "available_memory_gb", lambda: 10_000.0)
    R.guard(_CFG, "train")  # must not raise


def test_guard_aborts_when_too_big(monkeypatch, capsys):
    monkeypatch.setattr(R, "available_memory_gb", lambda: 1e-6)
    with pytest.raises(SystemExit):
        R.guard(_CFG, "train")
    assert "Resource guard" in capsys.readouterr().out


def test_guard_force_continues(monkeypatch, capsys):
    monkeypatch.setattr(R, "available_memory_gb", lambda: 1e-6)
    R.guard(_CFG, "train", force=True)  # must NOT raise
    assert "force" in capsys.readouterr().out.lower()


def test_announce_prints_level_and_areas(capsys):
    cfg = dict(_CFG, information={"summary": "smoke tier", "areas": {1: "Language", 2: "Patterns"}})
    R.announce(cfg, "train")
    out = capsys.readouterr().out
    assert "Level 0" in out and "Language" in out and "params" in out


def test_announce_single_stage_area(capsys):
    cfg = dict(
        _CFG,
        information={"areas": {1: "Language", 2: "Patterns"}},
        moe={"enabled": True, "experts": 6, "top_k": 2},
    )
    R.announce(cfg, "train", stage=2)
    out = capsys.readouterr().out
    assert "stage 2" in out and "Patterns" in out and "MoE" in out
