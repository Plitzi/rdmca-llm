"""Regression for the tokenizer CLI's data-dir resolution: it must resolve the stage-1
corpus through the registry's per-model layout (models/<model>/data/<stage>/levelN), NOT
a hardcoded `data/levelN/stage1`, and point users at `rdmca prepare` when it's missing.
Guards the stale-path bug fixed after the data relocation."""

import importlib.util
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]


def _load_tokenizer_cli():
    spec = importlib.util.spec_from_file_location(
        "train_tokenizer_cli", str(_REPO / "scripts" / "train_tokenizer.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_missing_data_points_at_per_model_path_and_rdmca(monkeypatch, capsys):
    cli = _load_tokenizer_cli()
    # hands_recognition has no prepared text corpus (it's a vision model) → the data check
    # must fail with the PER-MODEL path and the rdmca-CLI suggestion (never the old
    # hardcoded path/command). Resolved via the model's own per-model level ladder (--level).
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["train_tokenizer", "--level", "1", "--model", "hands_recognition"],
    )
    with pytest.raises(SystemExit):
        cli.main()
    out = capsys.readouterr().out
    assert "hands_recognition" in out and "data" in out  # resolved per-model path
    assert "rdmca" in out and "prepare" in out  # current CLI suggestion
    assert "scripts/prepare_data.py" not in out  # not the stale command
    assert "data/level1/stage1" not in out  # not the stale hardcoded path
