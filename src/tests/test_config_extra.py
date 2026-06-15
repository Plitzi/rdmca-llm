"""Config helpers beyond inheritance (src/config.py): level discovery, model/backend/
precision selection, language list, tokenizer-info readers."""

import pytest

import src.config as C


def test_available_levels_and_paths():
    levels = C.available_levels()
    assert levels == sorted(levels) and 0 in levels
    assert C.level_config_path(1).endswith("level1.yaml")


def test_resolve_config_path_level_wins_and_validates():
    assert C.resolve_config_path(level=1).endswith("level1.yaml")
    assert C.resolve_config_path(config="custom.yaml") == "custom.yaml"
    with pytest.raises(ValueError):
        C.resolve_config_path(level=999)


def test_get_level():
    assert C.get_level({"level": 3}) == 3
    assert C.get_level({}) is None


def test_select_model_override_and_default():
    assert C.select_model({"model_name": "cognition"}) == "cognition"
    assert (
        C.select_model({"model_name": "cognition"}, override="hands_recognition")
        == "hands_recognition"
    )
    C.select_model({"model_name": "cognition"})  # reset active model for other tests


def test_get_languages_default_and_explicit():
    assert C.get_languages({}) == ["en"]
    assert C.get_languages({"model": {"languages": ["en", "es"]}}) == ["en", "es"]


def test_get_backend_and_precision():
    assert C.get_backend({}) == "mlx"
    assert C.get_backend({"backend": "MLX"}) == "mlx"
    assert C.get_precision({}) == "bf16"
    assert C.get_precision({"training": {"precision": "fp32"}}) == "fp32"
    with pytest.raises(ValueError):
        C.get_precision({"training": {"precision": "fp9"}})


def test_require_backend_unknown_raises():
    with pytest.raises(ValueError):
        C.require_backend({"backend": "nonsense"})


def test_load_tokenizer_info_and_unified_vocab(tmp_path):
    assert C.load_tokenizer_info(str(tmp_path / "missing.json")) is None
    p = tmp_path / "info.json"
    p.write_text('{"vocab_size": 20480, "text_vocab_size": 8192}')
    info = C.load_tokenizer_info(str(p))
    assert info["vocab_size"] == 20480
    assert C.unified_vocab_size(info, 999) == 20480
    assert C.unified_vocab_size(None, 999) == 999
