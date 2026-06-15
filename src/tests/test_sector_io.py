"""Behavioral LoRA sectors I/O (src/model/sector_io.py): id mapping, paths, attach for
training (freeze core, sector trainable), save/load, and inference attachment."""

import src.backend as backend
from src.model import sector_io as S


def _tiny_model():
    from src.model.config import ModelConfig
    from src.model.transformer import RDMCAFoundational

    cfg = ModelConfig(
        d_model=32, n_layers=2, n_heads=2, n_kv_heads=1, ffn_dim=64, context_len=32,
        vocab_size=64, mrl_dims=[16, 32], dropout=0.0,
    )  # fmt: skip
    return RDMCAFoundational(cfg)


def test_sector_id_and_paths(tmp_path):
    assert S.sector_id_for_stage(8) == 108
    assert S.sectors_dir(tmp_path).name == "sectors"
    assert S.sector_path(tmp_path, 8).name == "sector_stage8.npz"
    assert S.frozen_core_path(tmp_path).parts[-2:] == ("foundational", "theta_f_frozen.npz")


def test_attach_for_training_and_save(tmp_path):
    model = _tiny_model()
    sid, adapter = S.attach_for_training(model, stage=8)
    assert sid == 108 and adapter is not None
    p = S.save_sector(adapter, tmp_path, 8)
    assert p.exists()
    assert S.trained_sector_stages(tmp_path) == [8]
    assert S.trained_sector_stages(tmp_path, up_to=7) == []  # filter by ≤ up_to


def test_trained_sector_stages_empty(tmp_path):
    assert S.trained_sector_stages(tmp_path) == []  # no sectors dir


def test_load_for_inference_needs_core_and_sectors(tmp_path):
    model = _tiny_model()
    # No frozen core yet → None (caller falls back to the plain checkpoint).
    assert S.load_for_inference(model, tmp_path, stage=8) is None
    # Save a frozen core + a sector, then it attaches.
    core = S.frozen_core_path(tmp_path)
    core.parent.mkdir(parents=True, exist_ok=True)
    backend.current().engine.save_weights(model, str(core))
    _sid, adapter = S.attach_for_training(model, stage=8)
    S.save_sector(adapter, tmp_path, 8)
    label = S.load_for_inference(_tiny_model(), tmp_path, stage=8)
    assert label and "sectors [8]" in label
