"""Real-hand path: the FreiHAND loader, the CNN, and the ModelSpec wiring that selects
them. Runs against a tiny FAKE FreiHAND tree (no download), so CI exercises the plumbing
(projection → [0,1] keypoints, disjoint train/val, CNN forward, spec arch/loader choice)."""

import json

import numpy as np
import pytest

import src.backend as backend
from models.hands_recognition.data_freihand import FreiHandLoader
from models.hands_recognition.pose import HandPoseCNN, build_pose_cnn, build_spec


@pytest.fixture
def fake_freihand(tmp_path):
    """A minimal FreiHAND layout: M annotations + JPGs, hand-centred so 2D lands in [0,1]."""
    from PIL import Image

    root = tmp_path / "freihand"
    (root / "training" / "rgb").mkdir(parents=True)
    m = 8
    rng = np.random.default_rng(0)
    xyz = [(rng.random((21, 3)) * 0.1 + [0.0, 0.0, 0.5]).tolist() for _ in range(m)]
    k = [[[200.0, 0.0, 112.0], [0.0, 200.0, 112.0], [0.0, 0.0, 1.0]] for _ in range(m)]
    (root / "training_xyz.json").write_text(json.dumps(xyz))
    (root / "training_K.json").write_text(json.dumps(k))
    for i in range(m):
        img = (rng.random((224, 224, 3)) * 255).astype("uint8")
        Image.fromarray(img).save(root / "training" / "rgb" / f"{i:08d}.jpg")
    return str(root)


def test_loader_shapes_and_normalized_keypoints(fake_freihand):
    loader = FreiHandLoader(fake_freihand, batch_size=4, img_size=64, in_channels=3, split="train")
    imgs, kpts = loader.next_batch()
    assert imgs.shape == (4, 3, 64, 64)
    assert kpts.shape == (4, 42)
    assert imgs.min() >= 0.0 and imgs.max() <= 1.0
    assert kpts.min() >= 0.0 and kpts.max() <= 1.0  # projected + normalized to the image


def test_loader_grayscale_channel(fake_freihand):
    loader = FreiHandLoader(fake_freihand, batch_size=2, img_size=32, in_channels=1, split="train")
    imgs, _ = loader.next_batch()
    assert imgs.shape == (2, 1, 32, 32)


def test_train_val_split_is_disjoint_and_reproducible(fake_freihand):
    tr = FreiHandLoader(fake_freihand, batch_size=2, split="train")
    va = FreiHandLoader(fake_freihand, batch_size=2, split="val")
    assert set(tr._indices).isdisjoint(set(va._indices))
    assert len(va._indices) >= 1 and tr.epoch_tokens == len(tr._indices)
    # Same split on a second construction (fixed split seed).
    assert set(va._indices) == set(
        FreiHandLoader(fake_freihand, batch_size=2, split="val")._indices
    )


def test_loader_telemetry_matches_synth_interface(fake_freihand):
    loader = FreiHandLoader(fake_freihand, batch_size=2, split="train")
    assert loader.replay_fraction == 0.0 and loader.last_was_replay is False
    assert loader.load_skip_index("x") is False  # no-op, like the synthetic loader
    loader.save_skip_index("x")  # no-op


def test_missing_dataset_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        FreiHandLoader(str(tmp_path / "nope"), batch_size=2)


def test_cnn_forward_shape():
    ops = backend.current().ops
    net = build_pose_cnn({"img_size": 64, "in_channels": 3, "d_model": 32})
    out = net(ops.array(np.zeros((2, 3, 64, 64), dtype=np.float32)))
    assert np.array(ops.to_numpy(out)).shape == (2, 42)
    assert isinstance(net, HandPoseCNN) and net.count_params() > 0


def test_spec_selects_cnn_and_real_loader(fake_freihand):
    cfg = {
        "model": {"arch": "cnn", "img_size": 64, "in_channels": 3, "d_model": 32},
        "dataset": {"root": fake_freihand},
        "training": {"batch_size": 2, "seed": 0},
        "gate": {"max_mpjpe": 0.5},
    }
    spec = build_spec(cfg)
    net, mcfg, _adapter, _prec, _seed = spec.build_model(1, cfg, None)
    assert mcfg.arch == "cnn" and mcfg.img_size == 64
    loader = spec.build_loader(1, cfg)
    assert isinstance(loader, FreiHandLoader)
    # Honest gate: evaluate builds its OWN val split from the dataset and returns a score.
    score, _passed = spec.evaluate(net, 1, val_batches=None, cfg=cfg, log=lambda *_: None)
    assert np.isfinite(score)


def test_spec_defaults_to_synthetic_without_dataset():
    cfg = {"model": {}, "training": {"batch_size": 4, "seed": 0}}
    spec = build_spec(cfg)
    loader = spec.build_loader(1, cfg)
    # No dataset → the synthetic loader (a real-hand CNN is opt-in via config + data).
    assert type(loader).__name__ == "_SynthLoader"
