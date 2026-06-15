"""Real-hand path: the FreiHAND heatmap loader, the heatmap FCN, soft-argmax, and the
ModelSpec wiring that selects them. Runs against a tiny FAKE FreiHAND tree (no download),
so CI exercises the plumbing (projection → [0,1] keypoints, localization, Gaussian target
heatmaps, 3D depth, FCN forward, soft-argmax localization, spec arch/loader choice)."""

import json

import numpy as np
import pytest

import src.backend as backend
from models.hands_recognition.data_freihand import FreiHandLoader, finger_states
from models.hands_recognition.pose import (
    N_KEYPOINTS,
    HandHeatmapNet,
    build_heatmap_net,
    build_spec,
    predict_hands,
    soft_argmax,
)


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


def test_loader_returns_multihand_targets(fake_freihand):
    loader = FreiHandLoader(
        fake_freihand, batch_size=4, img_size=64, in_channels=3, heatmap_size=16,
        n_hands=2, split="train",
    )  # fmt: skip
    imgs, heatmaps, z, presence, kpts, handedness, finger = loader.next_batch()
    assert imgs.shape == (4, 3, 64, 64)
    assert heatmaps.shape == (4, 2 * 21, 16, 16)  # n_hands·21 heatmap channels
    assert z.shape == (4, 2 * 21) and presence.shape == (4, 2)
    assert kpts.shape == (4, 2, 21, 2) and handedness.shape == (4, 2) and finger.shape == (4, 2, 5)
    assert imgs.min() >= 0.0 and imgs.max() <= 1.0
    assert kpts.min() >= 0.0 and kpts.max() <= 1.0  # localized keypoints stay on-canvas
    assert set(np.unique(presence)).issubset({0.0, 1.0})  # presence is a per-slot flag
    assert set(np.unique(handedness)).issubset({0, 1})  # 0=right, 1=left (from mirror)
    assert np.allclose(z[:, 0], 0.0) and np.allclose(z[:, 21], 0.0)  # depth RELATIVE to each wrist


def test_loader_grayscale_channel(fake_freihand):
    loader = FreiHandLoader(
        fake_freihand, batch_size=2, img_size=32, in_channels=1, heatmap_size=8, split="train"
    )
    imgs, *_ = loader.next_batch()
    assert imgs.shape == (2, 1, 32, 32)


def test_positional_slot_assignment_left_first(fake_freihand):
    # When two hands are present, the LEFT-MOST (smaller mean x) is assigned to slot 0.
    loader = FreiHandLoader(
        fake_freihand, batch_size=32, img_size=64, heatmap_size=16, n_hands=2, seed=3
    )
    _imgs, _hm, _z, presence, kpts, *_ = loader.next_batch()
    both = presence.sum(axis=1) == 2  # samples with two hands
    assert both.any()
    mx = kpts[both].mean(axis=2)[..., 0]  # [M, 2] mean-x per slot
    assert (mx[:, 0] <= mx[:, 1] + 1e-6).all()  # slot 0 is the left-most hand


def test_train_val_split_is_disjoint_and_reproducible(fake_freihand):
    tr = FreiHandLoader(fake_freihand, batch_size=2, split="train")
    va = FreiHandLoader(fake_freihand, batch_size=2, split="val")
    assert set(tr._indices).isdisjoint(set(va._indices))
    assert len(va._indices) >= 1 and tr.epoch_tokens == len(tr._indices) * tr.img_size
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


def test_heatmap_net_forward_shapes():
    ops = backend.current().ops
    net = build_heatmap_net(
        {"img_size": 64, "in_channels": 3, "d_model": 32, "heatmap_size": 16, "n_hands": 2}
    )
    heatmaps, z, presence = net(ops.array(np.zeros((2, 3, 64, 64), dtype=np.float32)))
    assert np.asarray(ops.to_numpy(heatmaps)).shape == (2, 2 * 21, 16, 16)
    assert np.asarray(ops.to_numpy(z)).shape == (2, 2 * 21)
    assert np.asarray(ops.to_numpy(presence)).shape == (2, 2)  # per-slot presence logit
    assert isinstance(net, HandHeatmapNet) and net.count_params() > 0


def test_predict_hands_decodes_per_slot():
    net = build_heatmap_net(
        {"img_size": 64, "in_channels": 3, "d_model": 32, "heatmap_size": 16, "n_hands": 2}
    )
    kpts, z, presence = predict_hands(net, np.zeros((3, 3, 64, 64), dtype=np.float32))
    assert kpts.shape == (3, 2, 21, 2) and z.shape == (3, 2, 21) and presence.shape == (3, 2)
    assert kpts.min() >= 0.0 and kpts.max() <= 1.0  # soft-argmax localizes each slot
    assert presence.min() >= 0.0 and presence.max() <= 1.0  # sigmoid probabilities


def test_finger_states_extended_vs_curled():
    # A flat hand (tips far from the wrist) → all extended; a fist (tips near wrist) → all curled.
    xyz = np.zeros((21, 3), dtype=np.float32)
    from models.hands_recognition.data_freihand import _FINGER_PIP_TIP

    for pip, tip in _FINGER_PIP_TIP:  # tips farther than mid joints → extended
        xyz[pip] = [0.0, 0.5, 0.0]
        xyz[tip] = [0.0, 1.0, 0.0]
    assert finger_states(xyz).tolist() == [1.0] * 5
    for pip, tip in _FINGER_PIP_TIP:  # tips closer than mid joints → curled
        xyz[pip] = [0.0, 1.0, 0.0]
        xyz[tip] = [0.0, 0.3, 0.0]
    assert finger_states(xyz).tolist() == [0.0] * 5


def test_heatmap_size_must_be_quarter_of_img_size():
    with pytest.raises(ValueError, match="heatmap_size"):
        build_heatmap_net({"img_size": 64, "heatmap_size": 8})  # 8 != 64//4


def test_soft_argmax_locates_the_peak():
    # A single bright pixel at (row=2, col=5) on an 8×8 grid → soft-argmax ≈ (5/7, 2/7).
    hm = np.zeros((1, 21, 8, 8), dtype=np.float32)
    hm[:, :, 2, 5] = 10.0
    coords = soft_argmax(hm)
    assert coords.shape == (1, 21, 2)
    assert np.allclose(coords[0, 0], [5 / 7, 2 / 7], atol=0.05)
    assert coords.min() >= 0.0 and coords.max() <= 1.0


def test_spec_selects_heatmap_and_real_loader(fake_freihand):
    cfg = {
        "model": {"arch": "heatmap", "img_size": 64, "in_channels": 3, "d_model": 32,
                  "heatmap_size": 16, "dims": 3},
        "dataset": {"root": fake_freihand, "localize": True},
        "training": {"batch_size": 2, "seed": 0},
        "gate": {"max_mpjpe": 5.0},
    }  # fmt: skip
    spec = build_spec(cfg)
    net, mcfg, _adapter, _prec, _seed = spec.build_model(1, cfg, None)
    assert mcfg.arch == "heatmap" and mcfg.img_size == 64 and mcfg.dims == 3 and mcfg.n_hands == 2
    assert net._active_stage == 1  # build_model tags the active stage for objective dispatch
    loader = spec.build_loader(1, cfg)
    assert isinstance(loader, FreiHandLoader)
    # Stage-1 objective consumes the multi-hand batch (heatmaps + depth + presence) → a loss.
    loss = spec.objective(net, loader.next_batch())
    assert np.isfinite(float(backend.current().engine.item(loss)))
    # Honest gate: evaluate builds its OWN localized val split + soft-argmax mpjpe over slots.
    score, _passed = spec.evaluate(net, 1, val_batches=None, cfg=cfg, log=lambda *_: None)
    assert np.isfinite(score)


def test_spec_defaults_to_synthetic_without_arch():
    cfg = {"model": {}, "training": {"batch_size": 4, "seed": 0}}
    spec = build_spec(cfg)
    loader = spec.build_loader(1, cfg)
    # No heatmap arch → the synthetic loader (the real-hand FCN is opt-in via config + data).
    assert type(loader).__name__ == "_SynthLoader"
