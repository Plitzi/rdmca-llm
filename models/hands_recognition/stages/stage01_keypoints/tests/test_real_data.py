"""Real-hand path: the FreiHAND heatmap loader, the heatmap FCN, soft-argmax, and the
ModelSpec wiring that selects them. Runs against a tiny FAKE FreiHAND tree (no download),
so CI exercises the plumbing (projection → [0,1] keypoints, localization, Gaussian target
heatmaps, 3D depth, FCN forward, soft-argmax localization, spec arch/loader choice)."""

import json

import numpy as np
import pytest

import src.backend as backend
from models.hands_recognition.data_freihand import FreiHandLoader
from models.hands_recognition.pose import (
    HandHeatmapNet,
    build_heatmap_net,
    build_spec,
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


def test_loader_returns_heatmaps_depth_and_keypoints(fake_freihand):
    loader = FreiHandLoader(
        fake_freihand, batch_size=4, img_size=64, in_channels=3, heatmap_size=16, split="train"
    )
    imgs, heatmaps, z, kpts = loader.next_batch()
    assert imgs.shape == (4, 3, 64, 64)
    assert heatmaps.shape == (4, 21, 16, 16)
    assert z.shape == (4, 21) and kpts.shape == (4, 21, 2)
    assert imgs.min() >= 0.0 and imgs.max() <= 1.0
    assert kpts.min() >= 0.0 and kpts.max() <= 1.0  # localized keypoints stay on-canvas
    assert heatmaps.max() <= 1.0 and heatmaps.max() > 0.5  # a Gaussian peak per keypoint
    assert np.allclose(z[:, 0], 0.0)  # depth is RELATIVE to the wrist (keypoint 0)


def test_loader_grayscale_channel(fake_freihand):
    loader = FreiHandLoader(
        fake_freihand, batch_size=2, img_size=32, in_channels=1, heatmap_size=8, split="train"
    )
    imgs, *_ = loader.next_batch()
    assert imgs.shape == (2, 1, 32, 32)


def test_localization_places_hand_off_centre(fake_freihand):
    # With localization the keypoint centroid varies sample-to-sample (random placement),
    # unlike the fills-the-frame path where it sits near the middle every time.
    loader = FreiHandLoader(
        fake_freihand, batch_size=8, img_size=64, heatmap_size=16, localize=True, seed=1
    )
    _imgs, _hm, _z, kpts = loader.next_batch()
    centroids = kpts.mean(axis=1)  # [N,2]
    assert centroids.std(axis=0).max() > 0.02  # placement actually moves the hand around


def test_train_val_split_is_disjoint_and_reproducible(fake_freihand):
    tr = FreiHandLoader(fake_freihand, batch_size=2, split="train")
    va = FreiHandLoader(fake_freihand, batch_size=2, split="val")
    assert set(tr._indices).isdisjoint(set(va._indices))
    assert len(va._indices) >= 1 and tr.epoch_tokens == len(tr._indices)
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
    net = build_heatmap_net({"img_size": 64, "in_channels": 3, "d_model": 32, "heatmap_size": 16})
    heatmaps, z = net(ops.array(np.zeros((2, 3, 64, 64), dtype=np.float32)))
    assert np.asarray(ops.to_numpy(heatmaps)).shape == (2, 21, 16, 16)
    assert np.asarray(ops.to_numpy(z)).shape == (2, 21)
    assert isinstance(net, HandHeatmapNet) and net.count_params() > 0


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
    assert mcfg.arch == "heatmap" and mcfg.img_size == 64 and mcfg.dims == 3
    loader = spec.build_loader(1, cfg)
    assert isinstance(loader, FreiHandLoader)
    # The objective consumes the 4-tuple batch (imgs, heatmaps, z, kpts) and returns a loss.
    loss = spec.objective(net, loader.next_batch())
    assert np.isfinite(float(backend.current().engine.item(loss)))
    # Honest gate: evaluate builds its OWN localized val split + soft-argmax mpjpe.
    score, _passed = spec.evaluate(net, 1, val_batches=None, cfg=cfg, log=lambda *_: None)
    assert np.isfinite(score)


def test_spec_defaults_to_synthetic_without_arch():
    cfg = {"model": {}, "training": {"batch_size": 4, "seed": 0}}
    spec = build_spec(cfg)
    loader = spec.build_loader(1, cfg)
    # No heatmap arch → the synthetic loader (the real-hand FCN is opt-in via config + data).
    assert type(loader).__name__ == "_SynthLoader"
