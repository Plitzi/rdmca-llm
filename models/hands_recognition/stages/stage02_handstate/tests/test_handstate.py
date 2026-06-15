"""Stage 2 — handedness + finger-state head on the frozen multi-hand backbone. Runs against
a tiny FAKE FreiHAND tree (no download): the head's shapes, that build_model attaches it and
trains ONLY the head (backbone frozen), and that a few steps reduce the stage-2 loss."""

import json

import numpy as np
import pytest

import src.backend as backend
from models.hands_recognition.pose import HandStateHead, build_spec, build_state_head


@pytest.fixture
def fake_freihand(tmp_path):
    from PIL import Image

    root = tmp_path / "freihand"
    (root / "training" / "rgb").mkdir(parents=True)
    rng = np.random.default_rng(0)
    xyz = [(rng.random((21, 3)) * 0.1 + [0.0, 0.0, 0.5]).tolist() for _ in range(8)]
    k = [[[200.0, 0.0, 112.0], [0.0, 200.0, 112.0], [0.0, 0.0, 1.0]] for _ in range(8)]
    (root / "training_xyz.json").write_text(json.dumps(xyz))
    (root / "training_K.json").write_text(json.dumps(k))
    for i in range(8):
        img = (rng.random((224, 224, 3)) * 255).astype("uint8")
        Image.fromarray(img).save(root / "training" / "rgb" / f"{i:08d}.jpg")
    return str(root)


def _cfg(fake_freihand):
    return {
        "model": {"arch": "heatmap", "img_size": 64, "in_channels": 3, "d_model": 32,
                  "heatmap_size": 16, "dims": 3, "n_hands": 2},
        "dataset": {"root": fake_freihand, "localize": True},
        "training": {"batch_size": 4, "seed": 0},
        "gate": {"max_handstate_err": 1.0},
    }  # fmt: skip


def test_state_head_shapes():
    ops = backend.current().ops
    head = build_state_head(hidden=16)
    handed, finger = head(ops.array(np.zeros((5, 21 * 3), dtype=np.float32)))
    assert np.asarray(ops.to_numpy(handed)).shape == (5, 2)  # right/left
    assert np.asarray(ops.to_numpy(finger)).shape == (5, 5)  # per-finger extended/curled
    assert isinstance(head, HandStateHead)


def test_build_model_attaches_head_and_evaluates(fake_freihand):
    cfg = _cfg(fake_freihand)
    spec = build_spec(cfg)
    net, _mcfg, _adapter, _prec, _seed = spec.build_model(2, cfg, None)
    assert net._active_stage == 2 and hasattr(net, "state_head")
    score, _passed = spec.evaluate(net, 2, val_batches=None, cfg=cfg, log=lambda *_: None)
    assert 0.0 <= score <= 1.0  # handstate_err = 1 − accuracy


def test_stage2_head_learns(fake_freihand):
    cfg = _cfg(fake_freihand)
    spec = build_spec(cfg)
    net, _mcfg, _adapter, _prec, _seed = spec.build_model(2, cfg, None)
    eng = backend.current().engine
    loader = spec.build_loader(2, cfg)
    batch = loader.next_batch()  # train on a fixed batch so the loss must drop
    loss_and_grad = eng.value_and_grad(net, spec.objective)
    opt = eng.make_optimizer(net, lr=1e-2, weight_decay=0.0)
    first = float(eng.item(spec.objective(net, batch)))
    for _ in range(40):
        _loss, grads = loss_and_grad(net, batch)
        eng.optimizer_step(opt, net, grads)
    last = float(eng.item(spec.objective(net, batch)))
    assert last < first  # the head genuinely learns on the frozen backbone
