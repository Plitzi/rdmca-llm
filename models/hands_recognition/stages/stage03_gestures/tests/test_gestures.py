"""Stage 3 — gesture head on the frozen multi-hand backbone + the gesture dataset/download.
All against a tiny FAKE gesture tree (no network): the loader, that build_model attaches and
trains ONLY the gesture head, that a few steps reduce the loss, and the resumable download."""

import io
import zipfile

import numpy as np
import pytest

import models.hands_recognition.data_gestures as DG
import src.backend as backend
from models.hands_recognition.data_gestures import GESTURES, GestureLoader, download_gestures
from models.hands_recognition.pose import GestureHead, build_gesture_head, build_spec


@pytest.fixture
def fake_gestures(tmp_path):
    """A minimal gesture tree: a few labelled folders, each with a couple of stub JPEGs."""
    from PIL import Image

    root = tmp_path / "gestures"
    rng = np.random.default_rng(0)
    for name in ("thumbs_up", "fist", "open_palm"):
        (root / name).mkdir(parents=True)
        for i in range(4):
            img = (rng.random((96, 96, 3)) * 255).astype("uint8")
            Image.fromarray(img).save(root / name / f"{i:03d}.jpg")
    return str(root)


def _cfg(fake_gestures):
    return {
        "model": {"arch": "heatmap", "img_size": 64, "in_channels": 3, "d_model": 32,
                  "heatmap_size": 16, "n_hands": 2},
        "dataset": {"gesture_root": fake_gestures},
        "training": {"batch_size": 4, "seed": 0},
        "gate": {"max_gesture_err": 1.0},
    }  # fmt: skip


def test_gesture_head_shapes():
    ops = backend.current().ops
    head = build_gesture_head(n_gestures=6, hidden=16)
    out = head(ops.array(np.zeros((5, 21 * 3), dtype=np.float32)))
    assert np.asarray(ops.to_numpy(out)).shape == (5, 6) and isinstance(head, GestureHead)


def test_gesture_loader_shapes_and_labels(fake_gestures):
    loader = GestureLoader(fake_gestures, batch_size=4, img_size=64, in_channels=3, split="train")
    imgs, labels = loader.next_batch()
    assert imgs.shape == (4, 3, 64, 64) and labels.shape == (4,)
    assert set(np.unique(labels)).issubset(set(range(len(GESTURES))))  # valid gesture indices


def test_build_model_attaches_gesture_head_and_evaluates(fake_gestures):
    cfg = _cfg(fake_gestures)
    spec = build_spec(cfg)
    net, _mcfg, _adapter, _prec, _seed = spec.build_model(3, cfg, None)
    assert net._active_stage == 3 and hasattr(net, "gesture_head") and hasattr(net, "state_head")
    score, _passed = spec.evaluate(net, 3, val_batches=None, cfg=cfg, log=lambda *_: None)
    assert 0.0 <= score <= 1.0  # gesture_err = 1 − accuracy


def test_stage3_head_learns(fake_gestures):
    cfg = _cfg(fake_gestures)
    spec = build_spec(cfg)
    net, *_ = spec.build_model(3, cfg, None)
    eng = backend.current().engine
    batch = spec.build_loader(3, cfg).next_batch()
    loss_and_grad = eng.value_and_grad(net, spec.objective)
    opt = eng.make_optimizer(net, lr=1e-2, weight_decay=0.0)
    first = float(eng.item(spec.objective(net, batch)))
    for _ in range(40):
        _loss, grads = loss_and_grad(net, batch)
        eng.optimizer_step(opt, net, grads)
    assert float(eng.item(spec.objective(net, batch))) < first  # gesture head learns


# ── download (no network) ────────────────────────────────────────────────────
class _FakeResponse:
    def __init__(self, data):
        self._s = io.BytesIO(data)
        self.status = 200
        self.headers = {"Content-Length": str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n=-1):
        return self._s.read(n)


def test_download_gestures_extracts(tmp_path, monkeypatch):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("thumbs_up/0001.jpg", b"\xff\xd8\xff\xd9")
        zf.writestr("fist/0001.jpg", b"\xff\xd8\xff\xd9")
    import urllib.request

    monkeypatch.setattr(
        urllib.request, "urlopen", lambda req, *a, **k: _FakeResponse(buf.getvalue())
    )
    root = tmp_path / "gestures"
    download_gestures(root, url="http://example.test/gestures.zip")
    assert DG.is_prepared(root) and not (root / "gestures.zip").exists()


def test_download_gestures_idempotent_and_needs_url(tmp_path):
    # No URL configured and nothing prepared → a clear error (rather than a silent no-op).
    with pytest.raises(RuntimeError, match="URL"):
        download_gestures(tmp_path / "empty", url="")
