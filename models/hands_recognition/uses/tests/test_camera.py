"""hands_recognition camera use case (uses/camera/run_camera.py). The webcam loop is
driven with a FAKE cv2 (no hardware), so we cover predict → skeleton overlay → display
headlessly, plus the selftest and the missing-opencv path."""

import sys
import types

import numpy as np
import pytest

from models.hands_recognition.uses.camera import run_camera as RC


def test_load_net_random_and_missing_checkpoint(capsys, tmp_path):
    net = RC._load_net(None)
    assert net is not None and "random weights" in capsys.readouterr().out.lower()
    RC._load_net(str(tmp_path / "nope.npz"))  # nonexistent → warn, still builds
    assert "not found" in capsys.readouterr().out.lower()


def test_predict_returns_keypoints():
    net = RC._load_net(None)
    frame = np.zeros(RC.IMG_SIZE * RC.IMG_SIZE, dtype=np.float32)
    pts, z = RC._predict(net, frame)
    assert pts.shape == (RC.N_KEYPOINTS, 2)
    assert z is None  # the synthetic MLP predicts only 2D (no depth branch)


def test_predict_heatmap_returns_keypoints_and_depth():
    net = RC._load_net(None, {"arch": "heatmap", "img_size": 64, "in_channels": 3, "d_model": 32})
    img = np.zeros((3, 64, 64), dtype=np.float32)
    pts, z = RC._predict(net, img)
    assert pts.shape == (RC.N_KEYPOINTS, 2) and z.shape == (RC.N_KEYPOINTS,)
    assert pts.min() >= 0.0 and pts.max() <= 1.0  # soft-argmax localizes within the frame


def test_selftest_runs(capsys):
    net = RC._load_net(None)
    assert RC.selftest(net) == 0
    out = capsys.readouterr().out
    assert "skeleton" in out and "bones/phalanges" in out


class _FakeCv2:
    """A minimal cv2 stand-in: yields two frames then stops; records draw calls."""

    COLOR_BGR2GRAY = 7
    COLOR_BGR2RGB = 4
    FONT_HERSHEY_SIMPLEX = 0
    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_BUFFERSIZE = 38

    def __init__(self):
        self.lines = 0
        self.circles = 0
        self.shown = 0
        self.hud_texts = []
        self.set_props = {}
        self.wait_delays = []

    def VideoCapture(self, idx):
        cap = self

        class _Cap:
            def __init__(self):
                self._n = 0

            def isOpened(self):
                return True

            def set(self, prop, val):
                cap.set_props[prop] = val

            def read(self):
                self._n += 1
                if self._n > 2:
                    return False, None
                return True, np.zeros((48, 64, 3), dtype=np.uint8)

            def release(self):
                cap.released = True

        return _Cap()

    def cvtColor(self, frame, code):
        return frame if code == self.COLOR_BGR2RGB else frame[:, :, 0]  # RGB keeps 3ch

    def resize(self, img, size):
        shape = (size[1], size[0]) + ((img.shape[2],) if img.ndim == 3 else ())
        return np.zeros(shape, dtype=np.uint8)

    def line(self, *a, **k):
        self.lines += 1

    def circle(self, *a, **k):
        self.circles += 1

    def putText(self, frame, text, *a, **k):
        self.hud_texts.append(text)

    def imshow(self, *a, **k):
        self.shown += 1

    def waitKey(self, n):
        self.wait_delays.append(n)
        return ord("q")  # quit after the first shown frame

    def destroyAllWindows(self):
        pass


def test_run_camera_loop_with_fake_cv2(monkeypatch):
    fake = _FakeCv2()
    monkeypatch.setitem(sys.modules, "cv2", fake)
    net = RC._load_net(None)
    assert RC.run_camera(net, camera_index=0, target_fps=30) == 0
    assert fake.shown >= 1 and fake.lines >= 1 and fake.circles >= 1
    assert any("FPS" in t and "target 30" in t for t in fake.hud_texts)  # HUD shows FPS
    assert fake.set_props.get(fake.CAP_PROP_FPS) == 30  # requested rate from the device
    # Paced to the target by waiting only the budget LEFT after processing — so 1 ≤ wait ≤
    # the per-frame budget (never the full budget added on top, which halved the rate).
    assert fake.wait_delays and 1 <= fake.wait_delays[0] <= int(1000 / 30)


def test_run_camera_60fps_paces_loop(monkeypatch):
    fake = _FakeCv2()
    monkeypatch.setitem(sys.modules, "cv2", fake)
    net = RC._load_net(None)
    assert RC.run_camera(net, camera_index=0, target_fps=60) == 0
    assert fake.set_props.get(fake.CAP_PROP_FPS) == 60
    assert fake.wait_delays and 1 <= fake.wait_delays[0] <= int(1000 / 60)
    assert any("target 60" in t for t in fake.hud_texts)


def test_run_camera_without_opencv(monkeypatch, capsys):
    # Simulate opencv not installed → graceful message, return code 1.
    import builtins

    real_import = builtins.__import__

    def _no_cv2(name, *a, **k):
        if name == "cv2":
            raise ModuleNotFoundError("No module named 'cv2'")
        return real_import(name, *a, **k)

    monkeypatch.delitem(sys.modules, "cv2", raising=False)
    monkeypatch.setattr(builtins, "__import__", _no_cv2)
    net = RC._load_net(None)
    assert RC.run_camera(net, camera_index=0) == 1
    assert "opencv" in capsys.readouterr().out.lower()


def test_main_selftest(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_camera.py", "--selftest"])
    assert RC.main() == 0


def test_main_passes_fps_choice(monkeypatch):
    fake = _FakeCv2()
    monkeypatch.setitem(sys.modules, "cv2", fake)
    monkeypatch.setattr(sys, "argv", ["run_camera.py", "--fps", "60"])
    assert RC.main() == 0
    assert fake.set_props.get(fake.CAP_PROP_FPS) == 60


# ── architecture plumbing (the "trained but acts untrained" trap) ───────────────
# Standard checkpoint discovery + trained_arch live in the framework (covered by
# src/tests/test_checkpoint_resolution.py). Here we only check the camera rebuilds the net to
# MATCH the checkpoint's arch — if that's wrong the loaded weights shape-mismatch and stay
# random. `arch` is the dict the camera gets from `trained_arch`.
def test_load_net_defaults_to_synthetic_mlp():
    net = RC._load_net(None)  # no arch → synthetic MLP at the default width
    assert getattr(net.cfg, "arch", None) != "heatmap" and net.cfg.d_model == RC._DEFAULT_HIDDEN


def test_load_net_builds_mlp_at_trained_width():
    net = RC._load_net(None, {"d_model": 64})
    assert net.cfg.d_model == 64


def test_load_net_builds_heatmap_from_arch():
    arch = {"arch": "heatmap", "img_size": 64, "in_channels": 3, "d_model": 32, "heatmap_size": 16}
    net = RC._load_net(None, arch)
    assert net.cfg.arch == "heatmap" and net.cfg.img_size == 64 and net.cfg.heatmap_size == 16


def test_preprocess_shapes_match_arch():
    # Heatmap → [C,H,W] at img_size; MLP → flat [_IN] grayscale. Driven entirely by net.cfg.
    import numpy as _np

    fake = _FakeCv2()
    frame = _np.zeros((48, 64, 3), dtype=_np.uint8)
    fcn = RC._load_net(None, {"arch": "heatmap", "img_size": 64, "in_channels": 3, "d_model": 32})
    assert RC._preprocess(fcn, frame, fake).shape == (3, 64, 64)
    mlp = RC._load_net(None)
    assert RC._preprocess(mlp, frame, fake).shape == (RC.IMG_SIZE * RC.IMG_SIZE,)


def test_selftest_heatmap_path(capsys):
    net = RC._load_net(None, {"arch": "heatmap", "img_size": 64, "in_channels": 3, "d_model": 32})
    assert RC.selftest(net) == 0
    out = capsys.readouterr().out
    assert "heatmap FCN" in out and "depth" in out


def test_draw_skeleton_with_depth_colors_joints():
    # With a depth vector the overlay draws a dot per joint (coloured/sized by z).
    fake = _FakeCv2()
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    pts = np.full((RC.N_KEYPOINTS, 2), 0.5, dtype=np.float32)
    z = np.linspace(-1.0, 1.0, RC.N_KEYPOINTS).astype(np.float32)
    RC._draw_skeleton(fake, frame, pts, z)
    assert fake.circles == RC.N_KEYPOINTS and fake.lines == len(RC.HAND_CONNECTIONS)
