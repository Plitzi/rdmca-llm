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
    pts = RC._predict(net, frame)
    assert pts.shape == (RC.N_KEYPOINTS, 2)


def test_selftest_runs(capsys):
    net = RC._load_net(None)
    assert RC.selftest(net) == 0
    out = capsys.readouterr().out
    assert "skeleton" in out and "bones/phalanges" in out


class _FakeCv2:
    """A minimal cv2 stand-in: yields two frames then stops; records draw calls."""

    COLOR_BGR2GRAY = 7
    FONT_HERSHEY_SIMPLEX = 0
    CAP_PROP_FPS = 5

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
        return frame[:, :, 0]

    def resize(self, img, size):
        return np.zeros(size, dtype=np.uint8)

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
    assert fake.wait_delays and fake.wait_delays[0] == int(1000 / 30)  # paced to target


def test_run_camera_60fps_paces_loop(monkeypatch):
    fake = _FakeCv2()
    monkeypatch.setitem(sys.modules, "cv2", fake)
    net = RC._load_net(None)
    assert RC.run_camera(net, camera_index=0, target_fps=60) == 0
    assert fake.set_props.get(fake.CAP_PROP_FPS) == 60
    assert fake.wait_delays[0] == int(1000 / 60)
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
