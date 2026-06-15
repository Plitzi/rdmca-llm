"""The FreiHAND download/prepare step exercised with NO network: a tiny FreiHAND zip is
built in memory and served through a fake `urllib.request.urlopen`, so CI checks the real
plumbing (stream → extract → flatten → idempotent skip) that `rdmca prepare` runs via the
model's `prepare_stage` hook — without fetching the multi-GB dataset."""

import io
import json
import zipfile
from pathlib import Path

import models.hands_recognition.data_freihand as freihand
from models.hands_recognition import prepare_stage


def _make_zip(*, nested: bool) -> bytes:
    """A minimal FreiHAND_pub_v2 layout zipped in memory. When `nested`, everything is
    under a top FreiHAND_pub_v2/ dir (some mirrors do this) so we test the flatten path."""
    prefix = "FreiHAND_pub_v2/" if nested else ""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"{prefix}training_xyz.json", json.dumps([[[0.0, 0.0, 0.5]] * 21]))
        zf.writestr(f"{prefix}training_K.json", json.dumps([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))
        zf.writestr(f"{prefix}training/rgb/00000000.jpg", b"\xff\xd8\xff\xd9")  # stub jpeg
    return buf.getvalue()


class _FakeResponse:
    """Stands in for the urlopen result: a context manager streaming `data` in chunks."""

    def __init__(self, data: bytes):
        self._stream = io.BytesIO(data)
        self.status = 200
        self.headers = {"Content-Length": str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n: int = -1) -> bytes:
        return self._stream.read(n)


def _serve(monkeypatch, zip_bytes: bytes):
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, *a, **k: _FakeResponse(zip_bytes))


def test_download_extracts_flat_layout(tmp_path, monkeypatch):
    _serve(monkeypatch, _make_zip(nested=False))
    root = tmp_path / "freihand"
    freihand.download_freihand(root)
    assert freihand.is_prepared(root)
    assert not (root / "FreiHAND_pub_v2.zip").exists()  # zip removed after extraction


def test_download_flattens_nested_top_dir(tmp_path, monkeypatch):
    _serve(monkeypatch, _make_zip(nested=True))
    root = tmp_path / "freihand"
    freihand.download_freihand(root)
    assert freihand.is_prepared(root)
    assert (root / "training_xyz.json").exists()  # lifted out of FreiHAND_pub_v2/


def test_download_is_idempotent(tmp_path, monkeypatch):
    calls = {"n": 0}

    def _counting_urlopen(req, *a, **k):
        calls["n"] += 1
        return _FakeResponse(_make_zip(nested=False))

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", _counting_urlopen)
    root = tmp_path / "freihand"
    freihand.download_freihand(root)
    freihand.download_freihand(root)  # second call must skip (already prepared)
    assert calls["n"] == 1


def test_prepare_stage_hook_downloads_when_dataset_root(tmp_path, monkeypatch):
    _serve(monkeypatch, _make_zip(nested=False))
    root = tmp_path / "freihand"
    prepare_stage(1, {"dataset": {"root": str(root)}}, langs=["en"], limit_mb=None)
    assert freihand.is_prepared(root)


def test_prepare_stage_hook_noop_without_dataset_root(tmp_path):
    # No dataset.root → synthetic data, nothing to download (must not raise / fetch).
    prepare_stage(1, {"model": {}}, langs=["en"], limit_mb=None)
    assert not any(Path(tmp_path).iterdir())
