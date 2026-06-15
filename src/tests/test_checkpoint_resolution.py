"""
Standard checkpoint resolution (src/training/checkpoint.py) — every model writes the same
best→final→latest layout, so EVERY consumer (chat, agent, camera, …) resolves a trained
model the same way. `resolve_stage_checkpoint` picks within one stage dir; `discover_checkpoint`
auto-finds the right stage dir when the caller doesn't know it; `trained_arch` recovers the
geometry the weights were trained at (so the net is rebuilt at the matching size).
"""

import json
from pathlib import Path

from src.training.checkpoint import (
    discover_checkpoint,
    resolve_stage_checkpoint,
    trained_arch,
)


def _touch(p: Path):
    p.write_bytes(b"")


def test_prefers_best_over_final_and_latest(tmp_path):
    _touch(tmp_path / "best.npz")
    _touch(tmp_path / "final.npz")
    _touch(tmp_path / "step_00008000.npz")
    (tmp_path / "best.json").write_text(
        json.dumps({"score": 14.2, "step": 7920, "tokens": 64_000_000})
    )
    (tmp_path / "latest.json").write_text(
        json.dumps({"checkpoint": str(tmp_path / "step_00008000.npz"), "step": 8000})
    )
    path, label, meta = resolve_stage_checkpoint(tmp_path)
    assert path == tmp_path / "best.npz"
    assert label == "best"
    assert meta["score"] == 14.2


def test_falls_back_to_final_when_no_best(tmp_path):
    _touch(tmp_path / "final.npz")
    (tmp_path / "stage_complete.json").write_text(
        json.dumps({"gate_score": 11.0, "step": 9000, "met_bar": True})
    )
    path, label, meta = resolve_stage_checkpoint(tmp_path)
    assert path == tmp_path / "final.npz"
    assert label == "final (graduated)"
    assert meta["met_bar"] is True


def test_falls_back_to_latest_in_progress(tmp_path):
    _touch(tmp_path / "step_00002000.npz")
    (tmp_path / "latest.json").write_text(
        json.dumps({"checkpoint": str(tmp_path / "step_00002000.npz"), "step": 2000})
    )
    path, label, _meta = resolve_stage_checkpoint(tmp_path)
    assert path == tmp_path / "step_00002000.npz"
    assert label == "latest (in-progress)"


def test_none_when_empty(tmp_path):
    path, label, meta = resolve_stage_checkpoint(tmp_path)
    assert path is None and label == "none" and meta is None


def test_latest_pointing_at_missing_file_is_ignored(tmp_path):
    (tmp_path / "latest.json").write_text(
        json.dumps({"checkpoint": str(tmp_path / "gone.npz"), "step": 1})
    )
    assert resolve_stage_checkpoint(tmp_path)[0] is None


# ── discover_checkpoint: find the right stage dir without being told it ──────────
def _model_tree(monkeypatch, tmp_path):
    """Point model_dist_root at a temp dist tree (resolved lazily inside discover)."""
    import src.config as C

    root = tmp_path / "amodel"
    monkeypatch.setattr(C, "model_dist_root", lambda model=None: root)
    return root / "checkpoints"


def test_discover_picks_most_recent_stage(monkeypatch, tmp_path):
    ckpts = _model_tree(monkeypatch, tmp_path)
    old = ckpts / "level0" / "stage1"
    new = ckpts / "level1" / "stage1"
    for d in (old, new):
        d.mkdir(parents=True)
        _touch(d / "final.npz")
    # Make level1 the most recently trained.
    import os

    os.utime(new, (10_000, 10_000))
    os.utime(old, (1_000, 1_000))
    path, _label, _meta = discover_checkpoint("amodel")
    assert path == new / "final.npz"


def test_discover_can_restrict_to_level_and_stage(monkeypatch, tmp_path):
    ckpts = _model_tree(monkeypatch, tmp_path)
    target = ckpts / "level0" / "stage1"
    target.mkdir(parents=True)
    _touch(target / "best.npz")
    path, label, _meta = discover_checkpoint("amodel", level=0, stage=1)
    assert path == target / "best.npz" and label == "best"


def test_discover_none_when_nothing_trained(monkeypatch, tmp_path):
    _model_tree(monkeypatch, tmp_path)  # tree doesn't exist
    assert discover_checkpoint("amodel") == (None, "none", None)


# ── trained_arch: recover the geometry the weights need ──────────────────────────
def test_trained_arch_reads_audit(tmp_path):
    (tmp_path / "audit.json").write_text(json.dumps({"model": {"d_model": 64, "n_layers": 3}}))
    arch = trained_arch(tmp_path / "final.npz")
    assert arch["d_model"] == 64 and arch["n_layers"] == 3


def test_trained_arch_empty_without_audit(tmp_path):
    assert trained_arch(tmp_path / "final.npz") == {}
