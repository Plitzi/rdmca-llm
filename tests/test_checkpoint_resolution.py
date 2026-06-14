"""
The use cases (chat + agent share run_chat.load_model) must ALWAYS load the BEST
checkpoint for a stage — the lowest-val-perplexity model the gate ratchets toward — and
report which one + its tracked quality, so what's running is always known.

Priority: best.npz (running best) > final.npz (graduated) > latest.json (in-progress).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from uses.chat.run_chat import describe_checkpoint_meta, resolve_stage_checkpoint


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
        json.dumps(
            {
                "checkpoint": str(tmp_path / "step_00002000.npz"),
                "step": 2000,
                "tokens_seen": 16_000_000,
            }
        )
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


def test_describe_checkpoint_meta_formats_quality():
    desc = describe_checkpoint_meta(
        {"score": 14.2, "step": 7920, "tokens": 64_000_000, "met_bar": True}
    )
    assert "val ppl 14.20" in desc and "step 7,920" in desc
    assert "64.0M tok" in desc and "met_bar=True" in desc
    assert describe_checkpoint_meta(None) == ""
    assert describe_checkpoint_meta({}) == ""
