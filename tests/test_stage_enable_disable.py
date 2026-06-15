"""
Enable/disable tests — a stage can be switched off two ways (plugin flag or a
per-level `curriculum.stageN.enabled: false` override), and both must drop it from
the curriculum the trainer and data pipeline iterate.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import plugins as stages
from src.training.curriculum import stage_enabled


def test_all_stages_enabled_by_default():
    assert all(p.enabled for p in stages.all_stages())
    assert {p.number for p in stages.enabled_stages()} == {p.number for p in stages.all_stages()}


def test_stage_enabled_true_without_config():
    assert stage_enabled(3) is True


def test_config_override_disables_a_stage():
    cfg = {"curriculum": {"stage3": {"name": "s3", "enabled": False}}}
    assert stage_enabled(3, cfg) is False
    # other stages unaffected
    assert stage_enabled(1, cfg) is True


def test_disabled_stage_dropped_from_prepare_iteration(tmp_path, monkeypatch, capsys):
    # prepare_stage_for_level must skip a disabled stage without preparing data.
    import scripts.prepare_data as pd

    cfg = {
        "level": 1,
        "curriculum": {"stage3": {"name": "s3", "entry_level": 1, "enabled": False, "data": {}}},
    }
    pd.prepare_stage_for_level(1, 3, cfg, langs=["en"])
    out = capsys.readouterr().out
    assert "disabled" in out.lower() and "skipping" in out.lower()
