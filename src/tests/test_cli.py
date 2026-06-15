"""Tests for the unified `rdmca` developer CLI (scripts/rdmca.py).

The CLI is pure routing — it must: parse a `--model` out of the forwarded args, build
the right command for each kind (script / run / module / builtin), forward `--model` only
to model-aware scripts, and answer `info` from the registry. We drive it without spawning
real subprocesses by stubbing `subprocess.call`.
"""

import importlib.util
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def _load_cli():
    spec = importlib.util.spec_from_file_location("rdmca_cli", str(_REPO / "scripts" / "rdmca.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CLI = _load_cli()


def test_extract_model_all_forms():
    assert CLI._extract_model(["--model", "hands_recognition", "--level", "1"]) == (
        "hands_recognition",
        ["--level", "1"],
    )
    assert CLI._extract_model(["--model=cognition", "--stage", "2"]) == (
        "cognition",
        ["--stage", "2"],
    )
    assert CLI._extract_model(["--level", "1"]) == (None, ["--level", "1"])


def test_commands_table_is_well_formed():
    for name, spec in CLI.COMMANDS.items():
        assert len(spec) == 4, name
        _group, kind, target, help_text = spec
        assert kind in {"script", "run", "module", "builtin"}
        assert help_text
        if kind == "run":  # a use case must exist on disk for at least the default model
            assert isinstance(target, str) and target


def test_run_use_cases_exist_on_disk():
    # Every `run` command resolves to models/<model>/uses/<target>/run_<target>.py
    pairs = {"chat": "cognition", "agent": "cognition", "camera": "hands_recognition"}
    for name, (_g, kind, target, _h) in CLI.COMMANDS.items():
        if kind == "run":
            model = pairs.get(name, "cognition")
            app = _REPO / "models" / model / "uses" / target / f"run_{target}.py"
            assert app.exists(), f"{name} → {app} missing"


def test_available_models_includes_both():
    models = CLI._available_models()
    assert "cognition" in models and "hands_recognition" in models


def test_dispatch_run_builds_use_case_path(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        CLI.subprocess, "call", lambda cmd, cwd: captured.setdefault("cmd", cmd) or 0
    )
    CLI._dispatch("camera", ["--model", "hands_recognition", "--selftest"])
    cmd = captured["cmd"]
    assert str(_REPO / "models" / "hands_recognition" / "uses" / "camera" / "run_camera.py") in cmd
    assert "--selftest" in cmd
    assert "--model" not in cmd  # the model selects the use case, not forwarded as an arg


def test_dispatch_script_forwards_model_only_when_aware(monkeypatch):
    calls = []
    monkeypatch.setattr(CLI.subprocess, "call", lambda cmd, cwd: calls.append(cmd) or 0)
    CLI._dispatch("train", ["--model", "cognition", "--level", "1"])
    assert calls[-1].count("--model") == 1 and "cognition" in calls[-1]  # train is model-aware
    CLI._dispatch("prepare-mm", ["--model", "cognition"])
    assert "--model" not in calls[-1]  # prepare-mm is NOT model-aware → not forwarded


def test_info_lists_models_and_stages(capsys):
    rc = CLI._cmd_info(["--model", "hands_recognition"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "cognition" in out and "hands_recognition" in out
    assert "Hand keypoint regression" in out


def test_info_unknown_model_errors(capsys):
    rc = CLI._cmd_info(["--model", "does_not_exist"])
    assert rc == 2
    assert "Unknown model" in capsys.readouterr().out


def test_info_level_adds_status_columns(capsys):
    CLI._cmd_info(["--model", "cognition", "--level", "1"])
    out = capsys.readouterr().out
    assert "data(L1)" in out and "trained" in out


def test_main_no_args_prints_overview(monkeypatch, capsys):
    monkeypatch.setattr(CLI.sys, "argv", ["rdmca"])
    assert CLI.main() == 0
    assert "rdmca" in capsys.readouterr().out.lower()


def test_main_unknown_command(monkeypatch, capsys):
    monkeypatch.setattr(CLI.sys, "argv", ["rdmca", "frobnicate"])
    assert CLI.main() == 2
    assert "unknown command" in capsys.readouterr().out.lower()
