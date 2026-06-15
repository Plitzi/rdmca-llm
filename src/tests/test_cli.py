"""Tests for the unified `rdmca` developer CLI (scripts/rdmca.py).

The CLI is pure routing — it must: parse a `--model` out of the forwarded args, build
the right command for each FRAMEWORK kind (script / module / builtin), forward `--model`
only to model-aware scripts, discover USE CASES per model (chat/agent/camera are NOT
hardcoded commands) and answer `info`/`uses` from the registry. We drive it without
spawning real subprocesses by stubbing `subprocess.call`.
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


def test_commands_table_is_framework_only():
    # Use cases (chat/agent/camera) are discovered per model, NOT framework commands.
    for name, spec in CLI.COMMANDS.items():
        assert len(spec) == 4, name
        _group, kind, _target, help_text = spec
        assert kind in {"script", "module", "builtin"}
        assert help_text
    for use_case in ("chat", "agent", "camera"):
        assert use_case not in CLI.COMMANDS


def test_model_uses_discovers_apps_skipping_non_runnables():
    cognition = CLI._model_uses("cognition")
    assert set(cognition) == {"chat", "agent"}  # common/, tests/, api/ stub are skipped
    hands = CLI._model_uses("hands_recognition")
    assert set(hands) == {"camera"}


def test_available_models_includes_both():
    models = CLI._available_models()
    assert "cognition" in models and "hands_recognition" in models


def test_uses_launches_use_case(monkeypatch):
    # `rdmca uses camera --model hands_recognition …` launches the run script; the model
    # selects the use case, so it is NOT forwarded as an arg.
    captured = {}
    monkeypatch.setattr(
        CLI.subprocess, "call", lambda cmd, cwd: captured.setdefault("cmd", cmd) or 0
    )
    CLI._cmd_uses(["camera", "--model", "hands_recognition", "--selftest"])
    cmd = captured["cmd"]
    assert str(_REPO / "models" / "hands_recognition" / "uses" / "camera" / "run_camera.py") in cmd
    assert "--selftest" in cmd
    assert "--model" not in cmd


def test_uses_infers_model_for_unambiguous_app(monkeypatch):
    # `camera` is owned only by hands_recognition → no --model needed; it must NOT fall
    # back to the default model (cognition) and fail.
    captured = {}
    monkeypatch.setattr(CLI.subprocess, "call", lambda cmd, cwd: captured.update(cmd=cmd) or 0)
    rc = CLI._cmd_uses(["camera", "--selftest"])
    assert rc == 0
    assert "run_camera.py" in str(captured["cmd"])


def test_uses_unknown_app_with_explicit_wrong_model_hints_owner(capsys):
    # Forcing the wrong model for a real app points at the model that DOES own it.
    rc = CLI._cmd_uses(["camera", "--model", "cognition"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "hands_recognition" in out and "rdmca uses camera --model hands_recognition" in out


def test_uses_duplicate_app_requires_model(monkeypatch, capsys):
    # If several models declare the SAME app name, it is ambiguous → --model is required,
    # nothing is assumed (not even cognition). Simulate a collision on `chat`.
    monkeypatch.setattr(
        CLI,
        "_all_uses_by_model",
        lambda: {"cognition": {"chat": object()}, "other": {"chat": object()}},
    )
    called = []
    monkeypatch.setattr(CLI.subprocess, "call", lambda cmd, cwd: called.append(cmd) or 0)
    rc = CLI._launch_use(None, "chat", [])
    out = capsys.readouterr().out
    assert rc == 2 and not called  # refused to launch
    assert "several models" in out and "cognition" in out and "other" in out
    assert "--model" in out


def test_uses_lists_per_model(capsys):
    rc = CLI._cmd_uses([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "cognition" in out and "chat" in out and "agent" in out
    # camera is unambiguous → listed WITHOUT --model (it's inferred when launching).
    assert "rdmca uses camera" in out and "rdmca uses camera --model" not in out


def test_bare_use_case_word_is_not_a_command(monkeypatch, capsys):
    # There is NO global `rdmca chat`; it must teach the `rdmca uses chat` form.
    monkeypatch.setattr(CLI.sys, "argv", ["rdmca", "chat"])
    rc = CLI.main()
    out = capsys.readouterr().out
    assert rc == 2
    assert "unknown command 'chat'" in out and "rdmca uses chat --model cognition" in out


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
    assert "Hand keypoint detection" in out


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
    # A word that is neither a framework command nor any model's use case.
    monkeypatch.setattr(CLI.sys, "argv", ["rdmca", "frobnicate"])
    assert CLI.main() == 2
    out = capsys.readouterr().out.lower()
    assert "unknown command 'frobnicate'" in out and "rdmca uses" in out
