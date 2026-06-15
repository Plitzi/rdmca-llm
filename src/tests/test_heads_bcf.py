"""BCF head training (src/model/bcf.py) + the stage-completion seam (src/training/heads.py):
probe accuracy/step/delta on frozen-core features, and the behavioral-sector / post_stage
side effects when a stage finishes."""

import json

import numpy as np

import src.backend as backend
from src.model.bcf import BCFHead, bcf_accuracy, bcf_probe_delta, bcf_train_step
from src.training import heads


def _tiny_model():
    from src.model.config import ModelConfig
    from src.model.transformer import RDMCAFoundational

    cfg = ModelConfig(
        d_model=32, n_layers=1, n_heads=2, n_kv_heads=1, ffn_dim=64, context_len=64,
        vocab_size=64, mrl_dims=[16, 32], dropout=0.0,
    )  # fmt: skip
    return RDMCAFoundational(cfg)


class _FakeTok:
    ready = True

    def encode(self, text, add_bos=False, add_eos=False):
        return [(ord(c) % 60) + 1 for c in text][:32] or [1]


_PROBES = [("be kind", 1), ("cause harm", 0), ("help out", 1), ("do damage", 0)]


def test_bcf_accuracy_step_delta():
    model, tok = _tiny_model(), _FakeTok()
    head = BCFHead(model.cfg.d_model)
    assert bcf_accuracy(model, tok, head, []) == 1.0  # empty → trivially 1.0
    acc0 = bcf_accuracy(model, tok, head, _PROBES)
    assert 0.0 <= acc0 <= 1.0
    opt = backend.current().engine.make_optimizer(head, 1e-2, 0.0)
    loss = bcf_train_step(model, tok, head, _PROBES, opt)
    assert np.isfinite(float(loss))
    delta = bcf_probe_delta(model, tok, head, _PROBES, baseline_acc=0.5)
    assert np.isfinite(delta)


def test_train_bcf_head_skips_without_probe_file(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # no data/benchmarks/bcf_probes.jsonl here
    heads.train_bcf_head(_tiny_model(), tmp_path / "ck")
    assert "No probe set" in capsys.readouterr().out


def test_train_bcf_head_trains_with_probe_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pf = tmp_path / "data" / "benchmarks" / "bcf_probes.jsonl"
    pf.parent.mkdir(parents=True, exist_ok=True)
    pf.write_text("".join(json.dumps({"text": t, "label": y}) + "\n" for t, y in _PROBES))
    # Use the fake tokenizer so no trained SP model is needed.
    monkeypatch.setattr("src.modalities.text.TextTokenizer", _FakeTok)
    model = _tiny_model()
    heads.train_bcf_head(model, tmp_path / "ck", epochs=1)
    assert (tmp_path / "ck" / "bcf_head.npz").exists()
    assert hasattr(model, "bcf_head")


def test_on_stage_complete_behavioral_saves_sector(tmp_path):
    from src.model import sector_io

    model = _tiny_model()
    _sid, adapter = sector_io.attach_for_training(model, stage=8)
    heads.on_stage_complete(model, 8, {}, tmp_path, tmp_path / "ck", "fp32", adapter=adapter)
    assert sector_io.trained_sector_stages(tmp_path) == [8]


def test_on_stage_complete_cognitive_runs_post_stage_hook(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(heads, "model_hook", lambda name: lambda *a, **k: calls.append(name))
    # A non-last cognitive stage → post_stage hook fires, no freeze.
    monkeypatch.setattr(heads, "last_cognitive_stage", lambda cfg: 99)
    heads.on_stage_complete(_tiny_model(), 2, {}, tmp_path, tmp_path / "ck", "fp32")
    assert calls == ["post_stage"]
