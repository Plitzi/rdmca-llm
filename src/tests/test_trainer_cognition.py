"""Real text-LM (cognition) training path WITHOUT a trained tokenizer: we drive the
actual `train_stage` loop with a fake loader that yields (tokens, mask) batches, so the
genuine text pieces run — build_stage_model, the MRL+aux objective, the perplexity gate
(eval_ce), checkpointing. Proves the cognition training path is wired correctly; the full
data+tokenizer run is yours (see the cognition GUIDE)."""

import numpy as np

from src.plugins import set_active_model

_CTX = 16
_VOCAB = 64
_CFG = {
    "model_name": "cognition",
    "level": 0,
    "skip_gate": True,
    "moe": {"enabled": False, "aux_loss_weight": 0.01},
    "model": {
        "d_model": 32,
        "n_layers": 1,
        "n_heads": 2,
        "n_kv_heads": 1,
        "ffn_dim": 64,
        "context_len": _CTX,
        "vocab_size": _VOCAB,
        "mrl_dims": [16, 32],
        "dropout": 0.0,
    },
    "curriculum": {"stage1": {"n_tokens": 2000}},
    "training": {
        "precision": "fp32",
        "lr": 5e-3,
        "lr_min": 1e-3,
        "weight_decay": 0.0,
        "batch_size": 4,
        "grad_accumulation": 1,
        "warmup_steps": 1,
        "save_every": 1000,
        "eval_every": 1000,
        "clip_grad_norm": 1.0,
        "max_corpus_passes": 1,
        "early_stop_patience": 0,
        "seed": 0,
    },
}


class _FakeTextLoader:
    """Yields (tokens, mask) numpy batches of shape [bs, ctx+1] — the same contract the
    real TextDataset gives the trainer — so the text objective/gate run unchanged."""

    def __init__(self):
        self.epoch_tokens = 0
        self.passes = 0
        self.last_was_replay = False
        self.replay_fraction = 0.0
        self.replay_dirs = []
        self._rng = np.random.default_rng(0)

    def next_batch(self):
        toks = self._rng.integers(0, _VOCAB, size=(4, _CTX + 1)).astype(np.int64)
        mask = np.ones((4, _CTX + 1), dtype=np.float32)
        return toks, mask

    def skip(self, n):
        return n

    def save_skip_index(self, path):
        pass

    def load_skip_index(self, path):
        return False


def test_train_stage_cognition_text_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # checkpoints under tmp, not the repo dist/
    set_active_model("cognition")
    # Swap the tokenizer-dependent loader for a fake — the rest of the text path is real.
    import src.training.dataload as dataload

    monkeypatch.setattr(dataload, "build_data_loader", lambda stage, cfg: _FakeTextLoader())
    from src.training.trainer import train_stage

    ok = train_stage(stage=1, cfg=_CFG, plain=True)
    assert ok is True  # skip_gate → graduates after the budget
    stage_dir = tmp_path / "dist" / "checkpoints" / "cognition" / "level0" / "stage1"
    assert (stage_dir / "final.npz").exists() or (stage_dir / "best.npz").exists()
    assert (stage_dir / "audit.json").exists()  # text-model audit (has n_heads/vocab/mrl)


def test_train_stage_retention_and_resume(tmp_path, monkeypatch):
    """A cognitive stage >1 takes the RETENTION-val path, and --resume re-loads the
    checkpoint and fast-forwards the stream."""
    monkeypatch.chdir(tmp_path)
    set_active_model("cognition")
    import src.training.dataload as dataload

    monkeypatch.setattr(dataload, "build_data_loader", lambda stage, cfg: _FakeTextLoader())
    from src.training.trainer import train_stage

    cfg = dict(_CFG)
    cfg["curriculum"] = {"stage1": {"n_tokens": 2000}, "stage2": {"n_tokens": 2000}}
    assert train_stage(stage=1, cfg=cfg, plain=True) is True
    # Stage 2 is cognitive → retention gate (folds in stage-1 conversation if present).
    assert train_stage(stage=2, cfg=cfg, plain=True) is True
    # Resume stage 1: re-loads the historical best + fast-forwards (no error, graduates).
    assert train_stage(stage=1, cfg=cfg, resume=True, plain=True) is True
