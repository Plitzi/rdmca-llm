"""
hands_recognition — REAL training integration (network-free). Drives the actual
`train_stage` loop on the synthetic hand-pose model for a few steps, proving the whole
agnostic training subsystem (trainer → model_spec → loader → objective → gate →
checkpoint) runs end-to-end for a NON-text model, and that training reduces the loss.

This is the trainer's own smoke: no dataset download, no tokenizer — the model's
ModelSpec generates its data. Checkpoints are written under a tmp cwd, never `dist/`.
"""

import numpy as np

from src.plugins import set_active_model


def _tiny_cfg() -> dict:
    return {
        "model_name": "hands_recognition",
        "level": 0,
        "skip_gate": True,  # smoke: don't block on the gate
        "gate": {"max_mpjpe": 0.05},
        "model": {"d_model": 64},
        "curriculum": {"stage1": {"n_tokens": 4000}},
        "training": {
            "precision": "fp32",
            "lr": 1e-2,
            "lr_min": 1e-3,
            "weight_decay": 0.0,
            "batch_size": 16,
            "grad_accumulation": 1,
            "warmup_steps": 2,
            "save_every": 1000,
            "eval_every": 1000,
            "clip_grad_norm": 1.0,
            "max_corpus_passes": 1,
            "early_stop_patience": 0,
            "seed": 0,
        },
    }


def test_train_stage_runs_and_learns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # checkpoints land under tmp, not the repo's dist/
    set_active_model("hands_recognition")
    from src.training.trainer import train_stage

    cfg = _tiny_cfg()
    ok = train_stage(stage=1, cfg=cfg, plain=True)
    assert ok is True  # skip_gate → graduates

    # Checkpoint artifacts were written by the real checkpoint path.
    stage_dir = tmp_path / "dist" / "hands_recognition" / "checkpoints" / "level0" / "stage1"
    assert (stage_dir / "final.npz").exists() or (stage_dir / "best.npz").exists()
    assert (stage_dir / "audit.json").exists()


def test_training_reduces_keypoint_error():
    """A few hundred optimizer steps on the synthetic task must measurably cut the mean
    keypoint error — the model genuinely LEARNS (not just runs)."""
    set_active_model("hands_recognition")
    import src.backend as backend
    from models.hands_recognition.pose import build_pose_net, mean_keypoint_error, synth_batch

    B = backend.current()
    ops = B.ops
    net = build_pose_net(64)
    frames, keypts = synth_batch(64, seed=0)
    x, y = ops.array(frames), ops.array(keypts)

    def loss_fn(m, batch):
        f, k = batch
        d = m(f) - k
        return ops.mean(d * d)

    before = mean_keypoint_error(net(x), y)
    opt = B.engine.make_optimizer(net, lr=1e-2, weight_decay=0.0)
    grad_fn = B.engine.value_and_grad(net, loss_fn)
    for _ in range(300):
        _, grads = grad_fn(net, (x, y))
        B.engine.optimizer_step(opt, net, grads)
    after = mean_keypoint_error(net(x), y)
    assert after < before * 0.6, f"training did not learn: {before:.4f} -> {after:.4f}"
    assert np.isfinite(after)


def test_train_stage_runs_on_multihand_heatmap(tmp_path, monkeypatch):
    """The REAL train_stage loop on the MULTI-HAND heatmap detector (stage 1) with a tiny FAKE
    FreiHAND tree — guards the loop staying batch-agnostic (the loader's 7-tuple flows through
    objective without the old (tokens, mask) unpack). No download; checkpoints under tmp."""
    import json

    from PIL import Image

    root = tmp_path / "freihand"
    (root / "training" / "rgb").mkdir(parents=True)
    rng = np.random.default_rng(0)
    xyz = [(rng.random((21, 3)) * 0.1 + [0.0, 0.0, 0.5]).tolist() for _ in range(8)]
    k = [[[200.0, 0.0, 112.0], [0.0, 200.0, 112.0], [0.0, 0.0, 1.0]] for _ in range(8)]
    (root / "training_xyz.json").write_text(json.dumps(xyz))
    (root / "training_K.json").write_text(json.dumps(k))
    for i in range(8):
        Image.fromarray((rng.random((96, 96, 3)) * 255).astype("uint8")).save(
            root / "training" / "rgb" / f"{i:08d}.jpg"
        )

    monkeypatch.chdir(tmp_path)
    set_active_model("hands_recognition")
    from src.training.trainer import train_stage

    cfg = {
        "model_name": "hands_recognition",
        "level": 1,
        "skip_gate": True,
        "gate": {"max_mpjpe": 5.0},
        "model": {"arch": "heatmap", "img_size": 64, "in_channels": 3, "d_model": 16,
                  "heatmap_size": 16, "dims": 3, "n_hands": 2},
        "dataset": {"root": str(root), "localize": True},
        "curriculum": {"stage1": {"n_tokens": 4000}},
        "training": {"precision": "fp32", "lr": 1e-3, "lr_min": 1e-4, "weight_decay": 0.0,
                     "batch_size": 4, "grad_accumulation": 1, "warmup_steps": 1,
                     "save_every": 1000, "eval_every": 1000, "clip_grad_norm": 1.0,
                     "max_corpus_passes": 1, "early_stop_patience": 0, "seed": 0},
    }  # fmt: skip
    assert train_stage(stage=1, cfg=cfg, plain=True) is True  # skip_gate → graduates
    stage_dir = tmp_path / "dist" / "hands_recognition" / "checkpoints" / "level1" / "stage1"
    assert (stage_dir / "final.npz").exists() or (stage_dir / "best.npz").exists()
