"""
hands_recognition — the second model's end-to-end smoke (proves the framework is
task-agnostic: a NON-text, NON-transformer model trains/evaluates through the same
ModelSpec seam as cognition). Lives with the stage so deleting the model removes it.
"""

import src.backend as backend
from models.hands_recognition.pose import (
    HAND_CONNECTIONS,
    LANDMARK_NAMES,
    N_KEYPOINTS,
    build_pose_net,
    build_spec,
    mean_keypoint_error,
    synth_batch,
)
from models.hands_recognition.stages.stage01_keypoints.plugin import PLUGIN


def test_plugin_metadata():
    assert PLUGIN.number == 1
    assert PLUGIN.gate.metric_key == "mpjpe"
    assert PLUGIN.frozen_base is True  # single stage IS its own base
    assert PLUGIN.sources == {}  # data is synthetic (loader-generated), nothing on disk


def test_hand_topology_is_a_full_articulated_skeleton():
    """The 21 landmarks + their bones describe a real, articulated hand: every joint is
    named, all bone endpoints are valid indices, and every landmark is part of the
    skeleton (so recognizing the points reconstructs every phalanx, not a scatter)."""
    assert len(LANDMARK_NAMES) == N_KEYPOINTS
    # 4 phalanx-bones × 5 fingers? no — 3 phalanges/finger ×5 = 15, + 6 palm edges = 21
    assert len(HAND_CONNECTIONS) == 21
    in_skeleton = set()
    for a, b in HAND_CONNECTIONS:
        assert 0 <= a < N_KEYPOINTS and 0 <= b < N_KEYPOINTS
        in_skeleton.update((a, b))
    assert in_skeleton == set(range(N_KEYPOINTS))  # no orphan landmark
    # every fingertip is a leaf (appears in exactly one bone)
    for tip in (4, 8, 12, 16, 20):
        assert sum(tip in c for c in HAND_CONNECTIONS) == 1


def test_synth_batch_shapes_and_range():
    frames, keypts = synth_batch(8, seed=0)
    assert frames.shape == (8, 32 * 32)
    assert keypts.shape == (8, N_KEYPOINTS * 2)
    assert keypts.min() >= 0.0 and keypts.max() <= 1.0  # normalized image coords


def test_forward_and_keypoint_error():
    net = build_pose_net()
    ops = backend.current().ops
    frames, keypts = synth_batch(4, seed=1)
    pred = net(ops.array(frames))
    assert pred.shape == (4, N_KEYPOINTS * 2)
    err = mean_keypoint_error(pred, ops.array(keypts))
    assert err >= 0.0  # a real (lower-is-better) metric


def test_spec_seam_resolves_and_evaluates():
    """The agnostic trainer drives any model through ModelSpec — build/loader/objective/
    evaluate must all work for the hand-pose model exactly like for cognition."""
    cfg = {
        "model_name": "hands_recognition",
        "model": {"d_model": 256},
        "training": {"batch_size": 8, "seed": 0},
        "gate": {"max_mpjpe": 0.05},
    }
    spec = build_spec(cfg)
    assert spec.name == "hand-pose" and spec.gate_metric == "mpjpe"

    net, _ncfg, _aux, _prec, _seed = spec.build_model(1, cfg, None)
    loader = spec.build_loader(1, cfg)
    ops = backend.current().ops
    frames, keypts = loader.next_batch()
    loss = spec.objective(net, (ops.array(frames), ops.array(keypts)))
    assert float(backend.current().engine.item(loss)) >= 0.0

    val = [
        (ops.array(synth_batch(8, seed=i)[0]), ops.array(synth_batch(8, seed=i)[1]))
        for i in range(2)
    ]
    score, passed = spec.evaluate(net, 1, val_batches=val, cfg=cfg)
    assert isinstance(score, float) and score >= 0.0
    assert passed is False  # random weights cannot clear the 0.05 gate

    # Loader is a no-op for the text-stream concepts (synthetic, unbounded).
    assert loader.epoch_tokens == 0 and loader.load_skip_index("nope") is False
    assert loader.skip(3) == 3
