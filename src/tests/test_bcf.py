"""BCF head (src/model/bcf.py): the binary permissibility classifier on the foundational
hidden state — forward logit, sigmoid score, threshold mask, and the BCE loss."""

import numpy as np

import src.backend as backend
from src.model.bcf import BCFHead, bcf_loss

ops = backend.current().ops


def test_bcf_head_forward_and_score_range():
    head = BCFHead(d_model=16)
    h = ops.array(np.zeros((5, 16), dtype=np.float32))
    logit = head(h)
    assert tuple(logit.shape) == (5, 1)
    score = np.asarray(ops.to_numpy(head.score(h)))
    assert score.min() >= 0.0 and score.max() <= 1.0  # sigmoid → probability


def test_bcf_is_permissible_is_boolean_mask():
    head = BCFHead(d_model=16)
    mask = head.is_permissible(ops.array(np.zeros((3, 16), dtype=np.float32)))
    vals = np.asarray(ops.to_numpy(mask)).astype(bool)
    assert vals.shape == (3, 1)


def test_bcf_loss_is_finite():
    head = BCFHead(d_model=16)
    logits = head(ops.array(np.zeros((4, 16), dtype=np.float32)))
    labels = ops.array(np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32))
    loss = bcf_loss(logits, labels)
    assert np.isfinite(float(backend.current().engine.item(loss)))
