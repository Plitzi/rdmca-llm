"""
Behavioral Constraint Function (BCF) — RDMCA §15
A trained safety head that classifies every candidate action/experience.
B(a, s) = 1 if the action is permissible, 0 if it violates constraints.
Trained on the BCF probe set during the ethics stage (stage 6) and frozen permanently.
Sector S7 (Behavioral) receives gradient ONLY from the adversarial buffer,
never from the standard consolidation buffer.

Constraint hierarchy (§15.2):
  C1 — Physical harm prevention           (weight 1.0, immutable)
  C2 — Epistemic integrity                (weight 0.9, immutable)
  C3 — Autonomy and agency preservation   (weight 0.8, immutable)
  C4 — Contextual appropriateness         (weight 0.5, adjustable)

Backend-neutral (written against `src.backend.current()`).
"""

from __future__ import annotations

import src.backend as backend

B = backend.current()
nn = B.nn
ops = B.ops


BCF_THRESHOLD = 0.5  # B(a,s) < threshold → action blocked


class BCFHead(nn.Module):
    """
    Lightweight binary classifier on top of the foundational hidden state.
    Input: hidden state h ∈ R^d_model at the final token position.
    Output: scalar logit → sigmoid → B(a,s) ∈ [0,1].
    """

    def __init__(self, d_model: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def __call__(self, h):
        """h: [..., d_model]  →  [..., 1] logit"""
        return self.fc2(ops.relu(self.fc1(h)))

    def score(self, h):
        """Returns B(a,s) ∈ [0,1] probability (permissible = high score)."""
        return ops.sigmoid(self(h))

    def is_permissible(self, h):
        """Boolean mask: True if B(a,s) ≥ BCF_THRESHOLD."""
        return self.score(h) >= BCF_THRESHOLD


def bcf_loss(logits, labels):
    """Binary cross-entropy (from logits) for BCF probe-set training."""
    return ops.bce_with_logits(logits.squeeze(-1), labels, reduction="mean")


def _hidden_states(model, tokenizer, texts, seq_len: int = 128):
    """Final-token foundational hidden state for each text (core only)."""
    if hasattr(model, "set_active_sectors"):
        model.set_active_sectors([])  # BCF reads the frozen core
    rows = []
    for t in texts:
        try:
            ids = tokenizer.encode(t, add_eos=True)
        except TypeError:
            ids = tokenizer.encode(t)
        ids = (ids or [0])[:seq_len]
        toks = ops.array(ids)[None]
        h = model(toks)[:, -1, :]  # [1, d_model]
        rows.append(h)
    return ops.concatenate(rows, axis=0)  # [N, d_model]


def bcf_accuracy(model, tokenizer, bcf_head: BCFHead, probes) -> float:
    """
    Accuracy of B(a,s) on a probe set.
    probes: iterable of (text, label) with label 1 = permissible, 0 = blocked.
    """
    if not probes:
        return 1.0
    texts = [p[0] for p in probes]
    labels = ops.array([float(p[1]) for p in probes])
    h = _hidden_states(model, tokenizer, texts)
    preds = ops.astype(bcf_head.score(h).squeeze(-1) >= BCF_THRESHOLD, ops.float32)
    return float(ops.astype(preds == labels, ops.float32).mean().item())


def bcf_train_step(model, tokenizer, bcf_head: BCFHead, probes, optimizer) -> float:
    """
    One supervised step on the BCF head over a probe batch. Only the BCF head
    is trainable — the foundational hidden states are read frozen. Returns loss.
    """
    texts = [p[0] for p in probes]
    labels = ops.array([float(p[1]) for p in probes])
    h = ops.stop_gradient(_hidden_states(model, tokenizer, texts))  # frozen features

    def loss_fn(head):
        return bcf_loss(head(h), labels)

    grad_fn = B.engine.value_and_grad(bcf_head, loss_fn)
    loss, grads = grad_fn(bcf_head)
    B.engine.optimizer_step(optimizer, bcf_head, grads)
    return B.engine.item(loss)


def bcf_probe_delta(model, tokenizer, bcf_head: BCFHead, probes, baseline_acc: float) -> float:
    """
    BCF accuracy change vs. a baseline (Consolidation §2.3.2). A positive
    return value means degradation (post-update accuracy fell below baseline).
    """
    return baseline_acc - bcf_accuracy(model, tokenizer, bcf_head, probes)
