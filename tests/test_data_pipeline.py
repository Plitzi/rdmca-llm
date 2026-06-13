"""
End-to-end data/training-pipeline tests:

  - prepare_data.write_jsonl applies the normalization + garbage gate to EVERY record
    it writes (the single ingestion choke point), so smart quotes are normalized and
    broken records are dropped on disk;
  - the graduation gate's validation_perplexity routes (tokens, mask) pairs through the
    COMPLETION-masked eval_ce (matching training) and bare arrays through the plain
    mean — the fix for the inflated-perplexity gate.
"""
import importlib.util
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

import src.backend as backend
from src.model.transformer import RDMCAFoundational
from src.model.config import ModelConfig


def _load_prepare_data():
    spec = importlib.util.spec_from_file_location(
        "prepare_data", str(Path(__file__).parent.parent / "scripts" / "prepare_data.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── write_jsonl applies normalization + garbage gate ───────────────────────────
def test_write_jsonl_normalizes_and_drops_garbage(tmp_path):
    pd = _load_prepare_data()
    records = [
        {"text": "“Hello”   world — it’s fine", "lang": "en"},     # smart punct + spaces
        {"text": "real words here " + "�" * 40, "lang": "en"},  # mojibake → dropped
        {"text": "=" * 50, "lang": "en"},                            # symbol soup → dropped
        {"text": "The quick brown fox jumps.", "lang": "en"},
    ]
    out = tmp_path / "src.jsonl"
    pd.write_jsonl(iter(records), out, token_budget_m=10, verbose=False, val_fraction=0.0)

    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    texts = [r["text"] for r in rows]
    assert '"Hello" world - it\'s fine' in texts          # normalized
    assert "The quick brown fox jumps." in texts
    assert len(texts) == 2                                  # the two garbage rows dropped
    assert not any("�" in t or t.startswith("==") for t in texts)


# ── gate validation_perplexity: masked routing ────────────────────────────────
def _tiny_model():
    cfg = ModelConfig(d_model=64, n_heads=2, ffn_dim=128, context_len=32,
                      vocab_size=256, mrl_dims=[32, 64], dropout=0.0)
    return RDMCAFoundational(cfg)


def test_validation_perplexity_routes_mask_and_bare(tmp_path):
    import train_stage as T
    m = _tiny_model()
    rng = np.random.RandomState(0)
    toks = rng.randint(0, 256, (2, 33)).astype(np.int32)
    mask = np.zeros((2, 33), dtype=np.int32); mask[:, 20:] = 1     # only the tail trains
    B = backend.current()

    # Tuple path → masked eval_ce; bare path → unmasked. Compare to direct calls.
    masked_ppl   = T.validation_perplexity(m, [(toks, mask)])
    unmasked_ppl = T.validation_perplexity(m, [toks])
    exp_masked   = float(np.exp(B.engine.item(
        m.eval_ce(B.ops.array(toks), mask=B.ops.array(mask)))))
    exp_unmasked = float(np.exp(B.engine.item(m.eval_ce(B.ops.array(toks)))))

    assert abs(masked_ppl - exp_masked) < 1e-2
    assert abs(unmasked_ppl - exp_unmasked) < 1e-2
    # The mask actually changes the metric (it isn't silently ignored).
    assert abs(masked_ppl - unmasked_ppl) > 1e-3


def test_validation_perplexity_averages_over_batches():
    import train_stage as T
    m = _tiny_model()
    rng = np.random.RandomState(1)
    batches = [rng.randint(0, 256, (2, 33)).astype(np.int32) for _ in range(3)]
    B = backend.current()
    got = T.validation_perplexity(m, batches)
    ces = [B.engine.item(m.eval_ce(B.ops.array(b))) for b in batches]
    assert abs(got - float(np.exp(np.mean(ces)))) < 1e-2


# ── graduation gate enforced from level 1 ──────────────────────────────────────
def test_level1_gate_enforced_by_default():
    """Quality-first: level 1 must NOT skip the graduation gate (skip_gate False)."""
    from src.config import resolve_config_path, load_config
    cfg = load_config(resolve_config_path(None, 1))
    assert cfg.get("skip_gate") is False


def test_gate_decision_ratchets_against_best():
    """The graduation gate RATCHETS. Returns (is_candidate, is_new_best, is_meaningful):
    is_candidate = cleared the floor (eligible to be a best); is_new_best = candidate AND
    STRICTLY beats the best (ANY gain → saved, never discarded); is_meaningful = the gain
    exceeds min_delta (plateau accounting only). An above-floor point is never a best, no
    matter how much it improves on a worse above-floor attempt. Mirrors the user's
    example with floor 50."""
    import train_stage as T
    # 35 (from ∞) → candidate, new best, meaningful → PASSED; bar is now 35.
    assert T.gate_decision(35.0, float("inf"), 50.0) == (True, True, True)
    # 39 clears the floor (candidate) but ≥ best 35 → not a new best → discarded.
    assert T.gate_decision(39.0, 35.0, 50.0) == (True, False, False)
    # 30 < best 35 → new best (meaningful) → PASSED; bar ratchets to 30.
    assert T.gate_decision(30.0, 35.0, 50.0) == (True, True, True)
    # A checkpoint ABOVE the floor is NOT a candidate and NOT a best — even from ∞.
    assert T.gate_decision(55.0, float("inf"), 50.0) == (False, False, False)
    # 87.59 > floor 50 → not viable, NOT a best (the reported log bug: it was being
    # labelled "new best" though it never passed the default gate).
    assert T.gate_decision(87.59, float("inf"), 50.0) == (False, False, False)


def test_gate_decision_any_gain_is_a_new_best():
    """A strictly-better checkpoint is ALWAYS a new best (saved) — the user's confusion
    ('16.11 ≥ best 16.14 → discarded' was wrong). min_delta NO LONGER gates saving; it
    only flags whether the gain is 'meaningful' (for plateau/early-stop)."""
    import train_stage as T
    # 16.11 vs 16.14 (a 0.2% gain): a NEW BEST (saved), but below the 0.5% min_delta so
    # not 'meaningful' — it ratchets the bar yet counts toward the plateau.
    cand, new_best, meaningful = T.gate_decision(16.11, 16.14, 50.0, min_delta=0.005)
    assert (cand, new_best) == (True, True)
    assert meaningful is False
    # A clearly-larger gain IS meaningful (resets the plateau when measured vs the ref).
    _, new_best2, meaningful2 = T.gate_decision(16.0, 16.14, 50.0, min_delta=0.005)
    assert new_best2 is True and meaningful2 is True
    # No improvement at all → not a new best.
    assert T.gate_decision(16.20, 16.14, 50.0)[1] is False


def test_narrow_stages_have_gentler_lr_scale():
    """Per-stage lr_scale/rehearsal are STAGE properties (apply at EVERY level) and live in
    src/training/stages.py, NOT each level's yaml. The narrow eroders (3 arithmetic, 5 CoT)
    train the SHARED core at a reduced LR so they nudge it instead of overwriting
    conversation; stage 1 (conversation) trains at full LR. The schedule scales linearly."""
    import train_stage as T
    from src.training.stages import STAGE_LR_SCALE, STAGE_REHEARSAL, DEFAULT_LR_SCALE
    assert STAGE_LR_SCALE.get(1, DEFAULT_LR_SCALE) == 1.0     # conversation: full LR
    assert STAGE_LR_SCALE[3] <= 0.5                           # arithmetic: gentlest
    assert STAGE_LR_SCALE[5] <= 0.5                           # CoT: gentlest
    assert STAGE_LR_SCALE[3] < STAGE_LR_SCALE[2] <= 1.0
    assert STAGE_REHEARSAL[3] >= 0.45 and STAGE_REHEARSAL[5] >= 0.45   # strongest rehearsal
    # A level's yaml may still OVERRIDE per stage; absent that, the stage default applies.
    from src.config import resolve_config_path, load_config
    cfg = load_config(resolve_config_path(None, 1))
    assert "lr_scale" not in cfg["curriculum"]["stage3"]      # inherited from code, not yaml
    # cosine_lr scales linearly with the (already-scaled) base/min it is handed.
    full = T.cosine_lr(300, 3e-4, 3e-5, 500, 5000)
    half = T.cosine_lr(300, 1.5e-4, 1.5e-5, 500, 5000)
    assert abs(half - full * 0.5) < 1e-9


def test_evaluate_gate_respects_threshold_and_config_override():
    """evaluate_gate passes iff ppl ≤ threshold, and cfg.gate.max_perplexity overrides
    the default — so the bar is both meaningful and tunable per stage."""
    import train_stage as T
    m = _tiny_model()
    rng = np.random.RandomState(2)
    val = [rng.randint(0, 256, (2, 33)).astype(np.int32)]
    ppl0, _ = T.evaluate_gate(m, 1, val, {"gate": {"max_perplexity": {1: 1e9}}},
                              log=lambda *a, **k: None)
    # A threshold just above the measured ppl passes; just below fails.
    _, passed_hi = T.evaluate_gate(m, 1, val, {"gate": {"max_perplexity": {1: ppl0 + 1}}},
                                   log=lambda *a, **k: None)
    _, passed_lo = T.evaluate_gate(m, 1, val, {"gate": {"max_perplexity": {1: ppl0 - 1}}},
                                   log=lambda *a, **k: None)
    assert passed_hi and not passed_lo
