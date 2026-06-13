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


def test_gate_thresholds_calibrated_for_masked_metric():
    """Thresholds must reflect the MASKED metric, not the old unmasked era (~50).
    A loose threshold would let a base pass at ppl ~20 immediately — no quality bar."""
    import train_stage as T
    assert T.DEFAULT_GATE_PPL[1] <= 15.0
    assert all(v <= 15.0 for v in T.DEFAULT_GATE_PPL.values())


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
