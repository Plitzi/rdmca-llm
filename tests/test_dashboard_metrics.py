"""
Dashboard construction + metrics.csv sink.

Regression guard: __init__ must finish setting model-geometry attributes (params,
n_layers, …) even though the metrics-sink setup sits in the middle of it — a stray
method def once truncated __init__ so `.params` was never set and `with dash:` crashed.
Also checks the metrics.csv schema/rows that scripts/plot_metrics.py consumes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.dashboard import TrainingDashboard


def _dash(tmp_path, **kw):
    return TrainingDashboard(stage=2, n_tokens_target=1_000_000, params=8_400_000,
                             n_layers=6, d_model=256, plain=True,
                             log_path=str(tmp_path / "train.log"), loss_ce_weight=1.3, **kw)


def test_init_sets_geometry_attrs(tmp_path):
    dash = _dash(tmp_path)
    # The attributes __enter__ / the layout read — must exist after construction.
    for attr in ("params", "n_layers", "d_model", "gate_score", "gate_floor", "gate_best"):
        assert hasattr(dash, attr), f"__init__ did not set {attr}"
    assert dash.params == 8_400_000
    dash.__exit__()


def test_metrics_csv_train_and_gate_rows(tmp_path):
    import csv
    dash = _dash(tmp_path)
    dash.update(500, 100_000, 0.42, 3e-4, 300.0, grad_norm=0.5, replay=False)  # skill batch
    dash.update(510, 110_000, 4.50, 3e-4, 300.0, grad_norm=0.5, replay=True)   # rehearsal
    dash.set_gate_result(17.2, False, threshold=45.0, best=17.25)   # not a new best
    dash.set_gate_result(16.9, True, threshold=45.0, best=17.2)     # new best
    dash.__exit__()

    rows = list(csv.DictReader((tmp_path / "metrics.csv").read_text().splitlines()))
    train = [r for r in rows if r["kind"] == "train"]
    gate  = [r for r in rows if r["kind"] == "gate"]
    assert len(train) == 2 and len(gate) == 2
    assert train[0]["loss"] == "0.4200" and train[0]["replay"] == "0"   # skill batch tagged
    assert train[1]["replay"] == "1"                                    # rehearsal tagged
    assert gate[0]["passed"] == "0" and gate[-1]["passed"] == "1"       # ratchet result


def test_ema_splits_bimodal_loss(tmp_path):
    """The per-type EMA separates the bimodal rehearsal loss so the trend is readable
    (the 'spikes' are just alternating populations, not instability)."""
    dash = _dash(tmp_path)
    for _ in range(20):
        dash.update(0, 0, 0.40, 3e-4, 300.0, replay=False)   # narrow skill ~0.4
        dash.update(0, 0, 4.50, 3e-4, 300.0, replay=True)    # conversation ~4.5
    assert dash._ema_primary < 1.0          # skill EMA settles low
    assert dash._ema_replay  > 3.0          # rehearsal EMA settles high
    dash.__exit__()
