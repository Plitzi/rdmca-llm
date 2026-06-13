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
    dash = _dash(tmp_path)
    dash.update(500, 100_000, 2.5, 3e-4, 300.0, grad_norm=0.5)
    dash.set_gate_result(17.2, False, threshold=45.0, best=17.25)   # not a new best
    dash.set_gate_result(16.9, True, threshold=45.0, best=17.2)     # new best
    dash.__exit__()

    lines = (tmp_path / "metrics.csv").read_text().splitlines()
    assert lines[0].startswith("kind,step,tokens_m,loss,ppl,lr,tps,grad_norm,"
                               "val_ppl,best_val_ppl,passed")
    train = [l for l in lines if l.startswith("train,")]
    gate  = [l for l in lines if l.startswith("gate,")]
    assert len(train) == 1 and len(gate) == 2
    assert train[0].split(",")[3] == "2.5000"          # loss column
    assert gate[-1].split(",")[-1] == "1"              # last gate passed (new best)
    assert gate[0].split(",")[-1] == "0"               # first gate not a new best
