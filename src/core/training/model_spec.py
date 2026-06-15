"""Model-spec resolution — the seam that makes the trainer task/modality-agnostic.

The trainer never builds a network, a loader, the loss, or the gate directly; it asks
the ACTIVE model's `ModelSpec` (see src/plugins/base.py) for them. The default spec here
wires the text-LLM pieces (`setup.build_stage_model`, `dataload.build_data_loader`, the
MRL+aux objective, `gates.evaluate_gate`), so the cognition model behaves exactly as
before. A different model (e.g. hands_recognition) overrides any of these by exposing a
module-level `SPEC = ModelSpec(...)` — or a `build_spec(cfg)` factory — in its package;
the engine then trains that model with no changes here.
"""

from __future__ import annotations

import importlib

from src.plugins import ModelSpec, active_model, set_active_model


def _default_spec(cfg: dict) -> ModelSpec:
    """The built-in text-LM model: the framework's original behavior, expressed through
    the ModelSpec contract so every other model plugs into the same seam."""
    from src.core.training.dataload import build_data_loader
    from src.core.training.gates import evaluate_gate
    from src.core.training.setup import build_stage_model

    # MoE load-balance aux weight — 0.0 (no-op) on cognitive stages without routing.
    aux_w = float((cfg.get("moe", {}) or {}).get("aux_loss_weight", 0.01))

    def objective(model, batch):
        toks, mask = batch  # (tokens, loss_mask) — completion-only CE
        return model.mrl_loss(toks, mask) + aux_w * model.aux_loss()

    return ModelSpec(
        name="text-lm",
        build_model=build_stage_model,
        build_loader=build_data_loader,
        objective=objective,
        evaluate=evaluate_gate,
        gate_metric="perplexity",
    )


def active_model_spec(cfg: dict) -> ModelSpec:
    """Resolve the ModelSpec for the run's model. Selects the model from
    `cfg["model_name"]` (registry default = cognition), then asks that model's package for
    a `SPEC` or a `build_spec(cfg)` factory; falls back to the default text-LM spec when
    the model declares neither (as cognition does — it IS the default)."""
    set_active_model(cfg.get("model_name"))
    package = importlib.import_module(f"models.{active_model()}")
    spec = getattr(package, "SPEC", None)
    if spec is None and hasattr(package, "build_spec"):
        spec = package.build_spec(cfg)
    return spec or _default_spec(cfg)
