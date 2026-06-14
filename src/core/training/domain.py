"""Domain resolution — the seam that makes the trainer task/modality-agnostic.

The trainer never builds a model, a loader, the loss, or the gate directly; it asks
the ACTIVE domain's `DomainSpec` (see src/plugins/base.py) for them. The default spec
here wires the text-LLM pieces (`setup.build_stage_model`, `dataload.build_data_loader`,
the MRL+aux objective, `gates.evaluate_gate`), so the cognition domain behaves exactly
as before. A different domain (e.g. hands_recognition) overrides any of these by
exposing a module-level `DOMAIN = DomainSpec(...)` — or a `build_domain(cfg)` factory —
in its package; the engine then trains that model with no changes here.
"""

from __future__ import annotations

import importlib

from src.plugins import DomainSpec, active_domain, set_domain


def _default_spec(cfg: dict) -> DomainSpec:
    """The built-in text-LM domain: the framework's original behavior, expressed through
    the DomainSpec contract so every other domain plugs into the same seam."""
    from src.core.training.dataload import build_data_loader
    from src.core.training.gates import evaluate_gate
    from src.core.training.setup import build_stage_model

    # MoE load-balance aux weight — 0.0 (no-op) on cognitive stages without routing.
    aux_w = float((cfg.get("moe", {}) or {}).get("aux_loss_weight", 0.01))

    def objective(model, batch):
        toks, mask = batch  # (tokens, loss_mask) — completion-only CE
        return model.mrl_loss(toks, mask) + aux_w * model.aux_loss()

    return DomainSpec(
        name="text-lm",
        build_model=build_stage_model,
        build_loader=build_data_loader,
        objective=objective,
        evaluate=evaluate_gate,
        gate_metric="perplexity",
    )


def active_domain_spec(cfg: dict) -> DomainSpec:
    """Resolve the DomainSpec for the run's domain. Selects the domain from
    `cfg["domain"]` (registry default = cognition), then asks that domain's package for
    a `DOMAIN` spec or a `build_domain(cfg)` factory; falls back to the default text-LM
    spec when the domain declares neither (as cognition does — it IS the default)."""
    set_domain(cfg.get("domain"))
    package = importlib.import_module(f"src.plugins.{active_domain()}")
    spec = getattr(package, "DOMAIN", None)
    if spec is None and hasattr(package, "build_domain"):
        spec = package.build_domain(cfg)
    return spec or _default_spec(cfg)
