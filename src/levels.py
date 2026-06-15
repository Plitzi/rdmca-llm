"""Level constructor — the standardized, REPEATED-across-models parts of a level config.

Each model owns its level configs under `models/<model>/configs/levels/` (per-model, so
multiple models don't share one global `configs/levels/`). But the boilerplate that repeats
between models — training cadence, resource estimates, and (for transformer models) the
model-size ladder by tier — lives HERE, once. A level config opts in with `tier: <name>`
and the named scaffold is deep-merged UNDERNEATH it (the config always wins), so a model's
level YAML only declares what is specific to it (data, stages, arch).

This is the "constructor" half of the per-model levels design: per-model files for what
differs, a shared tier table for what repeats. Unknown tier → empty scaffold (no-op).
"""

from __future__ import annotations

import copy

# tier → standardized scaffold (lowest-precedence layer of a level config). Keep ONLY the
# genuinely repeated, model-agnostic knobs here; anything model/arch-specific stays in the
# model's own level YAML. New tiers (e.g. a transformer size ladder shared by future
# transformer models) are added here so the boilerplate is never copy-pasted.
TIERS: dict[str, dict] = {
    # Small vision regressor/heatmap models (e.g. hands_recognition on a laptop GPU/MPS):
    # stable fp32 training, modest batch, frequent eval, tiny memory footprint.
    "vision-edge": {
        "training": {
            "precision": "fp32",
            "grad_accumulation": 1,
            "clip_grad_norm": 1.0,
            "batch_size": 32,
            "warmup_steps": 200,
            "save_every": 500,
            "eval_every": 500,
        },
        "resources": {
            "est_params_m": 0.5,
            "est_train_mem_gb": 1.0,
            "est_infer_mem_gb": 0.3,
            "min_ram_gb": 2,
        },
    },
}


def scaffold(tier: str | None) -> dict:
    """The standardized scaffold for `tier` (a deep copy so callers can't mutate the table),
    or an empty dict for None/unknown — so declaring no tier, or a tier that isn't defined,
    simply contributes nothing to the merge."""
    if not tier or tier not in TIERS:
        return {}
    return copy.deepcopy(TIERS[tier])


def available_tiers() -> list[str]:
    return sorted(TIERS)
