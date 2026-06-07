"""
LoRA Adaptive Sector Modules — RDMCA §9 Parameter Sectorization
Seven LoRA adapters attached to the frozen foundational model.
Each adapts Q, K, V, O projection layers of every attention block.
Zero-output initialization: at init the model behaves identically to the base.
Gradient masking ensures only the target sector receives updates per cycle.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn


# Sector registry — matches Implementation Guide §1.6
SECTORS: Dict[int, Dict] = {
    1: {"name": "Linguistic",    "trigger": "conversational, stylistic, discourse",       "rank": 16},
    2: {"name": "Formal",        "trigger": "mathematical, logical, symbolic",             "rank": 16},
    3: {"name": "WorldKnowledge","trigger": "factual, domain-specific, encyclopedic",      "rank": 8},
    4: {"name": "Procedural",    "trigger": "planning, tool use, sequential action",       "rank": 8},
    5: {"name": "Social",        "trigger": "pragmatics, intent, social norms",            "rank": 8},
    6: {"name": "Multimodal",    "trigger": "cross-modal grounding (image/audio ↔ text)", "rank": 8},
    7: {"name": "Behavioral",    "trigger": "ethics, constraints — adversarial buffer ONLY","rank": 4},
}


@dataclass
class LoRAConfig:
    d_model: int
    n_layers: int
    sector_id: int
    rank: int
    alpha: float = 1.0   # scaling: weight = alpha / rank


class LoRALinear(nn.Module):
    """Single LoRA-adapted linear layer. Zero-output at initialization."""

    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float = 1.0):
        super().__init__()
        self.scale = alpha / rank
        # A initialized with Kaiming normal; B initialized at zero
        self.lora_A = nn.Linear(in_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_dim, bias=False)
        # Zero-init B so output delta is 0 at start
        self.lora_B.weight = mx.zeros((out_dim, rank))

    def __call__(self, x: mx.array) -> mx.array:
        return self.lora_B(self.lora_A(x)) * self.scale

    def grow(self, delta: int) -> None:
        """Increase rank by `delta`, preserving current output (PGQ expansion,
        Guide §4.2 / GradMax-style): new A rows are small-random, new B columns
        are zero, so the added components produce zero output at first."""
        in_dim   = self.lora_A.weight.shape[1]
        out_dim  = self.lora_B.weight.shape[0]
        old_rank = self.lora_A.weight.shape[0]
        new_rank = old_rank + delta
        newA = mx.concatenate(
            [self.lora_A.weight, mx.random.normal((delta, in_dim)) * 0.02], axis=0)
        newB = mx.concatenate(
            [self.lora_B.weight, mx.zeros((out_dim, delta))], axis=1)
        self.lora_A = nn.Linear(in_dim, new_rank, bias=False)
        self.lora_A.weight = newA
        self.lora_B = nn.Linear(new_rank, out_dim, bias=False)
        self.lora_B.weight = newB
        self.scale = self.scale * old_rank / new_rank   # keep alpha/rank constant


class SectorAdapter(nn.Module):
    """
    LoRA adapters for all Q, K, V, O projections across all transformer layers.
    Sector s* — one per cognitive domain (see SECTORS registry).
    """

    def __init__(self, cfg: LoRAConfig):
        super().__init__()
        self.sector_id = cfg.sector_id
        self.rank      = cfg.rank
        # One LoRA adapter per (layer, projection) pair
        self.adapters: List[Dict[str, LoRALinear]] = [
            {
                "q": LoRALinear(cfg.d_model, cfg.d_model, cfg.rank, cfg.alpha),
                "k": LoRALinear(cfg.d_model, cfg.d_model, cfg.rank, cfg.alpha),
                "v": LoRALinear(cfg.d_model, cfg.d_model, cfg.rank, cfg.alpha),
                "o": LoRALinear(cfg.d_model, cfg.d_model, cfg.rank, cfg.alpha),
            }
            for _ in range(cfg.n_layers)
        ]

    def delta(self, layer_idx: int, proj: str, x: mx.array) -> mx.array:
        """Return the LoRA delta for a specific projection in a specific layer."""
        return self.adapters[layer_idx][proj](x)

    def grow_rank(self, delta: int) -> int:
        """Grow every projection's LoRA rank by `delta` (PGQ sector expansion).
        Returns the new rank."""
        for layer in self.adapters:
            for proj in layer.values():
                proj.grow(delta)
        self.rank += delta
        return self.rank


def build_all_sectors(d_model: int, n_layers: int) -> Dict[int, SectorAdapter]:
    """Instantiate all seven sector adapters."""
    sectors = {}
    for sid, meta in SECTORS.items():
        cfg = LoRAConfig(d_model=d_model, n_layers=n_layers,
                         sector_id=sid, rank=meta["rank"])
        sectors[sid] = SectorAdapter(cfg)
    return sectors


def _grad_norm(grads) -> float:
    """L2 norm over a (possibly nested) MLX gradient tree."""
    from mlx.utils import tree_flatten
    sq = 0.0
    for _, g in tree_flatten(grads):
        if isinstance(g, mx.array) and g.size > 0:
            sq += float((g * g).sum().item())
    return sq ** 0.5


def masked_sector_update(model,
                         sector_id: int,
                         loss_fn,
                         optimizer) -> tuple:
    """
    Apply a gradient update to exactly one sector, leaving the frozen
    foundational core and every other sector bit-for-bit unchanged
    (Implementation Guide §1.6.1 — sector isolation by construction).

    Isolation is enforced through MLX freeze/unfreeze: the whole model is
    frozen, only `sector_id`'s adapter is unfrozen, so `value_and_grad`
    differentiates w.r.t. that sector alone and the optimizer never allocates
    state for any other parameter.

    Args:
        model:     RDMCAFoundational with sectors attached.
        sector_id: the sector s* to update.
        loss_fn:   callable(model) -> scalar mx.array (already closes over the
                   consolidation batch and sets active sectors).
        optimizer: an mlx optimizer instance.

    Returns:
        (loss_value: float, grad_norm: float)
    """
    if not model.sectors or sector_id not in model.sectors:
        raise ValueError(f"sector {sector_id} not attached to model")

    model.freeze()                       # freeze foundational core + all sectors
    model.sectors[sector_id].unfreeze()  # unmask only the target sector

    lg = nn.value_and_grad(model, loss_fn)
    loss, grads = lg(model)
    mx.eval(loss)
    gnorm = _grad_norm(grads)

    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    model.freeze()                       # restore: nothing trainable between cycles
    return float(loss.item()), gnorm


# Backwards-compatible alias for older call sites / Implementation Guide naming.
def apply_masked_update(model, sector_id: int, loss_fn, optimizer) -> tuple:
    return masked_sector_update(model, sector_id, loss_fn, optimizer)
