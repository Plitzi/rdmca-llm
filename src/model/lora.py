"""
LoRA Adaptive Sector Modules — RDMCA §9 Parameter Sectorization
Seven LoRA adapters attached to the frozen foundational model.
Each adapts Q, K, V, O projection layers of every attention block.
Zero-output initialization: at init the model behaves identically to the base.
Gradient masking ensures only the target sector receives updates per cycle.

Backend-neutral (written against `src.backend.current()`).
"""
from __future__ import annotations
from typing import Dict, List

import src.backend as backend
from src.model.config import LoRAConfig

B = backend.current()
nn = B.nn
ops = B.ops


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


class LoRALinear(nn.Module):
    """Single LoRA-adapted linear layer. Zero-output at initialization."""

    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float = 1.0):
        super().__init__()
        self.scale = alpha / rank
        # A initialized with default init; B initialized at zero
        self.lora_A = nn.Linear(in_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_dim, bias=False)
        # Zero-init B so output delta is 0 at start
        self.lora_B.weight = nn.Parameter(ops.zeros((out_dim, rank)))

    def __call__(self, x):
        return self.lora_B(self.lora_A(x)) * self.scale

    def grow(self, delta: int) -> None:
        """Increase rank by `delta`, preserving current output (PGQ expansion,
        Guide §4.2 / GradMax-style): new A rows are small-random, new B columns
        are zero, so the added components produce zero output at first."""
        in_dim   = self.lora_A.weight.shape[1]
        out_dim  = self.lora_B.weight.shape[0]
        old_rank = self.lora_A.weight.shape[0]
        new_rank = old_rank + delta
        newA = ops.concatenate(
            [self.lora_A.weight, ops.randn((delta, in_dim)) * 0.02], axis=0)
        newB = ops.concatenate(
            [self.lora_B.weight, ops.zeros((out_dim, delta))], axis=1)
        self.lora_A = nn.Linear(in_dim, new_rank, bias=False)
        self.lora_A.weight = nn.Parameter(newA)
        self.lora_B = nn.Linear(new_rank, out_dim, bias=False)
        self.lora_B.weight = nn.Parameter(newB)
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
        # One LoRA adapter per (layer, projection) pair. ModuleList of
        # ModuleDict so params register under both backends.
        self.adapters = nn.ModuleList([
            nn.ModuleDict({
                "q": LoRALinear(cfg.d_model, cfg.d_model, cfg.rank, cfg.alpha),
                "k": LoRALinear(cfg.d_model, cfg.d_model, cfg.rank, cfg.alpha),
                "v": LoRALinear(cfg.d_model, cfg.d_model, cfg.rank, cfg.alpha),
                "o": LoRALinear(cfg.d_model, cfg.d_model, cfg.rank, cfg.alpha),
            })
            for _ in range(cfg.n_layers)
        ])

    def delta(self, layer_idx: int, proj: str, x):
        """Return the LoRA delta for a specific projection in a specific layer."""
        return self.adapters[layer_idx][proj](x)

    def grow_rank(self, delta: int) -> int:
        """Grow every projection's LoRA rank by `delta` (PGQ sector expansion).
        Returns the new rank."""
        for layer in self.adapters:
            for proj in ("q", "k", "v", "o"):
                layer[proj].grow(delta)
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


def masked_sector_update(model,
                         sector_id: int,
                         loss_fn,
                         optimizer) -> tuple:
    """
    Apply a gradient update to exactly one sector, leaving the frozen
    foundational core and every other sector bit-for-bit unchanged
    (Implementation Guide §1.6.1 — sector isolation by construction).

    Isolation is enforced through the backend's trainable-mask: the whole model
    is frozen, only `sector_id`'s adapter is made trainable, so the gradient is
    computed w.r.t. that sector alone and the optimizer only touches params with
    a gradient.

    Args:
        model:     RDMCAFoundational with sectors attached.
        sector_id: the sector s* to update.
        loss_fn:   callable(model) -> scalar loss (already closes over the
                   consolidation batch and sets active sectors).
        optimizer: a backend optimizer instance.

    Returns:
        (loss_value: float, grad_norm: float)
    """
    if not model.sectors or sector_id not in model.sectors:
        raise ValueError(f"sector {sector_id} not attached to model")

    B.engine.set_trainable(model, [model.sectors[sector_id]])

    grad_fn = B.engine.value_and_grad(model, loss_fn)
    loss, grads = grad_fn(model)
    B.engine.eval(loss)
    gnorm = B.engine.grad_norm(model, grads)

    B.engine.optimizer_step(optimizer, model, grads)

    B.engine.freeze_all(model)           # restore: nothing trainable between cycles
    return B.engine.item(loss), gnorm


# Backwards-compatible alias for older call sites / Implementation Guide naming.
def apply_masked_update(model, sector_id: int, loss_fn, optimizer) -> tuple:
    return masked_sector_update(model, sector_id, loss_fn, optimizer)
