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
    6: {"name": "Multimodal",    "trigger": "cross-modal grounding (Phase 3+)",            "rank": 8},
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


def build_all_sectors(d_model: int, n_layers: int) -> Dict[int, SectorAdapter]:
    """Instantiate all seven sector adapters."""
    sectors = {}
    for sid, meta in SECTORS.items():
        cfg = LoRAConfig(d_model=d_model, n_layers=n_layers,
                         sector_id=sid, rank=meta["rank"])
        sectors[sid] = SectorAdapter(cfg)
    return sectors


def apply_masked_update(sectors: Dict[int, SectorAdapter],
                        active_sector_id: int,
                        loss: mx.array,
                        optimizer) -> None:
    """
    Gradient masking — only sector s* receives gradients during consolidation.
    Implementation Guide §1.6.1.
    """
    # TODO: implement gradient masking with MLX stop_gradient
    # Only the active sector's parameters should be in the optimizer step.
    raise NotImplementedError
