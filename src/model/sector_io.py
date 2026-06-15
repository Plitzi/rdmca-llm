"""
Behavioral sectors: train/load helpers shared by the trainer and the runtimes.

The foundational cognitive core (stages 1..last-cognitive) is frozen permanently
after the cognitive base. The behavioral stages (tool use / MCP / skills, stages
7-9) must NOT overwrite that core — otherwise the model forgets how to talk. So
each behavioral stage trains a small LoRA **sector** on top of the frozen core and
is saved on its own; at inference the core is loaded and every trained behavioral
sector is attached as an always-on (explicit-routing) adapter.

This is the seam that makes the documented "frozen core + sectors" real: the
cognitive base stays intact for conversation/reasoning while tool/skill behaviour
is added additively. The cognitive MoE experts (S1..S6) and safety (S7) are a
separate concern (daily consolidation), so behavioral sectors use their own id
range and explicit routing.
"""

from __future__ import annotations

from pathlib import Path

import src.backend as backend

# Behavioral sectors live above the cognitive (S1..S6) / safety (S7) id range so
# they never collide with the MoE experts.
BEHAVIORAL_SECTOR_BASE = 100
BEHAVIORAL_RANK = 8


def sector_id_for_stage(stage: int) -> int:
    """Stable sector id for a behavioral stage (e.g. stage 8 → 108)."""
    return BEHAVIORAL_SECTOR_BASE + stage


def sectors_dir(root: Path) -> Path:
    return Path(root) / "sectors"


def sector_path(root: Path, stage: int) -> Path:
    return sectors_dir(root) / f"sector_stage{stage}.npz"


def frozen_core_path(root: Path) -> Path:
    return Path(root) / "foundational" / "theta_f_frozen.npz"


def _make_adapter(model, sector_id: int, rank: int = BEHAVIORAL_RANK):
    from src.model.config import LoRAConfig
    from src.model.lora import SectorAdapter

    return SectorAdapter(
        LoRAConfig(
            d_model=model.cfg.d_model,
            n_layers=len(model.blocks),
            sector_id=sector_id,
            rank=rank,
            kv_dim=model.cfg.kv_dim,
        )
    )


def attach_for_training(model, stage: int, rank: int = BEHAVIORAL_RANK):
    """Attach a single fresh behavioral sector for `stage`, freeze the core, and
    leave ONLY the sector trainable. Returns (sector_id, adapter). The forward
    applies the sector via explicit routing, so its delta trains while the loaded
    foundational core stays fixed."""
    B = backend.current()
    sid = sector_id_for_stage(stage)
    adapter = _make_adapter(model, sid, rank)
    model.attach_sectors({sid: adapter}, moe=False)  # explicit routing, no gate
    model.set_active_sectors([(sid, 1.0)])  # always-on during training
    B.engine.freeze_all(model)  # freeze everything…
    B.engine.set_trainable(model, [adapter])  # …then re-enable the sector only
    return sid, adapter


def save_sector(adapter, root: Path, stage: int) -> Path:
    """Persist one trained behavioral sector adapter (neutral .npz)."""
    p = sector_path(root, stage)
    p.parent.mkdir(parents=True, exist_ok=True)
    backend.current().engine.save_weights(adapter, str(p))
    return p


def trained_sector_stages(root: Path, up_to: int | None = None) -> list[int]:
    """Behavioral stages with a saved sector under `root` (optionally ≤ up_to)."""
    d = sectors_dir(root)
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("sector_stage*.npz")):
        try:
            s = int(p.stem.replace("sector_stage", ""))
        except ValueError:
            continue
        if up_to is None or s <= up_to:
            out.append(s)
    return sorted(out)


def load_for_inference(model, root: Path, stage: int | None) -> str | None:
    """Load the frozen cognitive core and attach every trained behavioral sector
    (≤ stage) onto `model`. Returns a short label, or None when there's no frozen
    core or no applicable sector — so the caller falls back to the plain per-stage
    checkpoint (e.g. when chatting at a cognitive stage, or before any freeze)."""
    root = Path(root)
    core = frozen_core_path(root)
    sects = trained_sector_stages(root, up_to=stage)
    if not (core.exists() and sects):
        return None
    backend.current().engine.load_weights(model, str(core))
    attach_trained_sectors(model, root, up_to=stage)
    return f"frozen cognitive core + behavioral sectors {sects}"


def attach_trained_sectors(model, root: Path, up_to: int | None = None) -> list[int]:
    """Attach + load every trained behavioral sector (≤ up_to) onto `model` as
    always-on adapters. Returns the attached stage list. No-op (returns []) if
    none are present, so a plain cognitive-stage checkpoint still works."""
    stages = trained_sector_stages(root, up_to)
    if not stages:
        return []
    B = backend.current()
    adapters = {
        sector_id_for_stage(s): _make_adapter(model, sector_id_for_stage(s)) for s in stages
    }
    model.attach_sectors(adapters, moe=False)
    for s in stages:
        B.engine.load_weights(adapters[sector_id_for_stage(s)], str(sector_path(root, s)))
    model.set_active_sectors([(sector_id_for_stage(s), 1.0) for s in stages])
    return stages
