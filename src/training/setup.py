"""Model construction + starting weights for a training stage.

Builds the model at the trained-tokenizer vocab, seeds every RNG before weight init,
and loads the right starting point: cognitive stages continue from the previous active
stage; behavioral stages load the FROZEN cognitive core and attach a trainable LoRA
sector on top. Split out of the trainer so the loop reads as just the loop.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import src.backend as backend
from src.config import get_precision
from src.training.curriculum import (
    is_behavioral_stage,
    last_cognitive_stage,
    prev_active_stage,
)


def build_stage_model(stage: int, cfg: dict, root: Path):
    """Build the stage's model and load its starting weights.

    Returns (model, model_cfg, adapter, precision). `adapter` is the trainable LoRA
    sector for a behavioral stage, else None.
    """
    mcfg = cfg["model"]

    # Override vocab_size from the trained tokenizer if available. The unified
    # multimodal layout reserves IDs 0..20479 (text+image+audio), but the text
    # tokenizer only ever emits IDs < text_vocab_size (8192). Sizing the embedding/head
    # to the full 20480 leaves ~60% of rows without gradient and lets phantom
    # image/audio logits steal softmax mass → incoherent text. Train at the real text
    # vocab.
    tok_info = Path("dist/tokenizer/tokenizer_info.json")
    if tok_info.exists():
        with open(tok_info) as f:
            info = json.load(f)
        actual_vocab = info.get("text_vocab_size", info["vocab_size"])
        if actual_vocab != mcfg.get("vocab_size"):
            print(
                f"  [vocab] Using tokenizer vocab_size={actual_vocab} "
                f"(config had {mcfg.get('vocab_size')})"
            )
            mcfg = dict(mcfg)
            mcfg["vocab_size"] = actual_vocab

    # Backend already selected by require_backend(); import model modules now so their
    # classes bind to it.
    from src.model.config import ModelConfig
    from src.model.transformer import RDMCAFoundational, set_model_precision

    B = backend.current()

    # Reproducibility: seed every RNG (Python/numpy/backend) BEFORE weight init so a run
    # is repeatable and gates are comparable. Configurable via `seed:` (top-level or
    # training.seed); fixed default keeps runs comparable across machines.
    seed = int(cfg.get("seed", (cfg.get("training", {}) or {}).get("seed", 42)))
    B.engine.set_seed(seed)

    model_cfg = ModelConfig(
        **{k: v for k, v in mcfg.items() if k in ModelConfig.__dataclass_fields__}
    )
    model = RDMCAFoundational(model_cfg)
    precision = get_precision(cfg)
    set_model_precision(model, precision)
    print(
        f"  Model: {model.count_params() / 1e6:.1f}M params | "
        f"d_model={model_cfg.d_model} | layers={model_cfg.n_layers} | "
        f"vocab={model_cfg.vocab_size} | precision={precision}"
    )

    # Starting weights. Cognitive stages continue from the previous active stage.
    # Behavioral stages (tool/MCP/skills) instead load the FROZEN cognitive core and
    # train a LoRA sector on top of it — so language/reasoning is preserved.
    adapter = None
    if is_behavioral_stage(stage):
        from src.model import sector_io

        core = sector_io.frozen_core_path(root)
        if not core.exists():
            print(
                f"ERROR: behavioral stage {stage} needs the frozen cognitive core, "
                f"but it is missing:\n  {core}"
            )
            print(
                f"  Train the cognitive base first (through stage "
                f"{last_cognitive_stage(cfg)}) — that freezes the core."
            )
            sys.exit(1)
        B.engine.load_weights(model, str(core))
        set_model_precision(model, precision)
        sid, adapter = sector_io.attach_for_training(model, stage)
        print(
            f"  Loaded frozen core; training behavioral sector S{sid} "
            f"({B.engine.param_count(adapter) / 1e3:.0f}K trainable params) on the frozen base"
        )
    else:
        prev_n = prev_active_stage(stage, cfg)
        if prev_n is not None:
            prev_ckpt = root / f"stage{prev_n}" / "latest.json"
            if prev_ckpt.exists():
                with open(prev_ckpt) as f:
                    prev_state = json.load(f)
                B.engine.load_weights(model, prev_state["checkpoint"])
                print(f"  Loaded Stage {prev_n} weights as starting point")
            else:
                print(f"  No Stage {prev_n} checkpoint found — starting from random init")

    return model, model_cfg, adapter, precision, seed
