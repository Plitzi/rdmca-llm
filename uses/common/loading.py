"""
Model + checkpoint loading for inference — shared by the chat and agent runtimes.

Picks the best checkpoint for a stage (or loads the frozen core + LoRA sectors for a
behavioral stage), syncs vocab size with the trained tokenizer, applies optional
quantization, and reports what was loaded. Kept here (not in uses/) so chat and agent
load models the same way.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import src.core.backend as backend
from src.core.config import get_precision, load_config, require_backend

# Quantization is not limited to a fixed menu: both backends do grouped-affine weight
# quantization at any bit-width in this range. 4-bit is just the smallest useful tier
# for limited-hardware testing, not the only option.
_QUANT_MIN, _QUANT_MAX = 2, 8


def parse_quant(value: str | int | None) -> int | None:
    """Parse a --quant value into a weight bit-width, or None for no quantization.

    Accepts 'none'/'off'/'' → None, or any bit-width as a plain number ('8') or
    'int'-prefixed ('int4'), clamped to the supported 2–8 bit range. Usable as an
    argparse `type=` (raises ArgumentTypeError on bad input)."""
    if value is None or isinstance(value, int):
        return value
    s = value.strip().lower()
    if s in ("none", "off", ""):
        return None
    s = s[3:] if s.startswith("int") else s
    try:
        bits = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid --quant {value!r}: use 'none' or a bit-width (e.g. 8, int4)"
        ) from None
    if not (_QUANT_MIN <= bits <= _QUANT_MAX):
        raise argparse.ArgumentTypeError(
            f"--quant bit-width {bits} out of range — supported: {_QUANT_MIN}-{_QUANT_MAX}"
        )
    return bits


def _apply_quant(model, quant) -> None:
    """Quantize model weights to a given bit-width for limited hardware; no-op for
    None/'none'. `quant` may be an int bit-width or a raw --quant string (parsed
    here). Real grouped-affine quantization on both backends at any 2–8 bit width
    — see engine.quantize in src/core/backend/{mlx,torch}_backend.py."""
    bits = parse_quant(quant)
    if bits is None:
        return
    B = backend.current()
    if not hasattr(B.engine, "quantize"):
        print(f"  [quant] backend '{B.name}' has no quantize — staying at float precision")
        return
    print(f"  Quantizing weights → {bits}-bit (limited-hardware mode)")
    B.engine.quantize(model, bits=bits)


def resolve_stage_checkpoint(stage_dir: Path):
    """Pick the checkpoint inference should use for a stage, ALWAYS preferring the BEST
    (lowest val-perplexity) over the latest training step. Returns (path|None, label,
    meta) — meta is the tracked JSON (best.json / stage_complete.json / latest.json) so
    the caller can report the model's quality. Priority:

      1. best.npz   — the running/ratcheted best (the gate's moving bar), meta=best.json;
      2. final.npz  — the graduated model (= the best at graduation);
      3. latest.json — only when no eval-best exists yet (training just started).
    """
    import json

    def _read(p: Path):
        try:
            return json.loads(p.read_text()) if p.exists() else None
        except (OSError, ValueError):
            return None

    best_npz, final_npz = stage_dir / "best.npz", stage_dir / "final.npz"
    if best_npz.exists():
        return best_npz, "best", _read(stage_dir / "best.json")
    if final_npz.exists():
        return (
            final_npz,
            "final (graduated)",
            (_read(stage_dir / "best.json") or _read(stage_dir / "stage_complete.json")),
        )
    state = _read(stage_dir / "latest.json")
    if state and state.get("checkpoint") and Path(state["checkpoint"]).exists():
        return Path(state["checkpoint"]), "latest (in-progress)", state
    return None, "none", None


def describe_checkpoint_meta(meta: dict | None) -> str:
    """One-line quality summary of a checkpoint's tracked metadata (best val ppl, step,
    tokens, graduation status) for the load banner — "" when nothing is known."""
    if not meta:
        return ""
    bits = []
    score = meta.get("score", meta.get("gate_score"))
    if isinstance(score, (int, float)):
        bits.append(f"val ppl {score:.2f}")
    if meta.get("step") is not None:
        bits.append(f"step {int(meta['step']):,}")
    toks = meta.get("tokens_seen", meta.get("tokens"))
    if isinstance(toks, (int, float)):
        bits.append(f"{toks / 1e6:.1f}M tok")
    if meta.get("met_bar") is not None:
        bits.append(f"graduated: met_bar={meta['met_bar']}")
    return " · ".join(bits)


def load_model(args):
    cfg = load_config(args.config)
    require_backend(cfg)  # selects the configured backend (mlx | torch)
    B = backend.current()
    precision = get_precision(cfg)

    # Announce what this level can do + guard inference memory against the device.
    from src import resources as R

    R.announce(cfg, mode="infer")
    R.guard(cfg, mode="infer", force=getattr(args, "force", False))

    # Import model modules now that the backend is selected.
    from src.core.model.config import ModelConfig
    from src.core.model.transformer import RDMCAFoundational, set_model_precision

    model_dict = dict(cfg["model"])
    # Sync vocab_size with trained tokenizer if available
    import json

    tok_info = Path("dist/tokenizer/tokenizer_info.json")
    if tok_info.exists():
        # Use the real text vocab (IDs the tokenizer actually emits), NOT the full
        # multimodal layout size — see the same fix in the trainer. Must match the
        # size the checkpoint was trained at, or the embedding/head won't load.
        info = json.loads(tok_info.read_text())
        actual_vocab = info.get("text_vocab_size", info["vocab_size"])
        if actual_vocab != model_dict.get("vocab_size"):
            model_dict["vocab_size"] = actual_vocab

    mcfg = ModelConfig(
        **{k: v for k, v in model_dict.items() if k in ModelConfig.__dataclass_fields__}
    )
    model = RDMCAFoundational(mcfg)

    if args.dummy:
        # Force-init weights with a dummy pass so parameters are allocated
        set_model_precision(model, precision)
        dummy = B.ops.array(np.zeros((1, 2), dtype=np.int64))
        _ = model(dummy)
        B.engine.eval(model.parameters())
        _apply_quant(model, getattr(args, "quant", "none"))
        B.engine.set_eval(model)
        print("  [dummy mode] Random weights — output will be gibberish.")
        print("  Run training first to get meaningful generations.\n")
        return model, mcfg

    # Behavioral stages (tool/MCP/skills) run as the FROZEN cognitive core + the
    # trained LoRA sectors — so language/reasoning stays intact and tool/skill
    # behaviour is added on top. Falls through to a plain checkpoint for cognitive
    # stages (or before any freeze).
    from src.core.model import sector_io
    from src.core.training.curriculum import ckpt_root

    root = ckpt_root(cfg)
    if not args.checkpoint and args.stage:
        label = sector_io.load_for_inference(model, root, args.stage)
        if label:
            print(f"  Loading: {label}")
            set_model_precision(model, precision)
            _apply_quant(model, getattr(args, "quant", "none"))
            B.engine.set_eval(model)
            return model, mcfg

    # Find checkpoint. ALWAYS prefer the BEST (lowest-val-perplexity) checkpoint, and
    # report which one + its tracked quality so the user knows exactly what's running.
    ckpt_path: Path | None = None
    label, meta = "explicit", None
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    elif args.stage:
        ckpt_path, label, meta = resolve_stage_checkpoint(root / f"stage{args.stage}")

    if ckpt_path is None or not ckpt_path.exists():
        stage_hint = args.stage or 1
        print("No checkpoint found. Options:")
        print(
            f"  Train first:  python scripts/train.py --stage {stage_hint} --config {args.config}"
        )
        print("  Or test now:  python uses/chat/run_chat.py --dummy")
        sys.exit(1)

    print(f"  Loading checkpoint [{label}]: {ckpt_path}")
    desc = describe_checkpoint_meta(meta)
    if desc:
        print(f"    └ tracking: {desc}")
    B.engine.load_weights(model, str(ckpt_path))
    set_model_precision(model, precision)  # cast to configured inference precision
    _apply_quant(model, getattr(args, "quant", "none"))  # optional 4-/8-bit
    B.engine.set_eval(model)  # disable dropout for inference
    return model, mcfg


def load_mood_head(model, args, mcfg):
    """Load this stage's mood head via the shared loader (None ⇒ stay neutral)."""
    from src.models.cognition.mood import load_mood_head as _load_mood_head

    head = _load_mood_head(
        mcfg.d_model,
        level=getattr(args, "level", None),
        stage=getattr(args, "stage", None),
        checkpoint=getattr(args, "checkpoint", None),
    )
    if head is not None:
        print("  Mood head: loaded — conversation mood tracking on (neutral default)")
    return head
