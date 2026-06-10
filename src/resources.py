"""
Resource estimation and OOM guard for the LEVEL system.

Design principle (RDMCA levels): a level's *size* is determined by the
**information** it teaches (vocabulary, context length, model width/depth), and
its resource consumption follows from that size. The **hardware** does not set
the level — it only limits **how far** (which level) you can actually run.

This module:
  - estimates a level's parameter count and peak memory (training & inference)
    directly from its config, WITHOUT building the model,
  - probes how much memory is actually available on the active backend/device,
  - reports the highest level the current hardware can run, and
  - guards a run: if a level would not fit, it stops *before* training/loading
    with a clear message instead of crashing with an OOM mid-run.

All formulas are intentionally simple and **conservative** (slightly over-
estimate) and are documented inline so the numbers can be reasoned about.
"""
from __future__ import annotations
from typing import Optional

# Headroom: never plan to use more than this fraction of available memory, since
# the OS, framework runtime and fragmentation all need slack. A run estimated
# above (available × SAFETY) is refused unless --force is passed.
SAFETY = 0.85

# Bytes per parameter by precision (weights/grad/activations).
_BYTES = {"fp32": 4, "bf16": 2, "fp16": 2}
# AdamW keeps two optimizer states (m, v); frameworks usually hold them in fp32.
_ADAM_STATE_BYTES = 4


# ─────────────────────────── parameter count ────────────────────────────────
def count_params(model: dict) -> int:
    """Exact-ish parameter count of RDMCAFoundational from its config dims
    (matches src/model/transformer.py), computed without instantiating it.

      embed        = vocab · d
      per layer    = attn(4·d²)  +  SwiGLU(3·d·ffn)  +  2 RMSNorm(2·d)
      final        = ln_f(d)  +  head(vocab · d)
    """
    d     = int(model["d_model"])
    nl    = int(model["n_layers"])
    ffn   = int(model.get("ffn_dim", 4 * d))
    vocab = int(model["vocab_size"])
    per_layer = 4 * d * d + 3 * d * ffn + 2 * d
    return vocab * d + nl * per_layer + d + vocab * d


# ─────────────────────────── memory estimates ───────────────────────────────
def _activation_bytes(model: dict, batch: int, b: int) -> int:
    """Peak activation memory for one forward/backward, dominated by attention
    scores [B,H,S,S], the projection/FFN intermediates, and the vocab logits
    (MRL evaluates the head at each prefix dim). Conservative."""
    d     = int(model["d_model"])
    nl    = int(model["n_layers"])
    h     = int(model["n_heads"])
    ffn   = int(model.get("ffn_dim", 4 * d))
    ctx   = int(model["context_len"])
    vocab = int(model["vocab_size"])
    mrl   = len(model.get("mrl_dims", [d]))

    per_layer = (h * ctx * ctx          # attention score matrix [B,H,S,S]
                 + 6 * ctx * d          # q,k,v,o + residual buffers
                 + 2 * ctx * ffn)       # SwiGLU gate/up intermediates
    logits = mrl * ctx * vocab          # head output at each MRL prefix dim
    return batch * b * (nl * per_layer + logits)


def estimate_train_memory_gb(model: dict, training: dict, precision: str) -> float:
    """Peak training memory (GB): weights + grads + AdamW states + activations."""
    b = _BYTES[precision]
    p = count_params(model)
    batch = int(training.get("batch_size", 1))
    weights = p * b
    grads   = p * b
    adam    = p * _ADAM_STATE_BYTES * 2          # m and v
    acts    = _activation_bytes(model, batch, b)
    return (weights + grads + adam + acts) / 1e9


def estimate_infer_memory_gb(model: dict, precision: str, batch: int = 1) -> float:
    """Peak inference memory (GB): weights + activations + a KV cache."""
    b = _BYTES[precision]
    p = count_params(model)
    d   = int(model["d_model"])
    nl  = int(model["n_layers"])
    ctx = int(model["context_len"])
    weights = p * b
    acts    = _activation_bytes(model, batch, b)
    kv      = batch * ctx * d * 2 * nl * b       # cached keys + values per layer
    return (weights + acts + kv) / 1e9


# ─────────────────────────── available memory ───────────────────────────────
def available_memory_gb() -> float:
    """Memory the active backend can actually use, in GB.

      - PyTorch + CUDA → free VRAM on the current device.
      - MLX, or PyTorch on MPS/CPU → available system RAM (unified memory).
    """
    try:
        import src.backend as backend
        if backend.is_selected() and backend.current().name == "torch":
            import torch
            if torch.cuda.is_available():
                free, _ = torch.cuda.mem_get_info()
                return free / 1e9
    except Exception:
        pass
    import psutil
    return psutil.virtual_memory().available / 1e9


# ─────────────────────────── level helpers ──────────────────────────────────
def estimate_for(cfg: dict, mode: str) -> float:
    """GB a config needs for `mode` ('train' | 'infer')."""
    model = cfg["model"]
    precision = (cfg.get("training", {}) or {}).get("precision", "bf16")
    if mode == "train":
        return estimate_train_memory_gb(model, cfg.get("training", {}) or {}, precision)
    return estimate_infer_memory_gb(model, precision)


def max_runnable_level(mode: str = "train") -> Optional[int]:
    """Highest level whose `mode` estimate fits in available memory (× SAFETY).
    Returns None if even level 1 does not fit. Scans configs/levels/levelN.yaml."""
    from src.config import load_config, level_config_path, available_levels
    budget = available_memory_gb() * SAFETY
    best = None
    for lvl in available_levels():          # data-driven; new levels picked up automatically
        try:
            cfg = load_config(level_config_path(lvl))
        except FileNotFoundError:
            continue
        if estimate_for(cfg, mode) <= budget:
            best = lvl
        else:
            break          # levels are monotonically heavier; stop at first miss
    return best


# ─────────────────────────── guard & announce ───────────────────────────────
def guard(cfg: dict, mode: str = "train", force: bool = False) -> None:
    """Abort before an OOM if `cfg`'s level won't fit. `force` overrides."""
    need = estimate_for(cfg, mode)
    have = available_memory_gb()
    budget = have * SAFETY
    if need <= budget:
        return
    level = cfg.get("level", "?")
    fits = max_runnable_level(mode)
    fits_msg = (f"this machine can run up to level {fits}"
                if fits else "this machine cannot run even level 1")
    msg = (
        f"\n  ✋ Resource guard: level {level} needs ~{need:.1f} GB for {mode}; "
        f"~{have:.1f} GB free, of which ~{budget:.1f} GB is usable "
        f"(after a {int((1-SAFETY)*100)}% safety margin).\n"
        f"     {fits_msg}.\n"
        f"     Pick a lower level, or pass --force to run anyway (risk of OOM)."
    )
    if force:
        print(msg.replace("✋ Resource guard", "⚠️  Resource guard (--force)"))
        print("     Continuing anyway — may crash with out-of-memory.\n")
        return
    print(msg + "\n")
    import sys
    sys.exit(1)


def announce(cfg: dict, mode: str = "train", stage: Optional[int] = None) -> None:
    """Print what this level teaches, from which areas, and its resource use."""
    level = cfg.get("level", "?")
    name  = cfg.get("name", "")
    info  = cfg.get("information", {}) or {}
    need  = estimate_for(cfg, mode)
    have  = available_memory_gb()
    params_m = count_params(cfg["model"]) / 1e6
    precision = (cfg.get("training", {}) or {}).get("precision", "bf16")

    print(f"\n  ── Level {level}: {name} ──")
    if info.get("summary"):
        print(f"  {info['summary']}")
    areas = info.get("areas") or {}
    if stage is not None and str(stage) in {str(k) for k in areas}:
        # show only the current stage's area when training a specific stage
        key = next(k for k in areas if str(k) == str(stage))
        print(f"  Learning now (stage {stage}): {areas[key]}")
    elif areas:
        print("  Learning across areas:")
        for k, v in areas.items():
            print(f"    • {v}")
    print(f"  Model: ~{params_m:.1f}M params | precision {precision} | "
          f"est. {mode} memory ~{need:.1f} GB | available ~{have:.1f} GB")
    moe = cfg.get("moe") or {}
    if moe.get("enabled"):
        print(f"  MoE sectors: {moe.get('experts', '?')} experts, "
              f"top-{moe.get('top_k', '?')} active per token (+ S7 always-on safety) — "
              f"active sector compute stays bounded as knowledge grows")
