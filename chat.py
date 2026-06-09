#!/usr/bin/env python3
from __future__ import annotations
# Auto-bootstrap: re-run with .venv/bin/python if dependencies are not available.
import sys, os
try:
    import numpy  # noqa: F401 — just checking the venv is active
except ModuleNotFoundError:
    venv_py = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           ".venv", "bin", "python")
    if os.path.exists(venv_py) and os.path.abspath(sys.executable) != os.path.abspath(venv_py):
        os.execv(venv_py, [venv_py] + sys.argv)
    print("ERROR: dependencies not found and .venv/bin/python not available.")
    print("Run:  source .venv/bin/activate   (or follow README setup)")
    sys.exit(1)

"""
RDMCA Interactive Chat
======================
Carga un checkpoint y permite conversar con el modelo para evaluar
coherencia, gramática, razonamiento, etc.

Uso:
  # Con checkpoint entrenado (Stage N completado)
  python chat.py --stage 1
  python chat.py --checkpoint dist/checkpoints/stage1/final.npz

  # Sin datos entrenados — pesos random, solo verifica que el pipeline funciona
  python chat.py --dummy

Comandos especiales durante el chat:
  /lang es          cambia el idioma de la sesión (en|es)
  /temp 0.7         ajusta temperatura (0.0 = greedy, 1.0 = creativo)
  /topp 0.9         ajusta nucleus sampling p
  /maxtok 256       ajusta máximo de tokens a generar
  /stats            muestra estadísticas de la última generación
  /reset            borra el historial de la sesión
  /quit  o  Ctrl+C  salir
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import src.backend as backend
from src.memory.experience_log import log_experience
from src.config import require_backend, get_precision, load_config

# Model/tokenizer modules are imported lazily inside load_model() — only AFTER
# require_backend() has selected the backend — so model classes bind to it.


# ──────────────────────────────────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────────────────────────────────

def sample_top_p(logits, temperature: float, top_p: float) -> int:
    logits_np = np.asarray(backend.current().ops.to_numpy(logits), dtype=np.float32)
    if temperature == 0.0:
        return int(np.argmax(logits_np))
    logits_np = logits_np / temperature
    logits_np -= logits_np.max()
    probs = np.exp(logits_np)
    probs /= probs.sum()
    sorted_idx  = np.argsort(probs)[::-1]
    sorted_prob = probs[sorted_idx]
    cumulative  = np.cumsum(sorted_prob)
    cutoff      = int(np.searchsorted(cumulative, top_p)) + 1
    top_idx     = sorted_idx[:cutoff]
    top_prob    = sorted_prob[:cutoff]
    top_prob   /= top_prob.sum()
    return int(np.random.choice(top_idx, p=top_prob))


def generate(model,
             input_ids: list,
             max_new_tokens: int,
             temperature: float,
             top_p: float,
             vocab_size: int,
             context_len: int = 2048,
             stream: bool = True) -> tuple[list[int], float]:
    """
    Returns (generated_ids, tokens_per_second).
    If stream=True, prints each token as it is generated.
    """
    ops = backend.current().ops
    engine = backend.current().engine
    tokens = ops.array(np.asarray([input_ids], dtype=np.int64))   # [1, S]
    generated: list[int] = []
    t0 = time.perf_counter()

    EOS_ID = 3

    for _ in range(max_new_tokens):
        # Keep context within the model's positional limit.
        if tokens.shape[1] > context_len:
            tokens = tokens[:, -context_len:]

        logits = model.logits(tokens)     # [1, S, vocab]
        next_logits = logits[0, -1, :]    # [vocab]
        engine.eval(next_logits)

        next_id = sample_top_p(next_logits, temperature, top_p)

        if next_id == EOS_ID:
            break

        generated.append(next_id)
        new_tok = ops.array(np.asarray([[next_id]], dtype=np.int64))
        tokens  = ops.concatenate([tokens, new_tok], axis=1)

        if stream:
            # Flush single token — will be decoded by caller
            print("▌", end="", flush=True)

    elapsed = time.perf_counter() - t0
    tps = len(generated) / elapsed if elapsed > 0 else 0.0
    return generated, tps


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model(args):
    cfg = load_config(args.config)
    require_backend(cfg)              # selects the configured backend (mlx | torch)
    B = backend.current()
    precision = get_precision(cfg)

    # Import model modules now that the backend is selected.
    from src.model.transformer import RDMCAFoundational, set_model_precision
    from src.model.config import ModelConfig

    model_dict = dict(cfg["model"])
    # Sync vocab_size with trained tokenizer if available
    import json as _j
    tok_info = Path("dist/tokenizer/tokenizer_info.json")
    if tok_info.exists():
        actual_vocab = _j.loads(tok_info.read_text())["vocab_size"]
        if actual_vocab != model_dict.get("vocab_size"):
            model_dict["vocab_size"] = actual_vocab

    mcfg = ModelConfig(**{k: v for k, v in model_dict.items()
                          if k in ModelConfig.__dataclass_fields__})
    model = RDMCAFoundational(mcfg)

    if args.dummy:
        # Force-init weights with a dummy pass so parameters are allocated
        set_model_precision(model, precision)
        dummy = B.ops.array(np.zeros((1, 2), dtype=np.int64))
        _ = model(dummy)
        B.engine.eval(model.parameters())
        B.engine.set_eval(model)
        print("  [dummy mode] Random weights — output will be gibberish.")
        print("  Run training first to get meaningful generations.\n")
        return model, mcfg

    # Find checkpoint — priority: final.npz > latest.json
    ckpt_path: Path | None = None
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    elif args.stage:
        import json as _json
        profile   = cfg.get("profile")
        root      = Path("dist/checkpoints") / profile if profile else Path("dist/checkpoints")
        stage_dir = root / f"stage{args.stage}"

        def _resolve_json(p: Path) -> Path | None:
            """Read a JSON state file and return the .npz path it points to."""
            state = _json.loads(p.read_text())
            target = Path(state["checkpoint"]) if "checkpoint" in state else None
            return target if (target and target.exists()) else None

        # 1. Prefer final.npz (gate passed)
        final = stage_dir / "final.npz"
        if final.exists():
            ckpt_path = final

        # 2. Fall back to latest checkpoint from latest.json
        if ckpt_path is None:
            latest_json = stage_dir / "latest.json"
            if latest_json.exists():
                ckpt_path = _resolve_json(latest_json)

    if ckpt_path is None or not ckpt_path.exists():
        stage_hint = args.stage or 1
        print(f"No checkpoint found. Options:")
        print(f"  Train first:  python train_stage.py --stage {stage_hint} --config {args.config}")
        print(f"  Or test now:  python chat.py --dummy")
        sys.exit(1)

    print(f"  Loading checkpoint: {ckpt_path}")
    B.engine.load_weights(model, str(ckpt_path))
    set_model_precision(model, precision)    # cast to configured inference precision
    B.engine.set_eval(model)                 # disable dropout for inference
    return model, mcfg


# ──────────────────────────────────────────────────────────────────────────────
# Chat loop
# ──────────────────────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════╗
║           RDMCA Interactive Chat                     ║
║  /lang es|en  /temp 0.7  /topp 0.9  /maxtok 256     ║
║  /stats       /reset     /quit                       ║
╚══════════════════════════════════════════════════════╝"""


def chat_loop(model, mcfg, tokenizer, args) -> None:
    print(BANNER)

    # Session state
    lang       = args.lang
    temperature = args.temp
    top_p       = args.topp
    max_tokens  = args.maxtok
    # Seed context with the multimodal grounding prefix (image/audio), if any.
    history: list[int] = list(getattr(args, "mm_prefix", []) or [])
    last_stats: dict   = {}

    tok_ready = tokenizer.ready
    if not tok_ready:
        print("\n  [tokenizer] Not found — run: python scripts/train_tokenizer.py")
        print("  Using vocab IDs as proxy output (for pipeline testing only).\n")

    while True:
        try:
            prompt = input(f"\n[{lang.upper()}] You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not prompt:
            continue

        # ── Special commands ──────────────────────────────────────────────
        if prompt.startswith("/"):
            parts = prompt.split()
            cmd   = parts[0].lower()

            if cmd == "/quit":
                print("Bye.")
                break

            elif cmd == "/lang" and len(parts) > 1:
                lang = parts[1].lower()
                print(f"  Language → {lang.upper()}")

            elif cmd == "/temp" and len(parts) > 1:
                temperature = float(parts[1])
                print(f"  Temperature → {temperature}")

            elif cmd == "/topp" and len(parts) > 1:
                top_p = float(parts[1])
                print(f"  Top-p → {top_p}")

            elif cmd == "/maxtok" and len(parts) > 1:
                max_tokens = int(parts[1])
                print(f"  Max tokens → {max_tokens}")

            elif cmd == "/stats":
                if last_stats:
                    print(f"  Tokens generated : {last_stats['n_tokens']}")
                    print(f"  Speed            : {last_stats['tps']:.1f} tok/s")
                    print(f"  Temperature      : {last_stats['temperature']}")
                    print(f"  Top-p            : {last_stats['top_p']}")
                else:
                    print("  No generation yet.")

            elif cmd == "/reset":
                history.clear()
                print("  History cleared.")

            else:
                print(f"  Unknown command: {cmd}")

            continue

        # ── Encode prompt ─────────────────────────────────────────────────
        if tok_ready:
            new_ids = tokenizer.encode(prompt, lang=lang,
                                       add_bos=not history, add_eos=False)
        else:
            # Fallback: hash characters to vocab IDs for smoke testing
            new_ids = [2] + [ord(c) % mcfg.vocab_size for c in prompt] + [10]

        history.extend(new_ids)
        # Trim history to fit context window (leave room for generation).
        max_hist = max(64, mcfg.context_len - max_tokens)
        if len(history) > max_hist:
            history = history[-max_hist:]

        # ── Generate ──────────────────────────────────────────────────────
        print(f"\nRDMCA: ", end="", flush=True)
        t_start = time.perf_counter()

        try:
            gen_ids, tps = generate(
                model, list(history),
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                vocab_size=mcfg.vocab_size,
                context_len=mcfg.context_len,
                stream=False,   # decode full response at once
            )
        except Exception as e:
            print(f"\n  [error] {e}")
            continue

        # ── Decode ────────────────────────────────────────────────────────
        if tok_ready and gen_ids:
            response = tokenizer.decode(gen_ids)
        elif gen_ids:
            # Fallback: show token IDs
            response = f"[token IDs — tokenizer not trained]: {gen_ids[:20]}…"
        else:
            response = "(empty response — model generated EOS immediately)"

        print(response)
        print(f"\n  [{len(gen_ids)} tokens · {tps:.1f} tok/s · "
              f"temp={temperature} · top_p={top_p}]")

        # Add response to history
        history.extend(gen_ids)
        last_stats = {
            "n_tokens": len(gen_ids), "tps": tps,
            "temperature": temperature, "top_p": top_p,
        }

        # Record the interaction as an experience for daily consolidation.
        if tok_ready:
            log_experience(prompt, lang=lang, modality="text")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RDMCA Interactive Chat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python chat.py --dummy                         # test pipeline (random weights)
  python chat.py --stage 1                       # load Stage 1 checkpoint
  python chat.py --checkpoint dist/checkpoints/stage3/final.npz
  python chat.py --stage 1 --lang es --temp 0.8
        """,
    )
    parser.add_argument("--config",     default="configs/rdmca_t2.yaml")
    parser.add_argument("--profile",    type=str, default=None,
                        help="Hardware profile: nano | m2max | a100 | cluster")
    parser.add_argument("--stage",      type=int, default=None,
                        help="Load latest checkpoint from this stage")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a specific .npz checkpoint")
    parser.add_argument("--dummy",      action="store_true",
                        help="Use random weights (no checkpoint needed)")
    parser.add_argument("--image",      type=str, default=None,
                        help="Image file to ground the conversation on (multimodal)")
    parser.add_argument("--audio",      type=str, default=None,
                        help="Audio file to ground the conversation on (multimodal)")
    parser.add_argument("--lang",       default="en",
                        help="Starting language code (default: en)")
    parser.add_argument("--temp",       type=float, default=0.8,
                        help="Sampling temperature (default: 0.8)")
    parser.add_argument("--topp",       type=float, default=0.9,
                        help="Nucleus sampling p (default: 0.9)")
    parser.add_argument("--maxtok",     type=int,   default=256,
                        help="Max new tokens per turn (default: 256)")
    args = parser.parse_args()

    if args.profile:
        args.config = f"configs/profiles/{args.profile}.yaml"

    if not args.dummy and args.stage is None and args.checkpoint is None:
        print("Specify --stage N, --checkpoint PATH, or --dummy")
        print("Example: python chat.py --dummy")
        sys.exit(1)

    print("Loading model…")
    model, mcfg = load_model(args)
    from src.modalities.text import TextTokenizer
    tokenizer   = TextTokenizer()

    print(f"  d_model={mcfg.d_model} | vocab={mcfg.vocab_size} | "
          f"layers={mcfg.n_layers} | context={mcfg.context_len}")
    print(f"  Tokenizer: {'ready' if tokenizer.ready else 'NOT trained yet'}")

    # Optional multimodal grounding prefix (image/audio) via the perception layer.
    args.mm_prefix = []
    if args.image or args.audio:
        from src.modalities.perception import MultimodalPerception
        mpl = MultimodalPerception(text_tok=tokenizer)
        segments = []
        if args.image:
            segments.append(("image", args.image))
        if args.audio:
            segments.append(("audio", args.audio))
        try:
            args.mm_prefix = mpl.build_sequence(segments)
            print(f"  Multimodal prefix: {len(args.mm_prefix)} tokens "
                  f"({'image ' if args.image else ''}{'audio' if args.audio else ''})")
        except RuntimeError as e:
            print(f"  [multimodal] {e}")
            sys.exit(1)

    chat_loop(model, mcfg, tokenizer, args)


if __name__ == "__main__":
    main()
