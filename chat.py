#!/usr/bin/env python3
from __future__ import annotations
# Auto-bootstrap: re-run with .venv/bin/python if mlx is not available.
import sys, os
try:
    import mlx.core  # noqa: F401 — just checking availability
except ModuleNotFoundError:
    venv_py = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           ".venv", "bin", "python")
    if os.path.exists(venv_py) and os.path.abspath(sys.executable) != os.path.abspath(venv_py):
        os.execv(venv_py, [venv_py] + sys.argv)
    print("ERROR: mlx not found and .venv/bin/python not available.")
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

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.model.transformer import RDMCAFoundational, ModelConfig
from src.modalities.text import TextTokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────────────────────────────────

def sample_top_p(logits: mx.array, temperature: float, top_p: float) -> int:
    if temperature == 0.0:
        return int(mx.argmax(logits).item())
    logits_np = np.array((logits / temperature).tolist(), dtype=np.float32)
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


def generate(model: RDMCAFoundational,
             input_ids: list,
             max_new_tokens: int,
             temperature: float,
             top_p: float,
             vocab_size: int,
             stream: bool = True) -> tuple[list[int], float]:
    """
    Returns (generated_ids, tokens_per_second).
    If stream=True, prints each token as it is generated.
    """
    tokens = mx.array(input_ids)[None]   # [1, S]
    generated: list[int] = []
    t0 = time.perf_counter()

    EOS_ID = 3

    for _ in range(max_new_tokens):
        # Keep context within model limit (last 2048 tokens)
        if tokens.shape[1] > 2048:
            tokens = tokens[:, -2048:]

        logits = model.logits(tokens)     # [1, S, vocab]
        next_logits = logits[0, -1, :]    # [vocab]
        mx.eval(next_logits)

        next_id = sample_top_p(next_logits, temperature, top_p)

        if next_id == EOS_ID:
            break

        generated.append(next_id)
        new_tok = mx.array([[next_id]])
        tokens  = mx.concatenate([tokens, new_tok], axis=1)

        if stream:
            # Flush single token — will be decoded by caller
            print("▌", end="", flush=True)

    elapsed = time.perf_counter() - t0
    tps = len(generated) / elapsed if elapsed > 0 else 0.0
    return generated, tps


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model(args) -> tuple[RDMCAFoundational, ModelConfig]:
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

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
        dummy = mx.array(np.zeros((1, 2), dtype=np.int32))
        _ = model(dummy)
        mx.eval(model.parameters())
        print("  [dummy mode] Random weights — output will be gibberish.")
        print("  Run training first to get meaningful generations.\n")
        return model, mcfg

    # Find checkpoint — priority: final.npz > latest.json
    ckpt_path: Path | None = None
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    elif args.stage:
        import json as _json
        stage_dir = Path(f"dist/checkpoints/stage{args.stage}")

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
    weights = mx.load(str(ckpt_path))
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
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


def chat_loop(model: RDMCAFoundational, mcfg: ModelConfig,
              tokenizer: TextTokenizer, args) -> None:
    print(BANNER)

    # Session state
    lang       = args.lang
    temperature = args.temp
    top_p       = args.topp
    max_tokens  = args.maxtok
    history: list[int] = []    # accumulated token IDs for context
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
        # Trim history to fit context window (keep last 1800 tokens)
        if len(history) > 1800:
            history = history[-1800:]

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
    parser.add_argument("--stage",      type=int, default=None,
                        help="Load latest checkpoint from this stage")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a specific .npz checkpoint")
    parser.add_argument("--dummy",      action="store_true",
                        help="Use random weights (no checkpoint needed)")
    parser.add_argument("--lang",       default="en", choices=["en", "es"],
                        help="Starting language (default: en)")
    parser.add_argument("--temp",       type=float, default=0.8,
                        help="Sampling temperature (default: 0.8)")
    parser.add_argument("--topp",       type=float, default=0.9,
                        help="Nucleus sampling p (default: 0.9)")
    parser.add_argument("--maxtok",     type=int,   default=256,
                        help="Max new tokens per turn (default: 256)")
    args = parser.parse_args()

    if not args.dummy and args.stage is None and args.checkpoint is None:
        print("Specify --stage N, --checkpoint PATH, or --dummy")
        print("Example: python chat.py --dummy")
        sys.exit(1)

    print("Loading model…")
    model, mcfg = load_model(args)
    tokenizer   = TextTokenizer()

    print(f"  d_model={mcfg.d_model} | vocab={mcfg.vocab_size} | "
          f"layers={mcfg.n_layers} | context={mcfg.context_len}")
    print(f"  Tokenizer: {'ready' if tokenizer.ready else 'NOT trained yet'}")

    chat_loop(model, mcfg, tokenizer, args)


if __name__ == "__main__":
    main()
