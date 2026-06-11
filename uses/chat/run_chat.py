#!/usr/bin/env python3
from __future__ import annotations
# Auto-bootstrap: re-run with .venv/bin/python if dependencies are not available.
import sys, os
try:
    import numpy  # noqa: F401 — just checking the venv is active
except ModuleNotFoundError:
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    venv_py = os.path.join(_repo, ".venv", "bin", "python")
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
  python uses/chat/run_chat.py --level 1 --stage 1
  python uses/chat/run_chat.py --checkpoint dist/checkpoints/level1/stage1/final.npz

  # Sin datos entrenados — pesos random, solo verifica que el pipeline funciona
  python uses/chat/run_chat.py --dummy

Comandos especiales durante el chat:
  /lang es          cambia el idioma de la sesión (en|es)
  /temp 0.7         ajusta temperatura (0.0 = greedy, 1.0 = creativo)
  /topp 0.9         ajusta nucleus sampling p
  /maxtok 256       ajusta máximo de tokens a generar
  /think medium     nivel de razonamiento (off|low|medium|high) — muestra el <think>
  /format text|json formato de salida
  /stream on|off    transmite tokens en vivo (por defecto: on)
  /stats            muestra estadísticas de la última generación
  /reset            borra el historial de la sesión
  /quit  o  Ctrl+C  salir
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root on path

import src.backend as backend
from src import agent
from src.memory.experience_log import log_experience
from src.config import require_backend, get_precision, load_config

# Model/tokenizer modules are imported lazily inside load_model() — only AFTER
# require_backend() has selected the backend — so model classes bind to it.


# ──────────────────────────────────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────────────────────────────────

# Anti-logic-bomb generation guards. Generation is already bounded by
# `max_new_tokens`, but an adversarial prompt can still drive a tiny model into a
# degenerate loop that burns the whole budget (and, with thinking on, stalls the
# turn). These detect that and stop early:
_MAX_TOKEN_REPEAT = 32      # same token N× in a row → degenerate
_CYCLE_MAX_LEN    = 8       # look for a repeating cycle up to this length …
_CYCLE_MIN_REPS   = 5       # … repeated at least this many times → looping
GEN_DEADLINE_S    = 90.0    # default per-generation wall-clock cap (0 = unlimited)


def _looping(generated: list) -> bool:
    """True if the tail of `generated` is a short cycle repeated many times — the
    signature of a stuck 'thinking' loop. O(_CYCLE_MAX_LEN) per call."""
    n = len(generated)
    for c in range(1, _CYCLE_MAX_LEN + 1):
        span = c * _CYCLE_MIN_REPS
        if n < span:
            break
        tail = generated[-span:]
        if all(tail[i] == tail[i % c] for i in range(span)):
            return True
    return False


def sample_top_p(logits, temperature: float, top_p: float, top_k: int = 0,
                 recent_ids=None, rep_penalty: float = 1.0) -> int:
    logits_np = np.asarray(backend.current().ops.to_numpy(logits), dtype=np.float32).copy()
    # Repetition penalty (HF-style): push down logits of recently emitted tokens
    # so the model stops looping ("I'm sorry. I'm sorry. …"). Window-limited so it
    # never blocks tokens that legitimately recur over a longer span.
    if rep_penalty and rep_penalty != 1.0 and recent_ids:
        idx = np.fromiter(set(int(i) for i in recent_ids), dtype=np.int64)
        if idx.size:
            vals = logits_np[idx]
            logits_np[idx] = np.where(vals > 0, vals / rep_penalty, vals * rep_penalty)
    if temperature == 0.0:
        return int(np.argmax(logits_np))
    logits_np = logits_np / temperature
    logits_np -= logits_np.max()
    probs = np.exp(logits_np)
    probs /= probs.sum()
    sorted_idx  = np.argsort(probs)[::-1]
    if top_k and top_k > 0:                       # restrict to the top_k most likely
        sorted_idx = sorted_idx[:top_k]
    sorted_prob = probs[sorted_idx]
    cumulative  = np.cumsum(sorted_prob)
    cutoff      = int(np.searchsorted(cumulative, top_p)) + 1
    top_idx     = sorted_idx[:cutoff]
    top_prob    = probs[top_idx]
    top_prob   /= top_prob.sum()
    return int(np.random.choice(top_idx, p=top_prob))


def generate(model,
             input_ids: list,
             max_new_tokens: int,
             temperature: float,
             top_p: float,
             vocab_size: int,
             context_len: int = 2048,
             stream: bool = True,
             decode_fn=None,
             max_seconds: float | None = None,
             stop_strings: tuple[str, ...] | None = None,
             top_k: int = 0,
             rep_penalty: float = 1.0,
             rep_window: int = 128) -> tuple[list[int], float]:
    """
    Returns (generated_ids, tokens_per_second).

    If stream=True, prints tokens as they are generated. When `decode_fn` is
    given (e.g. tokenizer.decode), it decodes the running output and prints the
    new text delta each step — real token streaming; otherwise it falls back to
    a per-token ▌ marker (used on the no-tokenizer plumbing path).

    Anti-logic-bomb guards (besides the `max_new_tokens` cap): generation also
    stops on a degenerate token loop (`_looping`) and on an optional wall-clock
    deadline (`max_seconds`), so a crafted prompt can't wedge the turn.
    """
    ops = backend.current().ops
    engine = backend.current().engine
    tokens = ops.array(np.asarray([input_ids], dtype=np.int64))   # [1, S]
    generated: list[int] = []
    printed = ""
    repeat_run = 0
    boundary_hit = False        # broke at a turn-boundary leak → don't flush past it
    t0 = time.perf_counter()

    EOS_ID = 3

    for _ in range(max_new_tokens):
        # Keep context within the model's positional limit.
        if tokens.shape[1] > context_len:
            tokens = tokens[:, -context_len:]

        logits = model.logits(tokens)     # [1, S, vocab]
        next_logits = logits[0, -1, :]    # [vocab]
        engine.eval(next_logits)

        next_id = sample_top_p(next_logits, temperature, top_p, top_k=top_k,
                               recent_ids=generated[-rep_window:] if rep_penalty != 1.0 else None,
                               rep_penalty=rep_penalty)

        if next_id == EOS_ID:
            break

        # Anti-loop: a single token repeated many times, or a short repeating
        # cycle, is a stuck generation — stop rather than burn the budget.
        repeat_run = repeat_run + 1 if (generated and next_id == generated[-1]) else 1
        if repeat_run >= _MAX_TOKEN_REPEAT:
            break

        generated.append(next_id)
        new_tok = ops.array(np.asarray([[next_id]], dtype=np.int64))
        tokens  = ops.concatenate([tokens, new_tok], axis=1)

        if _looping(generated):
            break
        if max_seconds is not None and (time.perf_counter() - t0) > max_seconds:
            break

        # Stop-string check (e.g. role-tag turn-boundary leakage). Needs the
        # decoded text, so it runs on the tokenizer path; we print only up to the
        # boundary (in stream mode) and stop before emitting the leaked new turn.
        if stop_strings and decode_fn is not None:
            text = decode_fn(generated)
            cut = agent.first_stop_index(text, stop_strings)
            if cut is not None:
                if stream:
                    sys.stdout.write(text[len(printed):cut]); sys.stdout.flush()
                    printed = text[:cut]
                boundary_hit = True
                break

        if stream:
            if decode_fn is not None:
                text = decode_fn(generated)         # re-decode (subwords join cleanly)
                # Emit only up to the safe boundary — hold back a trailing fragment
                # that could still grow into a role tag (e.g. 'User' → 'User:'), so a
                # forming turn boundary is never half-printed.
                safe = agent.safe_stream_len(text)
                if safe > len(printed):
                    sys.stdout.write(text[len(printed):safe])
                    sys.stdout.flush()
                    printed = text[:safe]
            else:
                print("▌", end="", flush=True)

    # Flush any held-back tail (real text that never became a role tag). Skipped
    # when we stopped at a boundary — everything past it is the leaked next turn.
    if stream and decode_fn is not None and not boundary_hit:
        text = decode_fn(generated)
        if len(text) > len(printed):
            sys.stdout.write(text[len(printed):]); sys.stdout.flush()

    elapsed = time.perf_counter() - t0
    tps = len(generated) / elapsed if elapsed > 0 else 0.0
    return generated, tps


def generate_thinking(model, prompt_ids: list, *, tokenizer, lang: str,
                      max_new_tokens: int, think_budget: int,
                      temperature: float, top_p: float, vocab_size: int,
                      context_len: int, stream: bool = False,
                      think_prefix: str = "", answer_prefix: str = "",
                      max_seconds: float | None = None,
                      answer_stop: tuple[str, ...] | None = agent.ANSWER_STOP_STRINGS,
                      top_k: int = 0, rep_penalty: float = 1.0,
                      ) -> tuple[str, list[int], float]:
    """Two-phase generation: a budget-capped <think> scratchpad, then the answer.

    Returns (think_text, answer_ids, tok_per_s). With think_budget <= 0 it is a
    single plain generation and think_text is "". The scratchpad is force-closed
    when the budget is hit (or trimmed at </think> if the model closes early), so
    the answer is always generated from a well-formed `… </think>` prefix —
    mirroring how Claude bounds extended thinking with a token budget.

    When `stream=True` the scratchpad and answer are printed live (decoded
    incrementally), each preceded by its prefix label; callers then skip their
    own printing of the same content.
    """
    decode_fn = tokenizer.decode
    sgen = dict(temperature=temperature, top_p=top_p, vocab_size=vocab_size,
                context_len=context_len, stream=stream,
                decode_fn=(decode_fn if stream else None), max_seconds=max_seconds,
                top_k=top_k, rep_penalty=rep_penalty)

    def _label(prefix):
        if stream and prefix:
            sys.stdout.write(prefix); sys.stdout.flush()

    if think_budget <= 0:
        _label(answer_prefix)
        ids, tps = generate(model, list(prompt_ids), max_new_tokens=max_new_tokens,
                            stop_strings=answer_stop, **sgen)
        return "", ids, tps

    # Raw pieces only — NO `<lang:XX>` prefix. encode() would inject the language
    # token mid-sequence (the model only saw it at the start), degrading the
    # scratchpad/answer continuation. See TextTokenizer.encode_raw.
    enc = lambda s: tokenizer.encode_raw(s)
    # Phase A — scratchpad. Prime with the opening tag so generation starts inside it.
    # Stop if the model runs into a new turn (a 'User:'/… leak) — reasoning should
    # never cross a turn boundary, same as the answer phase.
    _label(think_prefix)
    think_ids, tps_a = generate(model, list(prompt_ids) + enc(agent.THINK_OPEN),
                                max_new_tokens=think_budget, stop_strings=answer_stop,
                                **sgen)
    think_text = tokenizer.decode(think_ids) if think_ids else ""
    if agent.THINK_CLOSE in think_text:                 # model closed early
        think_text = think_text.split(agent.THINK_CLOSE)[0]
    think_text = think_text.strip()

    # Phase B — answer, from a force-closed scratchpad prefix.
    closed = f"{agent.THINK_OPEN} {think_text} {agent.THINK_CLOSE}\n"
    _label(answer_prefix)
    answer_ids, tps_b = generate(model, list(prompt_ids) + enc(closed),
                                 max_new_tokens=max_new_tokens,
                                 stop_strings=answer_stop, **sgen)
    tps = next((t for t in (tps_b, tps_a) if t), 0.0)   # report a meaningful rate
    return think_text, answer_ids, tps


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

# Quantization is not limited to a fixed menu: both backends do grouped-affine
# weight quantization at any bit-width in this range. 4-bit is just the smallest
# useful tier for limited-hardware testing, not the only option.
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
            f"invalid --quant {value!r}: use 'none' or a bit-width (e.g. 8, int4)")
    if not (_QUANT_MIN <= bits <= _QUANT_MAX):
        raise argparse.ArgumentTypeError(
            f"--quant bit-width {bits} out of range — supported: {_QUANT_MIN}-{_QUANT_MAX}")
    return bits


def _apply_quant(model, quant) -> None:
    """Quantize model weights to a given bit-width for limited hardware; no-op for
    None/'none'. `quant` may be an int bit-width or a raw --quant string (parsed
    here). Real grouped-affine quantization on both backends at any 2–8 bit width
    — see engine.quantize in src/backend/{mlx,torch}_backend.py."""
    bits = parse_quant(quant)
    if bits is None:
        return
    B = backend.current()
    if not hasattr(B.engine, "quantize"):
        print(f"  [quant] backend '{B.name}' has no quantize — staying at float precision")
        return
    print(f"  Quantizing weights → {bits}-bit (limited-hardware mode)")
    B.engine.quantize(model, bits=bits)


def load_model(args):
    cfg = load_config(args.config)
    require_backend(cfg)              # selects the configured backend (mlx | torch)
    B = backend.current()
    precision = get_precision(cfg)

    # Announce what this level can do + guard inference memory against the device.
    from src import resources as R
    R.announce(cfg, mode="infer")
    R.guard(cfg, mode="infer", force=getattr(args, "force", False))

    # Import model modules now that the backend is selected.
    from src.model.transformer import RDMCAFoundational, set_model_precision
    from src.model.config import ModelConfig

    model_dict = dict(cfg["model"])
    # Sync vocab_size with trained tokenizer if available
    import json as _j
    tok_info = Path("dist/tokenizer/tokenizer_info.json")
    if tok_info.exists():
        # Use the real text vocab (IDs the tokenizer actually emits), NOT the full
        # multimodal layout size — see the same fix in train_stage.py. Must match
        # the size the checkpoint was trained at, or the embedding/head won't load.
        _info = _j.loads(tok_info.read_text())
        actual_vocab = _info.get("text_vocab_size", _info["vocab_size"])
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
        _apply_quant(model, getattr(args, "quant", "none"))
        B.engine.set_eval(model)
        print("  [dummy mode] Random weights — output will be gibberish.")
        print("  Run training first to get meaningful generations.\n")
        return model, mcfg

    # Behavioral stages (tool/MCP/skills) run as the FROZEN cognitive core + the
    # trained LoRA sectors — so language/reasoning stays intact and tool/skill
    # behaviour is added on top. Falls through to a plain checkpoint for cognitive
    # stages (or before any freeze).
    from src.model import sector_io
    level = cfg.get("level")
    root  = (Path("dist/checkpoints") if level is None
             else Path("dist/checkpoints") / f"level{level}")
    if not args.checkpoint and args.stage:
        label = sector_io.load_for_inference(model, root, args.stage)
        if label:
            print(f"  Loading: {label}")
            set_model_precision(model, precision)
            _apply_quant(model, getattr(args, "quant", "none"))
            B.engine.set_eval(model)
            return model, mcfg

    # Find checkpoint — priority: final.npz > latest.json
    ckpt_path: Path | None = None
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    elif args.stage:
        import json as _json
        level     = cfg.get("level")                # NB: level 0 is valid → use `is None`
        root      = Path("dist/checkpoints") if level is None else Path("dist/checkpoints") / f"level{level}"
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
        print(f"  Or test now:  python uses/chat/run_chat.py --dummy")
        sys.exit(1)

    print(f"  Loading checkpoint: {ckpt_path}")
    B.engine.load_weights(model, str(ckpt_path))
    set_model_precision(model, precision)    # cast to configured inference precision
    _apply_quant(model, getattr(args, "quant", "none"))   # optional 4-/8-bit
    B.engine.set_eval(model)                 # disable dropout for inference
    return model, mcfg


# ──────────────────────────────────────────────────────────────────────────────
# Chat loop
# ──────────────────────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════╗
║           RDMCA Interactive Chat                     ║
║  /lang es|en  /temp 0.7  /topp 0.9  /maxtok 256     ║
║  /think off|low|medium|high   /format text|json      ║
║  /stream on|off  /stats  /reset  /quit               ║
╚══════════════════════════════════════════════════════╝"""


def chat_loop(model, mcfg, tokenizer, args) -> None:
    print(BANNER)

    # Session state
    lang       = args.lang
    temperature = args.temp
    top_p       = args.topp
    top_k       = getattr(args, "topk", 0)
    rep_penalty = getattr(args, "rep_penalty", 1.0)
    max_tokens  = args.maxtok
    out_format  = agent.normalize_format(getattr(args, "format", "text"))
    think_level = agent.normalize_thinking(getattr(args, "think", "medium"))
    stream      = bool(getattr(args, "stream", True))
    deadline    = getattr(args, "max_seconds", GEN_DEADLINE_S) or None   # 0 → unlimited
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

            elif cmd == "/format" and len(parts) > 1:
                try:
                    out_format = agent.normalize_format(parts[1])
                    print(f"  Output format → {out_format}")
                except ValueError as e:
                    print(f"  {e}")

            elif cmd == "/think" and len(parts) > 1:
                try:
                    think_level = agent.normalize_thinking(parts[1])
                    print(f"  Thinking → {think_level}")
                except ValueError as e:
                    print(f"  {e}")

            elif cmd == "/stream" and len(parts) > 1:
                stream = parts[1].lower() in ("on", "true", "1", "yes")
                print(f"  Streaming → {'on' if stream else 'off'}")

            elif cmd == "/stats":
                if last_stats:
                    print(f"  Tokens generated : {last_stats['n_tokens']}")
                    print(f"  Speed            : {last_stats['tps']:.1f} tok/s")
                    print(f"  Temperature      : {last_stats['temperature']}")
                    print(f"  Top-p            : {last_stats['top_p']}")
                    print(f"  Thinking         : {last_stats['think']}")
                    print(f"  Streaming        : {'on' if stream else 'off'}")
                else:
                    print("  No generation yet.")

            elif cmd == "/reset":
                history.clear()
                print("  History cleared.")

            else:
                print(f"  Unknown command: {cmd}")

            continue

        # ── Encode prompt (primed for the chosen output format + thinking) ─
        enc_prompt = agent.wrap_prompt(prompt, out_format, think=think_level)
        if tok_ready:
            new_ids = tokenizer.encode(enc_prompt, lang=lang,
                                       add_bos=not history, add_eos=False)
        else:
            # Fallback: hash characters to vocab IDs for smoke testing
            new_ids = [2] + [ord(c) % mcfg.vocab_size for c in enc_prompt] + [10]

        history.extend(new_ids)
        # Trim history to fit context window (leave room for generation).
        max_hist = max(64, mcfg.context_len - max_tokens)
        if len(history) > max_hist:
            history = history[-max_hist:]

        # ── Generate (two-phase when thinking is on, streamed when asked) ──
        # Thinking needs a real tokenizer (the scratchpad is decoded/re-encoded
        # between phases), so it is disabled on the vocab-ID fallback path.
        # Streaming likewise needs a tokenizer to decode the live deltas.
        budget    = agent.think_budget(think_level, max_tokens) if tok_ready else 0
        stream_on = stream and tok_ready
        think_text = ""
        try:
            think_text, gen_ids, tps = generate_thinking(
                model, list(history), tokenizer=tokenizer, lang=lang,
                max_new_tokens=max_tokens, think_budget=budget,
                temperature=temperature, top_p=top_p,
                vocab_size=mcfg.vocab_size, context_len=mcfg.context_len,
                stream=stream_on, max_seconds=deadline,
                think_prefix="\n💭 thinking: ", answer_prefix="\nRDMCA: ",
                top_k=top_k, rep_penalty=rep_penalty,
            )
        except Exception as e:
            print(f"\n  [error] {e}")
            continue

        # ── Decode the answer ─────────────────────────────────────────────
        if tok_ready and gen_ids:
            response = tokenizer.decode(gen_ids)
        elif gen_ids:
            response = f"[token IDs — tokenizer not trained]: {gen_ids[:20]}…"
        else:
            response = "(empty response — model generated EOS immediately)"

        # ── Present the answer (text | json) ──────────────────────────────
        # In stream mode the scratchpad + answer were already printed live, so
        # only re-render for JSON (to pretty-print the parsed object).
        if not stream_on:
            if think_text:
                print(f"\n💭 thinking: {think_text}")
            print(f"\nRDMCA: ", end="", flush=True)
        result = agent.parse_output(response, out_format)
        if result["format"] == "json":
            if result["valid"]:
                import json as _json
                print(("\n" if stream_on else "")
                      + _json.dumps(result["json"], ensure_ascii=False, indent=2))
            elif not stream_on:
                print(f"{response}\n  [warning: output is not valid JSON]")
        elif not stream_on:
            print(result["text"])           # role-tag leakage already trimmed
        print(f"\n  [{len(gen_ids)} tokens · {tps:.1f} tok/s · "
              f"temp={temperature} · top_p={top_p} · format={out_format} · "
              f"think={think_level} · stream={'on' if stream_on else 'off'}]")

        # Add response to history — store the *cleaned* reply so role-tag leakage
        # doesn't compound across turns (re-encode the trimmed text on the tok path).
        if tok_ready:
            cleaned = agent.clean_answer(response)
            history.extend(tokenizer.encode(cleaned, lang=lang, add_bos=False, add_eos=False))
        else:
            history.extend(gen_ids)
        last_stats = {
            "n_tokens": len(gen_ids), "tps": tps,
            "temperature": temperature, "top_p": top_p, "think": think_level,
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
  python uses/chat/run_chat.py --dummy                     # test pipeline (random weights)
  python uses/chat/run_chat.py --level 1 --stage 1         # load Stage 1 checkpoint
  python uses/chat/run_chat.py --checkpoint dist/checkpoints/level1/stage3/final.npz
  python uses/chat/run_chat.py --level 1 --stage 1 --lang es --temp 0.8
        """,
    )
    parser.add_argument("--config",     default=None,
                        help="Explicit config path (overrides --level)")
    parser.add_argument("--level",      type=int, default=None,
                        help="Educational level 1-5 (which base to chat with)")
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
    parser.add_argument("--topk",       type=int,   default=0,
                        help="Top-k sampling cutoff (0 = off, the default)")
    parser.add_argument("--rep-penalty", dest="rep_penalty", type=float, default=1.3,
                        help="Repetition penalty over recent tokens (1.0 = off; "
                             "default 1.3 curbs the loops small models fall into)")
    parser.add_argument("--seed",       type=int, default=None,
                        help="Seed the sampler RNG for reproducible generations")
    parser.add_argument("--maxtok",     type=int,   default=256,
                        help="Max new tokens per turn (default: 256)")
    parser.add_argument("--format",     choices=agent.OUTPUT_FORMATS, default="text",
                        help="Output format: text (default) or json (structured)")
    parser.add_argument("--think",      choices=agent.THINKING_LEVELS, default="medium",
                        help="Reasoning effort: off, low, medium (default), high — "
                             "more thinking generally means better answers")
    parser.add_argument("--stream",     action=argparse.BooleanOptionalAction, default=True,
                        help="Stream tokens as they generate (default: on; --no-stream to disable)")
    parser.add_argument("--quant",       type=parse_quant, default=None, metavar="none|N",
                        help=f"Weight quantization bit-width: none (default) or "
                             f"{_QUANT_MIN}-{_QUANT_MAX} bits (e.g. 8, int4). Smaller = less "
                             f"memory; 4-bit (≈⅛ size) is the limited-hardware testing tier")
    parser.add_argument("--max-seconds", type=float, default=GEN_DEADLINE_S,
                        help=f"Per-generation wall-clock cap, anti-loop guard "
                             f"(default {GEN_DEADLINE_S:g}s; 0 = unlimited)")
    parser.add_argument("--force",      action="store_true",
                        help="Run even if the resource guard says it won't fit (risk OOM)")
    args = parser.parse_args()

    if args.seed is not None:           # reproducible sampling across runs
        np.random.seed(args.seed)

    from src.config import resolve_config_path
    args.config = resolve_config_path(args.config, args.level)

    if not args.dummy and args.stage is None and args.checkpoint is None:
        print("Specify --stage N, --checkpoint PATH, or --dummy")
        print("Example: python uses/chat/run_chat.py --dummy")
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
