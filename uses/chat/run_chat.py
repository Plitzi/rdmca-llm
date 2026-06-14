#!/usr/bin/env python3
from __future__ import annotations

import os

# Auto-bootstrap: re-run with .venv/bin/python if dependencies are not available.
import sys

try:
    import numpy  # noqa: F401 — just checking the venv is active
except ModuleNotFoundError:
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    venv_py = os.path.join(_repo, ".venv", "bin", "python")
    if os.path.exists(venv_py) and os.path.abspath(sys.executable) != os.path.abspath(venv_py):
        os.execv(venv_py, [venv_py, *sys.argv])
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
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on path

import src.backend as backend
from src import agent
from src.config import get_precision, load_config, require_backend

# Model/tokenizer modules are imported lazily inside load_model() — only AFTER
# require_backend() has selected the backend — so model classes bind to it.
# Generation core (sampling, KV-cached decode loop, two-phase <think>/answer)
# lives in src/inference/generate.py so the chat and agent runtimes share it.
# Re-exported here for backward compatibility (tests + run_agent import via run_chat).
from src.inference.generate import (  # noqa: F401  (re-exported for tests + run_agent)
    GEN_DEADLINE_S,
    IncrementalDecoder,
    _looping,
    generate,
    generate_thinking,
    sample_top_p,
)
from src.memory.experience_log import detect_correction, load_experiences, log_experience
from src.modalities.moods import MOODS
from src.modalities.text import BOS_ID
from src.observability import ContextReport, count_tokens
from uses.common.interaction import InterruptGuard, SessionInput

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


def resolve_stage_checkpoint(stage_dir: Path):
    """Pick the checkpoint inference should use for a stage, ALWAYS preferring the BEST
    (lowest val-perplexity) over the latest training step. Returns (path|None, label,
    meta) — meta is the tracked JSON (best.json / stage_complete.json / latest.json) so
    the caller can report the model's quality. Priority:

      1. best.npz   — the running/ratcheted best (the gate's moving bar), meta=best.json;
      2. final.npz  — the graduated model (= the best at graduation);
      3. latest.json — only when no eval-best exists yet (training just started).
    """
    import json as _json

    def _read(p: Path):
        try:
            return _json.loads(p.read_text()) if p.exists() else None
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
    from src.model.config import ModelConfig
    from src.model.transformer import RDMCAFoundational, set_model_precision

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
    from src.model import sector_io

    level = cfg.get("level")
    root = Path("dist/checkpoints") if level is None else Path("dist/checkpoints") / f"level{level}"
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
        level = cfg.get("level")  # NB: level 0 is valid → use `is None`
        root = (
            Path("dist/checkpoints")
            if level is None
            else Path("dist/checkpoints") / f"level{level}"
        )
        ckpt_path, label, meta = resolve_stage_checkpoint(root / f"stage{args.stage}")

    if ckpt_path is None or not ckpt_path.exists():
        stage_hint = args.stage or 1
        print("No checkpoint found. Options:")
        print(f"  Train first:  python train_stage.py --stage {stage_hint} --config {args.config}")
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


# ──────────────────────────────────────────────────────────────────────────────
# Chat loop
# ──────────────────────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════╗
║           RDMCA Interactive Chat                     ║
║  /lang es|en  /temp 0.7  /topp 0.9  /maxtok 256      ║
║  /think off|low|medium|high   /format text|json      ║
║  /system <prompt>   /mood <name|auto|off>            ║
║  /context  (full token/context breakdown)            ║
║  /stream on|off  /stats  /reset  /quit               ║
║  /ok  (last answer was good)   /fix <correct answer> ║
╚══════════════════════════════════════════════════════╝"""


def _load_mood_head(model, args, mcfg):
    """Load this stage's mood head via the shared loader (None ⇒ stay neutral)."""
    from src.model.mood import load_mood_head

    head = load_mood_head(
        mcfg.d_model,
        level=getattr(args, "level", None),
        stage=getattr(args, "stage", None),
        checkpoint=getattr(args, "checkpoint", None),
    )
    if head is not None:
        print("  Mood head: loaded — conversation mood tracking on (neutral default)")
    return head


def chat_loop(model, mcfg, tokenizer, args) -> None:
    print(BANNER)

    # Session state
    lang = args.lang
    temperature = args.temp
    top_p = args.topp
    top_k = getattr(args, "topk", 0)
    rep_penalty = getattr(args, "rep_penalty", 1.0)
    max_tokens = args.maxtok
    out_format = agent.normalize_format(getattr(args, "format", "text"))
    # think=auto: thinking only makes sense once the reasoning stage is trained. When
    # testing a stage BELOW it, an under-trained <think> just emits garbage (e.g. stage 1
    # answering "hi" with a broken scratchpad), so default think OFF there.
    _think_arg = getattr(args, "think", "auto")
    if _think_arg == "auto":
        from src.training.stages import STAGE_NAMES

        # The <think>/CoT stage is the one named exactly "Reasoning" (stage 5) — NOT
        # stage 4 "Causal and procedural reasoning", which also contains the substring.
        reasoning_stage = next(
            (s for s, n in STAGE_NAMES.items() if n.lower().strip() == "reasoning"), 5
        )
        _stage = getattr(args, "stage", None)
        if _stage is not None and _stage < reasoning_stage:
            think_level = agent.normalize_thinking("off")
            print(
                f"  Thinking → off (stage {_stage} is below the reasoning stage "
                f"{reasoning_stage}; use --think to override)"
            )
        else:
            think_level = agent.normalize_thinking("medium")
    else:
        think_level = agent.normalize_thinking(_think_arg)
    stream = bool(getattr(args, "stream", True))
    deadline = getattr(args, "max_seconds", GEN_DEADLINE_S) or None  # 0 → unlimited
    # Seed context with the multimodal grounding prefix (image/audio), if any.
    history: list[int] = list(getattr(args, "mm_prefix", []) or [])
    last_report = None  # ContextReport of the last turn (/context, billing)
    grounding_tokens = len(getattr(args, "mm_prefix", []) or [])  # injected memory/grounding
    # System prompt + conversation mood. The system line opens every generation
    # (BOS + lang + `System: <persona> (mood: …)` + turns), refreshed each turn so
    # a shifting mood is reflected while staying at the front (in-distribution).
    system_text = (getattr(args, "system", None) or "").strip()
    # mood_off: moods disabled entirely → always neutral, focused on answering the
    # question (a calm, direct assistant). Otherwise: a pinned mood, or automatic
    # tracking when a mood head is loaded (neutral by default).
    mood_off = bool(getattr(args, "no_mood", False))
    mood_head = None if mood_off else _load_mood_head(model, args, mcfg)
    mood_pin = getattr(args, "mood", None)  # None ⇒ track automatically
    current_mood = mood_pin if (mood_pin in MOODS and not mood_off) else "neutral"
    mood_auto = (not mood_off) and (mood_pin not in MOODS) and (mood_head is not None)
    # Conversation-aware running mood (memory across turns), neutral by default.
    from src.model.mood import MoodTracker

    mood_tracker = MoodTracker(mood_head)
    if mood_off:
        print("  Mood: off — neutral, focused on answering.")
    # The previous turn, held back until we know its outcome (accepted/corrected/none).
    # We only persist a turn as a learning experience once it has a feedback signal —
    # a turn the user just moved on from is NOT saved (no benefit). See experience_log.
    pending: dict | None = None

    def _resolve_pending(feedback: str, correction: str | None = None) -> None:
        nonlocal pending
        if not pending:
            print("  (no previous answer to mark)")
            return
        wrote = log_experience(
            pending["prompt"],
            pending["response"],
            feedback=feedback,
            correction=correction,
            lang=pending["lang"],
            modality="text",
        )
        if wrote:
            print(f"  ✓ saved as a learning experience ({feedback}).")
        pending = None

    tok_ready = tokenizer.ready
    if not tok_ready:
        print("\n  [tokenizer] Not found — run: python scripts/train_tokenizer.py")
        print("  Using vocab IDs as proxy output (for pipeline testing only).\n")

    # Memory recall (read side): each turn the user message is embedded and the
    # most relevant consolidated (LTSS) + recent (experience-log) memories are
    # injected as a leading <mem>…</mem> block. Lazy/optional — empty stores ⇒ no
    # injection. Needs a real tokenizer (the proxy path can't embed/decode).
    recall = None
    if tok_ready:
        try:
            from src.memory.recall import MemoryRecall

            recall = MemoryRecall(model, tokenizer)
            print("  Memory recall: on (LTSS + experiences, injected as <mem>).")
        except Exception as e:
            print(f"  Memory recall: off ({e}).")

    # Optional STR sector context-slots (§12): route turn chunks to per-sector slots
    # (gated by the trained MoE gate), evict overflow to the episodic buffer, and
    # assemble the active context from the slots instead of a flat truncated window.
    # OPT-IN (--context-slots) and additive — off by default, base path unchanged.
    cm = None
    if getattr(args, "context_slots", False) and tok_ready:
        try:
            from src.routing.context_manager import build_context_manager

            cm = build_context_manager(model, tokenizer, context_len=mcfg.context_len)
            gate_on = getattr(model, "gate", None) is not None
            print(
                f"  Context slots: on (§12 STR; routing via "
                f"{'trained MoE gate' if gate_on else 'classifier/single-slot'})."
            )
        except Exception as e:
            print(f"  Context slots: off ({e}).")

    # Background stdin reader: lets you TYPE WHILE the model generates — those lines
    # QUEUE and are handled on the next turn (so you can correct a reply going wrong),
    # and Ctrl-C during a reply ABORTS just that reply (see InterruptGuard below).
    session = SessionInput()
    print("  [tip] type while it answers to queue a follow-up · Ctrl-C to stop a reply\n")

    while True:
        try:
            queued = session.pending()  # messages typed during the last reply
            hint = f" ({queued} queued)" if queued else ""
            line = session.next_message(f"\n[{lang.upper()}]{hint} You: ")
        except KeyboardInterrupt:
            print("\nBye.")
            break
        if line is None:  # EOF (Ctrl-D / piped input done)
            print("\nBye.")
            break
        prompt = line.strip()

        if not prompt:
            continue

        # ── Special commands ──────────────────────────────────────────────
        if prompt.startswith("/"):
            parts = prompt.split()
            cmd = parts[0].lower()

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

            elif cmd in ("/stats", "/context"):
                if last_report is not None:
                    print(last_report.render())
                else:
                    print("  No generation yet.")

            elif cmd == "/system":
                system_text = prompt[len("/system") :].strip()
                print(
                    f"  System prompt → {system_text!r}"
                    if system_text
                    else "  System prompt cleared."
                )

            elif cmd == "/mood":
                arg = parts[1].lower() if len(parts) > 1 else ""
                if not arg:
                    src = "off" if mood_off else ("auto" if mood_auto else "pinned")
                    print(
                        f"  Mood: {current_mood} ({src}). "
                        f"Set with /mood <{'|'.join(MOODS)}>, /mood auto, or /mood off."
                    )
                elif arg == "off":
                    mood_off, mood_auto, current_mood = True, False, "neutral"
                    print("  Mood → off (always neutral, focused on answering).")
                elif arg == "auto":
                    mh = _load_mood_head(model, args, mcfg) if mood_head is None else mood_head
                    if mh is None:
                        print("  No mood head loaded — mood stays neutral.")
                    else:
                        mood_head, mood_off, mood_auto = mh, False, True
                        mood_tracker.head = mh
                        mood_tracker.reset()
                        print("  Mood → auto (tracked from the conversation).")
                elif arg in MOODS:
                    current_mood, mood_auto, mood_off = arg, False, False
                    print(f"  Mood → {arg} (pinned).")
                else:
                    print(f"  Unknown mood {arg!r}. Choose from: {', '.join(MOODS)}, auto, off.")

            elif cmd == "/reset":
                history.clear()
                if cm is not None:
                    cm.clear()
                mood_tracker.reset()
                if mood_auto:
                    current_mood = "neutral"
                print("  History cleared.")

            elif cmd == "/ok":  # explicit: last answer was good
                _resolve_pending("accepted")

            elif cmd == "/fix":  # explicit: here is the right answer
                correction = prompt[len("/fix") :].strip()
                if not correction:
                    print("  Usage: /fix <the correct answer>")
                else:
                    _resolve_pending("corrected", correction)

            else:
                print(f"  Unknown command: {cmd}")

            continue

        # ── Implicit feedback on the PREVIOUS turn ────────────────────────
        # If this new message reads as a correction, save the previous turn as a
        # `corrected` experience (this message IS the correction) — then still answer
        # it as a normal turn. If it's just a new topic, the previous turn carried no
        # learning signal, so we drop it (silence ≠ acceptance → nothing is saved).
        if pending is not None:
            if detect_correction(prompt):
                _resolve_pending("corrected", prompt)
            else:
                pending = None

        # ── Track the conversation's mood (neutral by default) ────────────
        # A non-neutral mood is only chosen when the exchange clearly carries one
        # (see classify_mood's margin) — otherwise we stay neutral, like a calm
        # assistant. The mood then conditions tone via the System line below.
        if mood_auto and tok_ready and mood_head is not None:
            # The current message supplies the live signal; the tracker's running
            # state carries the WHOLE conversation's mood (memory + decay to neutral).
            # A little recent context disambiguates short/terse messages.
            ctx = tokenizer.decode(history[-120:]) if history else ""
            new_mood = mood_tracker.update(model, tokenizer, prompt, context=ctx)
            if new_mood != current_mood:
                print(f"  (mood → {new_mood})")
            current_mood = new_mood

        # ── Encode prompt (primed for the chosen output format + thinking) ─
        # `history` holds only the User:/Assistant: turn bodies (no BOS). The full
        # generation context is synthesized each turn as BOS + lang + System line
        # (persona + current mood) + history, so the system/mood always leads.
        enc_prompt = agent.wrap_prompt(prompt, out_format, think=think_level)
        if tok_ready:
            new_ids = tokenizer.encode(enc_prompt, lang=lang, add_bos=False, add_eos=False)
        else:
            # Fallback: hash characters to vocab IDs for smoke testing
            new_ids = [ord(c) % mcfg.vocab_size for c in enc_prompt]

        history.extend(new_ids)
        if cm is not None:
            cm.add(new_ids)  # route this turn's chunks to sector slots
        # Trim history to fit context window (leave room for generation + preamble).
        max_hist = max(64, mcfg.context_len - max_tokens - 48)
        if len(history) > max_hist:
            history = history[-max_hist:]

        # Recall relevant memory for THIS message → leading <mem> block (placed
        # right after the System line, before the turns). mem_ids stays [] when
        # there is nothing relevant, so an ordinary turn is unchanged.
        mem_ids: list[int] = []
        if recall is not None:
            try:
                mem_text = recall.as_context(recall.recall(prompt))
                if mem_text:
                    mem_ids = tokenizer.encode(mem_text, lang=lang, add_bos=False, add_eos=False)
            except Exception:
                mem_ids = []

        # Build the leading System preamble (BOS + lang + persona + mood). When
        # there is neither a system prompt nor an active mood this is just BOS+lang.
        # Active context body: the sector-slot union (§12) when context slots are
        # on, else the flat trimmed history (default).
        body = cm.assemble(max_hist) if cm is not None else history
        if tok_ready:
            pre = agent.system_preamble(system_text, current_mood)
            pre_ids = tokenizer.encode(pre, lang=lang, add_bos=True, add_eos=False)
            gen_history = pre_ids + mem_ids + body
        else:
            gen_history = [BOS_ID, *body]

        # ── Generate (two-phase when thinking is on, streamed when asked) ──
        # Thinking needs a real tokenizer (the scratchpad is decoded/re-encoded
        # between phases), so it is disabled on the vocab-ID fallback path.
        # Streaming likewise needs a tokenizer to decode the live deltas.
        budget = agent.think_budget(think_level, max_tokens) if tok_ready else 0
        stream_on = stream and tok_ready
        think_text = ""
        try:
            with InterruptGuard() as guard:  # Ctrl-C aborts THIS reply only
                think_text, gen_ids, tps = generate_thinking(
                    model,
                    list(gen_history),
                    tokenizer=tokenizer,
                    lang=lang,
                    max_new_tokens=max_tokens,
                    think_budget=budget,
                    temperature=temperature,
                    top_p=top_p,
                    vocab_size=mcfg.vocab_size,
                    context_len=mcfg.context_len,
                    stream=stream_on,
                    max_seconds=deadline,
                    think_prefix="\n💭 thinking: ",
                    answer_prefix="\nRDMCA: ",
                    top_k=top_k,
                    rep_penalty=rep_penalty,
                    should_stop=guard.stopped,
                )
            if guard.was_interrupted:
                print("\n  [stopped]")
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
            print("\nRDMCA: ", end="", flush=True)
        result = agent.parse_output(response, out_format)
        if result["format"] == "json":
            if result["valid"]:
                import json as _json

                print(
                    ("\n" if stream_on else "")
                    + _json.dumps(result["json"], ensure_ascii=False, indent=2)
                )
            elif not stream_on:
                print(f"{response}\n  [warning: output is not valid JSON]")
        elif not stream_on:
            print(result["text"])  # role-tag leakage already trimmed
        # ── Context & token accounting (legible now, billing-ready via to_dict) ─
        try:
            mem_files = len(load_experiences())
        except Exception:
            mem_files = 0
        last_report = ContextReport(
            surface="chat",
            context_len=mcfg.context_len,
            system_tokens=count_tokens(tokenizer, agent.system_preamble(system_text, current_mood))
            if tok_ready
            else 0,
            memory_tokens=grounding_tokens + len(mem_ids),  # grounding + recalled <mem>
            history_tokens=len(history),
            tokens_in=len(gen_history),
            tokens_out=len(gen_ids),
            tokens_reasoning=count_tokens(tokenizer, think_text) if tok_ready else 0,
            mood=current_mood,
            mood_dist=mood_tracker.distribution() if mood_auto else None,
            memory_files=mem_files,
            tps=tps,
            params={
                "temp": temperature,
                "top_p": top_p,
                "think": think_level,
                "format": out_format,
                "lang": lang,
                "stream": "on" if stream_on else "off",
            },
        )
        print(last_report.render_compact())

        # Add response to history — store the *cleaned* reply so role-tag leakage
        # doesn't compound across turns (re-encode the trimmed text on the tok path).
        if tok_ready:
            cleaned = agent.clean_answer(response)
            resp_ids = tokenizer.encode(cleaned, lang=lang, add_bos=False, add_eos=False)
            history.extend(resp_ids)
            if cm is not None:
                cm.add(resp_ids)  # the assistant's reply also fills slots
        else:
            history.extend(gen_ids)

        # Hold this turn back as a *candidate* experience: it is saved for daily
        # consolidation ONLY once it earns a learning signal — the user types /ok
        # (accepted), /fix <answer> (corrected), or the next message reads as a
        # correction. A turn the user just moves on from is never saved (no benefit).
        if tok_ready:
            pending = {"prompt": prompt, "response": agent.clean_answer(response), "lang": lang}


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
  python uses/chat/run_chat.py --level 1 --stage 1 --system "You are a kind, simple assistant."
  python uses/chat/run_chat.py --level 1 --stage 1 --no-mood   # always neutral, just answer
        """,
    )
    parser.add_argument("--config", default=None, help="Explicit config path (overrides --level)")
    parser.add_argument(
        "--level", type=int, default=None, help="Educational level 1-5 (which base to chat with)"
    )
    parser.add_argument(
        "--stage", type=int, default=None, help="Load latest checkpoint from this stage"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to a specific .npz checkpoint"
    )
    parser.add_argument(
        "--dummy", action="store_true", help="Use random weights (no checkpoint needed)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Image file to ground the conversation on (multimodal)",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Audio file to ground the conversation on (multimodal)",
    )
    parser.add_argument("--lang", default="en", help="Starting language code (default: en)")
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="System prompt that opens every turn (persona/instructions)",
    )
    parser.add_argument(
        "--mood",
        type=str,
        default=None,
        help=f"Pin the conversation mood ({', '.join(MOODS)}); "
        "omit to track it automatically (neutral default)",
    )
    parser.add_argument(
        "--no-mood",
        dest="no_mood",
        action="store_true",
        help="Disable moods entirely: always neutral, focused on "
        "answering the question (respecting the system prompt)",
    )
    parser.add_argument(
        "--context-slots",
        dest="context_slots",
        action="store_true",
        help="Use STR per-sector context slots (§12): route turn "
        "chunks to sector slots, evict overflow to memory, and "
        "assemble context from the slots (experimental; best with "
        "trained sectors). Off by default (flat history).",
    )
    parser.add_argument(
        "--temp", type=float, default=0.8, help="Sampling temperature (default: 0.8)"
    )
    parser.add_argument("--topp", type=float, default=0.9, help="Nucleus sampling p (default: 0.9)")
    parser.add_argument(
        "--topk", type=int, default=0, help="Top-k sampling cutoff (0 = off, the default)"
    )
    parser.add_argument(
        "--rep-penalty",
        dest="rep_penalty",
        type=float,
        default=1.3,
        help="Repetition penalty over recent tokens (1.0 = off; "
        "default 1.3 curbs the loops small models fall into)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed the sampler RNG for reproducible generations"
    )
    parser.add_argument(
        "--maxtok", type=int, default=256, help="Max new tokens per turn (default: 256)"
    )
    parser.add_argument(
        "--format",
        choices=agent.OUTPUT_FORMATS,
        default="text",
        help="Output format: text (default) or json (structured)",
    )
    parser.add_argument(
        "--think",
        choices=[*agent.THINKING_LEVELS, "auto"],
        default="auto",
        help="Reasoning effort: off, low, medium, high, or auto (default). "
        "auto = off when testing a stage BELOW the reasoning stage "
        "(the model hasn't learned to think yet), medium otherwise.",
    )
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream tokens as they generate (default: on; --no-stream to disable)",
    )
    parser.add_argument(
        "--quant",
        type=parse_quant,
        default=None,
        metavar="none|N",
        help=f"Weight quantization bit-width: none (default) or "
        f"{_QUANT_MIN}-{_QUANT_MAX} bits (e.g. 8, int4). Smaller = less "
        f"memory; 4-bit (≈⅛ size) is the limited-hardware testing tier",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=GEN_DEADLINE_S,
        help=f"Per-generation wall-clock cap, anti-loop guard "
        f"(default {GEN_DEADLINE_S:g}s; 0 = unlimited)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even if the resource guard says it won't fit (risk OOM)",
    )
    args = parser.parse_args()

    if args.seed is not None:  # reproducible sampling across runs
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

    tokenizer = TextTokenizer()

    print(
        f"  d_model={mcfg.d_model} | vocab={mcfg.vocab_size} | "
        f"layers={mcfg.n_layers} | context={mcfg.context_len}"
    )
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
            print(
                f"  Multimodal prefix: {len(args.mm_prefix)} tokens "
                f"({'image ' if args.image else ''}{'audio' if args.audio else ''})"
            )
        except RuntimeError as e:
            print(f"  [multimodal] {e}")
            sys.exit(1)

    chat_loop(model, mcfg, tokenizer, args)


if __name__ == "__main__":
    main()
