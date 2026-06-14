#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

_venv = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv", "bin", "python"
)
if os.path.exists(_venv) and os.path.abspath(sys.executable) != os.path.abspath(_venv):
    os.execv(_venv, [_venv, *sys.argv])

"""
Train SentencePiece BPE tokenizer — Bilingual EN + ES.
Must be run AFTER prepare_data.py --stage 1.

Output: dist/tokenizer/rdmca_spm.model  +  dist/tokenizer/rdmca_spm.vocab

Usage:
  python scripts/train_tokenizer.py
  python scripts/train_tokenizer.py --vocab_size 65536 --sample_mb 500
"""
import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from collections import deque

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from src.config import get_languages, load_config, resolve_config_path
from src.modalities.vocab import (
    CONTROL_SPECIALS,
    build_modality_layout,
    tokenizer_symbols,
)

console = Console()

progress = Progress(
    SpinnerColumn(),
    TextColumn("[bold]{task.description}"),
    BarColumn(bar_width=34),
    TextColumn("[dim]{task.fields[info]}[/dim]"),
    TimeElapsedColumn(),
    console=console,
)

# Last N lines captured from the SentencePiece subprocess output
spm_lines: deque = deque(maxlen=5)

live = Live(console=console, refresh_per_second=10)


def _renderable():
    """Combined renderable: progress bars + last SPM output lines."""
    parts = [progress]
    if spm_lines:
        parts.append(
            Panel(
                "\n".join(spm_lines),
                title="[dim]SPM output[/dim]",
                border_style="dim",
                padding=(0, 1),
            )
        )
    return Group(*parts)


# ──────────────────────────────────────────────────────────────────────────────
# Text sample builder with rich progress bar
# ──────────────────────────────────────────────────────────────────────────────


def build_text_sample(
    files: list,
    label: str,
    out_txt: str,
    max_mb: int,
    task_id,
    lang_filter: str | None = None,
    default_lang: str = "en",
) -> int:
    """Read the given JSONL `files` and write plain text to out_txt, updating shared
    progress. Returns the number of CHARACTERS written.

    When `lang_filter` is set, only records whose `lang` matches are written (a
    record with no `lang` counts as `default_lang`). This routes a mixed-language
    corpus into per-language samples by the record tag — the files themselves are
    not split by language (prepare_data writes one file per SOURCE, not per lang)."""
    max_bytes = max_mb * 1024 * 1024
    jsonl_files = sorted(files)
    total_size = sum(f.stat().st_size for f in jsonl_files)
    bar_total = min(total_size, max_bytes)
    progress.update(task_id, total=bar_total)

    written = 0
    docs = 0
    with open(out_txt, "w", encoding="utf-8") as out:
        for jsonl in jsonl_files:
            with open(jsonl, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if lang_filter is not None and rec.get("lang", default_lang) != lang_filter:
                        continue
                    text = rec.get("text", "").strip()
                    if not text:
                        continue
                    out.write(text + "\n")
                    written += len(text)
                    docs += 1
                    done_mb = written / 1_000_000
                    total_mb = min(max_bytes, total_size) / 1_000_000
                    progress.update(
                        task_id,
                        completed=min(written, max_bytes),
                        info=f"{done_mb:.1f} / {total_mb:.0f} MB  •  {docs:,} docs",
                    )
                    if written >= max_bytes:
                        progress.update(task_id, completed=bar_total)
                        return written
    # All input consumed before the cap: the extracted TEXT is smaller than the raw
    # JSONL bytes (JSON wrapper + UTF-8 multibyte chars), so `written` never reaches
    # the file-size-based total. Snap the bar to 100% so the task completes and its
    # spinner stops — the read really is done.
    progress.update(
        task_id, completed=bar_total, info=f"{written / 1_000_000:.1f} MB  •  {docs:,} docs"
    )
    return written


# ──────────────────────────────────────────────────────────────────────────────
# Spinner for the blocking SentencePiece training call
# ──────────────────────────────────────────────────────────────────────────────


def train_spm(
    combined_input: str, prefix: str, vocab_size: int, langs: list, num_threads: int
) -> None:
    """
    Run SentencePiece in a subprocess (avoids GIL freeze).
    Captures stdout+stderr and shows the last 5 lines in the live panel.

    User-defined symbols are derived from the configured languages
    (`<lang:XX>`) plus the multimodal boundary tokens, so the vocabulary always
    matches the project's language selection.
    """
    import json as _json
    import queue
    import subprocess
    import tempfile
    import threading

    # Single source of truth (src/modalities/vocab.py): per-language tags +
    # modality boundaries + every stage's control delimiters (<think>, <tool_call>,
    # …). Registered as user-defined symbols so they tokenize atomically instead of
    # BPE-splitting into combos the corpus never contains. New stage marker → add it
    # to vocab.CONTROL_SPECIALS and every level's tokenizer picks it up.
    user_symbols = tokenizer_symbols(langs)
    params = {
        "input": combined_input,
        "model_prefix": prefix,
        "vocab_size": vocab_size,
        # character_coverage=1.0 + byte_fallback: EVERY character is representable —
        # rare/unseen symbols (and anything in future messier corpora) decompose into
        # UTF-8 byte tokens instead of collapsing to <unk>. No <unk> entropy spikes,
        # and the model degrades gracefully on out-of-distribution input rather than
        # going blind. (The 256 byte pieces cost a sliver of vocab; well worth it.)
        "character_coverage": 1.0,
        "model_type": "bpe",
        "byte_fallback": True,
        "pad_id": 0,
        "unk_id": 1,
        "bos_id": 2,
        "eos_id": 3,
        "user_defined_symbols": user_symbols,
        "num_threads": num_threads,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        _json.dump(params, f)
        params_file = f.name

    script = (
        "import json, sentencepiece as spm\n"
        f"p = json.load(open({params_file!r}))\n"
        "spm.SentencePieceTrainer.Train(**p)\n"
        f"import os; os.unlink({params_file!r})\n"
    )

    lang_label = "+".join(l.upper() for l in langs)
    task = progress.add_task(
        f"Training BPE  vocab={vocab_size}  {lang_label}  {num_threads} threads",
        total=None,
        info="",
    )

    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Read subprocess output in a thread; push lines to a queue
    line_q: queue.Queue = queue.Queue()

    def _reader():
        for line in proc.stdout:
            line_q.put(line.rstrip())
        line_q.put(None)  # sentinel

    threading.Thread(target=_reader, daemon=True).start()

    # Main loop: drain queue, update live panel
    sentinel_seen = False
    while proc.poll() is None or not sentinel_seen:
        try:
            line = line_q.get(timeout=0.05)
            if line is None:
                sentinel_seen = True
            elif line:
                spm_lines.append(line)
                live.update(_renderable())
        except queue.Empty:
            pass

    if proc.returncode != 0:
        raise RuntimeError(f"SentencePiece training failed (exit {proc.returncode})")

    progress.update(task, description="[green]BPE trained ✓[/green]", info="")
    live.update(_renderable())


# ──────────────────────────────────────────────────────────────────────────────
# Summary panel
# ──────────────────────────────────────────────────────────────────────────────


def show_summary(
    prefix: str,
    vocab_size: int,
    unified_size: int,
    langs: list,
    output_dir: str,
    tests: list[tuple],
) -> None:
    model_size = Path(prefix + ".model").stat().st_size / 1024

    tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    tbl.add_column("key", style="bold cyan", no_wrap=True, width=18)
    tbl.add_column("value", style="white")

    tbl.add_row("Output", str(Path(output_dir)))
    tbl.add_row("Model size", f"{model_size:.0f} KB")
    tbl.add_row("Text vocab", str(vocab_size))
    tbl.add_row("Unified vocab", f"{unified_size}  (text + image + audio)")
    tbl.add_row("Languages", " + ".join(l.upper() for l in langs))

    tbl.add_row("", "")
    tbl.add_row("Verification", "")
    for lang, text, n_tokens, ok in tests:
        status = "[green]OK[/green]" if ok else "[red]MISMATCH[/red]"
        tbl.add_row(f"  [{lang}]", f"{text[:45]}…  → {n_tokens} tokens  {status}")

    console.print(
        Panel(tbl, title="[bold]Tokenizer trained[/bold]", border_style="green", padding=(0, 1))
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

# ── Multimodal VQ-VAE tokenizers (image + audio) ──────────────────────────────
# Folded into this one script so a single command trains every tokenizer the model
# needs. Each modality is OPTIONAL: with no data for it, it is SKIPPED (we move on
# to the next) rather than failing — text is the only required tokenizer.
AUDIO_OUT = "dist/tokenizer/audio_vqvae.npz"
IMAGE_OUT = "dist/tokenizer/image_vqvae.npz"
AUDIO_SR = 16_000  # mirror of audio.SAMPLE_RATE
AUDIO_CLIP = 1.0  # seconds
IMG_SIZE = 32  # CIFAR-scale default


def _vqvae_train_loop(B, console, model, sample_batch, steps, batch, lr, out, label):
    """Shared VQ-VAE training loop: `sample_batch(idx_size)` returns one backend
    input tensor; trains `steps` steps and saves to `out`."""
    B.engine.set_precision(model, "fp32")
    opt = B.engine.make_optimizer(model, lr=lr, weight_decay=0.0)
    lg = B.engine.value_and_grad(model, lambda m, x: m.loss(x))
    console.print(f"  Training {label}: {steps} steps (batch {batch}) …")
    t0 = time.time()
    for step in range(1, steps + 1):
        x = sample_batch(batch)
        loss, grads = lg(model, x)
        B.engine.optimizer_step(opt, model, grads)
        if step % 200 == 0 or step == 1:
            console.print(
                f"    step {step:5d} | loss {B.engine.item(loss):.4f} | "
                f"{step / (time.time() - t0):.1f} it/s"
            )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    console.print(f"  [green]Saved {label} → {out}[/green]")


def train_image_tokenizer(
    console, images_dir, dataset, n, img_size, steps, batch, lr, out, backend_name=None
) -> bool:
    """Train the image VQ-VAE if image data is available; else SKIP (return False).
    Data: a directory of images (`images_dir`) or a HF dataset (`dataset`). With
    neither, returns False so the caller continues to the next modality."""
    import numpy as np

    from src.modalities.image import _resize

    arrs = []
    if images_dir and Path(images_dir).is_dir():
        from PIL import Image

        paths = [
            p
            for p in Path(images_dir).rglob("*")
            if p.suffix.lower()
            in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".gif")
        ]
        for p in paths[:n]:
            arrs.append(
                _resize(
                    np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0, img_size
                )
            )
    elif dataset:
        try:
            from datasets import load_dataset

            ds = load_dataset(dataset, split="train", streaming=True)
            key = None
            for ex in ds:
                if key is None:
                    key = "img" if "img" in ex else ("image" if "image" in ex else None)
                    if key is None:
                        console.print(
                            f"  [yellow]image: dataset '{dataset}' has no "
                            f"img/image field — skipping[/yellow]"
                        )
                        return False
                img = np.asarray(ex[key].convert("RGB"), dtype=np.float32) / 255.0
                arrs.append(_resize(img, img_size))
                if len(arrs) >= n:
                    break
        except Exception as e:
            console.print(f"  [yellow]image: dataset unavailable ({e}) — skipping[/yellow]")
            return False
    if not arrs:
        console.print(
            "  [dim]image: no data (pass --images-dir or --image-dataset) — skipping[/dim]"
        )
        return False

    import src.backend as backend

    if backend_name:
        backend.select(backend_name)
    B = backend.current()
    from src.modalities.image import ImageVQVAE

    data = np.stack(arrs, axis=0).astype(np.float32)  # [N,H,W,3]
    console.print(f"  image: {data.shape[0]} images at {img_size}×{img_size}")
    model = ImageVQVAE(img_size=img_size)

    def _sample(bs):
        idx = np.random.randint(0, data.shape[0], size=bs)
        return B.ops.array(np.transpose(data[idx], (0, 3, 1, 2)))  # NHWC→NCHW

    _vqvae_train_loop(B, console, model, _sample, steps, batch, lr, out, "image VQ-VAE")
    ids = model.encode_ids(data[0])
    console.print(f"  image round-trip: {len(ids)} tokens/image")
    return True


def train_audio_tokenizer(
    console, audio_dir, synthetic, n, steps, batch, lr, out, backend_name=None
) -> bool:
    """Train the audio VQ-VAE if audio data is available; else SKIP (return False).
    Data: a directory of audio clips (`audio_dir`), or a synthetic tone corpus when
    `synthetic` is set (offline smoke test). With neither, returns False."""
    import numpy as np

    clips = []
    if audio_dir and Path(audio_dir).is_dir():
        from src.modalities.perception import load_audio

        paths = [
            p
            for p in Path(audio_dir).rglob("*")
            if p.suffix.lower() in (".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif", ".m4a")
        ]
        clips = [load_audio(p, AUDIO_SR) for p in paths[:n]]
    elif synthetic:
        t = np.linspace(0, AUDIO_CLIP, int(AUDIO_SR * AUDIO_CLIP), endpoint=False)
        for _ in range(n):
            f1, f2 = np.random.uniform(110, 880, size=2)
            wav = (
                0.5 * np.sin(2 * np.pi * f1 * t)
                + 0.3 * np.sin(2 * np.pi * f2 * t)
                + 0.05 * np.random.randn(t.size)
            )
            clips.append(wav.astype(np.float32))
    if not clips:
        console.print(
            "  [dim]audio: no data (pass --audio-dir or --audio-synthetic) — skipping[/dim]"
        )
        return False

    import src.backend as backend

    if backend_name:
        backend.select(backend_name)
    B = backend.current()
    from src.modalities.audio import AudioVQVAE, logmel

    console.print(f"  audio: {len(clips)} clips | sr={AUDIO_SR}")
    model = AudioVQVAE()

    def _sample(bs):
        idx = np.random.randint(0, len(clips), size=min(bs, len(clips)))
        mels = [logmel(clips[i]) for i in idx]  # each [T, N_MELS]
        tlen = min(m.shape[0] for m in mels)
        b = np.stack([m[:tlen] for m in mels], axis=0)  # [B,T,N_MELS]
        return B.ops.array(np.transpose(b, (0, 2, 1)).astype(np.float32))  # [B,N_MELS,T]

    _vqvae_train_loop(B, console, model, _sample, steps, batch, lr, out, "audio VQ-VAE")
    ids = model.encode_ids(clips[0])
    console.print(f"  audio round-trip: {len(ids)} tokens for a {AUDIO_CLIP}s clip")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--level", type=int, default=None, help="Educational level 1-5 (sets vocab + data dir)"
    )
    parser.add_argument(
        "--data_dir", default=None, help="Override the stage-1 data dir to sample from"
    )
    parser.add_argument("--output_dir", default="dist/tokenizer")
    parser.add_argument("--config", default=None, help="Explicit config path (overrides --level)")
    parser.add_argument("--lang", default=None, help="Comma-separated override of config languages")
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=None,
        help="Override vocab size (default: the level's model.vocab_size)",
    )
    parser.add_argument("--sample_mb", type=int, default=500)
    # Multimodal tokenizers (optional): trained after text, each SKIPPED if its data
    # is absent. Text is the only required tokenizer.
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Train only the text tokenizer (skip image + audio)",
    )
    parser.add_argument("--images-dir", default=None, help="Directory of images → image VQ-VAE")
    parser.add_argument(
        "--image-dataset",
        default=None,
        help="HF image dataset for the image VQ-VAE (e.g. uoft-cs/cifar10)",
    )
    parser.add_argument("--audio-dir", default=None, help="Directory of audio clips → audio VQ-VAE")
    parser.add_argument(
        "--audio-synthetic",
        action="store_true",
        help="Use a synthetic tone corpus for the audio VQ-VAE (offline smoke)",
    )
    parser.add_argument("--mm-n", type=int, default=5000, help="Max image/audio samples to load")
    parser.add_argument("--mm-steps", type=int, default=1500, help="Image/audio training steps")
    parser.add_argument("--mm-batch", type=int, default=64, help="Image/audio batch size")
    parser.add_argument("--mm-lr", type=float, default=3e-4, help="Image/audio learning rate")
    parser.add_argument("--mm-img-size", type=int, default=IMG_SIZE)
    parser.add_argument(
        "--backend",
        default=None,
        choices=["mlx", "torch"],
        help="Compute backend for the image/audio VQ-VAEs",
    )
    args = parser.parse_args()

    # Languages: --lang override > config(model.languages) > ['en']
    cfg = load_config(resolve_config_path(args.config, args.level))
    langs = [l.strip() for l in args.lang.split(",")] if args.lang else get_languages(cfg)
    console.print(
        f"  Level: {cfg.get('level', 'custom')} ({cfg.get('name', '')}) | "
        f"Languages: {', '.join(langs)}"
    )

    # Vocab target: explicit flag > level's model.vocab_size > 65536. The level
    # sets a small "child" vocab at low levels; it is auto-capped to data size below.
    vocab_target = args.vocab_size or (cfg.get("model", {}) or {}).get("vocab_size", 65536)

    # Data dir: explicit > the level's stage-1 data dir > legacy default.
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        s1 = (cfg.get("curriculum", {}) or {}).get("stage1", {}) or {}
        lvl = cfg.get("level")  # per-level layout: data/level{N}/stage1
        default_dir = f"data/level{lvl}/stage1" if lvl is not None else "data/stage1"
        data_dir = Path(s1.get("data_dir", default_dir))
    if not any(data_dir.glob("*.jsonl")):
        console.print(f"[red]ERROR:[/red] No .jsonl files in {data_dir}")
        lvl = cfg.get("level", 1)
        console.print(f"Run: python scripts/prepare_data.py --level {lvl} --stage 1 first")
        sys.exit(1)

    try:
        import sentencepiece as spm
    except ImportError:
        console.print("[red]ERROR:[/red] sentencepiece not installed")
        console.print("Run: pip install sentencepiece")
        sys.exit(1)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    prefix = str(Path(args.output_dir) / "rdmca_spm")

    # One sample stream per configured language, routed by each record's `lang`
    # tag (prepare_data writes one file per SOURCE, not per language, so we filter
    # records — not filenames — by language). Languages with no records are dropped
    # after the read so SentencePiece never gets an empty input.
    all_jsonl = list(data_dir.glob("*.jsonl"))
    langs_to_build = [(lang, all_jsonl) for lang in langs]
    if not all_jsonl:
        console.print("[red]ERROR:[/red] No language files found.")
        sys.exit(1)

    # Pre-allocate tmp files
    lang_tmp: dict[str, str] = {}
    tmp_paths: list = []
    for lang, _ in langs_to_build:
        with tempfile.NamedTemporaryFile(
            suffix=f"_{lang.lower()}.txt", delete=False, mode="w"
        ) as f:
            lang_tmp[lang] = f.name
        tmp_paths.append(lang_tmp[lang])

    results = []
    vocab_size = vocab_target

    live.update(_renderable())
    with live:
        try:
            # ── Phase 1: read samples in parallel ────────────────────────────
            task_ids = {
                lang: progress.add_task(f"Reading {lang}", total=None, info="")
                for lang, _ in langs_to_build
            }

            primary = langs[0]
            char_counts: dict = {}  # CHARS written per language (M5)

            def _build(lang: str, files: list) -> str:
                char_counts[lang] = build_text_sample(
                    files,
                    lang,
                    lang_tmp[lang],
                    args.sample_mb,
                    task_ids[lang],
                    lang_filter=lang,
                    default_lang=primary,
                )
                progress.update(task_ids[lang], description=f"[green]Read {lang} ✓[/green]")
                return lang

            with ThreadPoolExecutor(max_workers=len(langs_to_build)) as pool:
                for fut in as_completed({pool.submit(_build, l, f): l for l, f in langs_to_build}):
                    fut.result()

            # Drop languages that produced no text (e.g. an EN-only corpus when ES
            # is also configured) so SPM is not fed an empty file.
            built_langs = [l for l, _ in langs_to_build if char_counts.get(l, 0) > 0]
            if not built_langs:
                raise RuntimeError("No text sampled for any configured language.")
            combined_input = ",".join(lang_tmp[l] for l in built_langs)

            # ── Auto-cap vocab ────────────────────────────────────────────────
            # Use real CHARACTER count (not file bytes — UTF-8 inflates multibyte
            # scripts). The /200 heuristic keeps ≥~200 chars of evidence per piece
            # so SentencePiece isn't asked for more pieces than the corpus supports.
            total_chars = sum(char_counts.get(l, 0) for l in built_langs)
            max_safe = max(500, int(total_chars / 200))
            vocab_size = min(vocab_target, max_safe)
            if vocab_size < vocab_target:
                progress.console.print(
                    f"  [yellow]Vocab auto-reduced[/yellow] "
                    f"{vocab_target} → {vocab_size} "
                    f"(corpus {total_chars / 1e6:.1f}M chars)"
                )

            # ── Phase 2: train BPE (spinner in same live display) ─────────────
            # built_langs computed above (languages that produced text).
            train_spm(
                combined_input,
                prefix,
                vocab_size,
                langs=built_langs,
                num_threads=os.cpu_count() or 4,
            )

            # ── Phase 3: load model, build unified vocab metadata ─────────────
            sp = spm.SentencePieceProcessor()
            sp.Load(prefix + ".model")
            text_vocab = sp.GetPieceSize()
            layout = build_modality_layout(text_vocab)

            lang_token_ids = {l: sp.PieceToId(f"<lang:{l}>") for l in built_langs}
            modality_tokens = {
                "mod_text": sp.PieceToId("<mod:text>"),
                "mod_image": sp.PieceToId("<mod:image>"),
                "mod_audio": sp.PieceToId("<mod:audio>"),
                "mod_end": sp.PieceToId("<mod_end>"),
            }

            # Control delimiters (<think>, <tool_call>, …) — persisted so consumers
            # read stable ids from here instead of the private sp.PieceToId, and so
            # an order change (e.g. adding a language) can't silently shift them.
            control_token_ids = {s: sp.PieceToId(s) for s in CONTROL_SPECIALS}

            # Unified vocab_size spans text ∪ image ∪ audio so the model's
            # embedding table covers every modality from the start.
            with open(Path(args.output_dir) / "tokenizer_info.json", "w") as f:
                json.dump(
                    {
                        "vocab_size": layout["total"],
                        "text_vocab_size": text_vocab,
                        "model": prefix + ".model",
                        "languages": built_langs,
                        "lang_token_ids": lang_token_ids,
                        "modality_tokens": modality_tokens,
                        "control_token_ids": control_token_ids,
                        "modality_layout": layout,
                    },
                    f,
                    indent=2,
                )

            unified_size = layout["total"]

            # ── Verify round-trip per configured language ─────────────────────
            samples = {
                "en": "The quick brown fox jumps over the lazy dog.",
                "es": "El zorro marrón rápido salta sobre el perro perezoso.",
                "fr": "Le rapide renard brun saute par-dessus le chien paresseux.",
                "de": "Der schnelle braune Fuchs springt über den faulen Hund.",
            }
            for lang in built_langs:
                text = samples.get(lang, "Hello world, 123.")
                enc = sp.EncodeAsIds(text)
                dec = sp.DecodeIds(enc)
                results.append((lang, text, len(enc), dec.strip() == text.strip()))

        finally:
            for p in tmp_paths:
                if os.path.exists(p):
                    os.unlink(p)

    console.print()
    show_summary(prefix, vocab_size, unified_size, built_langs, args.output_dir, results)

    # ── Image + audio tokenizers (optional, skipped when data is absent) ───────
    if not args.text_only:
        console.print("\n[bold]Multimodal tokenizers[/bold] (skipped if no data):")
        train_image_tokenizer(
            console,
            args.images_dir,
            args.image_dataset,
            args.mm_n,
            args.mm_img_size,
            args.mm_steps,
            args.mm_batch,
            args.mm_lr,
            IMAGE_OUT,
            backend_name=args.backend,
        )
        train_audio_tokenizer(
            console,
            args.audio_dir,
            args.audio_synthetic,
            args.mm_n,
            args.mm_steps,
            args.mm_batch,
            args.mm_lr,
            AUDIO_OUT,
            backend_name=args.backend,
        )

    console.print("\nNext: [bold]python train_stage.py --stage 1[/bold]")


if __name__ == "__main__":
    main()
