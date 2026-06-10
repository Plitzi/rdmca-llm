#!/usr/bin/env python3
import sys, os
from pathlib import Path
_venv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv", "bin", "python")
if os.path.exists(_venv) and os.path.abspath(sys.executable) != os.path.abspath(_venv):
    os.execv(_venv, [_venv] + sys.argv)

"""
Build interleaved multimodal grounding data (RDMCA §7.5, §1.4.2).

Each record is a pre-tokenized unified-vocab sequence the trainer consumes
directly: <image tokens> <mod_end> <mod:text> <caption tokens>. Grounding data
goes into the Stage-2 (patterns) corpus by default.

Prereqs: text tokenizer + the relevant modality VQ-VAE must already be trained.

Usage:
  # image–caption pairs from CIFAR-10 (class name as caption)
  python scripts/prepare_multimodal.py --images --n 2000
  # audio–transcript pairs from a dir of .wav with sidecar .txt
  python scripts/prepare_multimodal.py --audio-dir path/ --n 1000
"""
import argparse
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.modalities.perception import MultimodalPerception, load_audio
from src.modalities.text import TextTokenizer

OUT_DIR = Path("data/level5/stage2")


def write(records, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  {len(records)} records → {path}")


def images(mpl, n, lang):
    from datasets import load_dataset
    ds = load_dataset("uoft-cs/cifar10", split="train", streaming=True)
    names = ["airplane", "automobile", "bird", "cat", "deer",
             "dog", "frog", "horse", "ship", "truck"]
    recs = []
    for ex in ds:
        img = ex.get("img") or ex.get("image")
        cap = f"a photo of a {names[ex['label']]}" if "label" in ex else "an image"
        toks = mpl.encode_image(img) + mpl.encode_text(cap, lang=lang, boundary=True)
        recs.append({"tokens": toks, "modality": "mixed"})
        if len(recs) >= n:
            break
    write(recs, "grounding_image.jsonl")


def audio(mpl, audio_dir, n, lang):
    paths = [p for p in Path(audio_dir).rglob("*")
             if p.suffix.lower() in (".wav", ".flac", ".ogg")]
    recs = []
    for p in paths[:n]:
        txt = p.with_suffix(".txt")
        cap = txt.read_text().strip() if txt.exists() else "an audio clip"
        toks = mpl.encode_audio(load_audio(p)) + mpl.encode_text(cap, lang=lang, boundary=True)
        recs.append({"tokens": toks, "modality": "mixed"})
    write(recs, "grounding_audio.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", action="store_true", help="Build image–caption pairs")
    ap.add_argument("--audio-dir", default=None, help="Dir of .wav + sidecar .txt")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--lang", default="en")
    args = ap.parse_args()

    tok = TextTokenizer()
    if not tok.ready:
        raise SystemExit("Train the text tokenizer first: scripts/train_tokenizer.py")
    mpl = MultimodalPerception(text_tok=tok)

    if args.images:
        images(mpl, args.n, args.lang)
    if args.audio_dir:
        audio(mpl, args.audio_dir, args.n, args.lang)
    if not args.images and not args.audio_dir:
        print("Nothing to do. Pass --images and/or --audio-dir.")


if __name__ == "__main__":
    main()
