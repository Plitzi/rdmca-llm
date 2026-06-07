#!/usr/bin/env python3
import sys, os
from pathlib import Path
_venv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv", "bin", "python")
if os.path.exists(_venv) and os.path.abspath(sys.executable) != os.path.abspath(_venv):
    os.execv(_venv, [_venv] + sys.argv)

"""
Train the audio VQ-VAE tokenizer over log-mel spectrograms (RDMCA §7.2).
Maps waveforms → discrete tokens in the unified vocabulary's audio range.

Data: a directory of .wav files (--audio-dir). With no data, a synthetic tone
corpus is generated so the pipeline can be smoke-tested offline.

Usage:
  python scripts/train_audio_tokenizer.py --audio-dir path/to/wavs
  python scripts/train_audio_tokenizer.py            # synthetic smoke corpus
"""
import argparse
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.modalities.audio import AudioVQVAE, logmel, SAMPLE_RATE, N_MELS
from src.modalities.perception import load_audio

OUT_PATH = "dist/tokenizer/audio_vqvae.npz"
CLIP_SECS = 1.0


def synthetic_corpus(n: int, sr: int) -> list:
    """n short clips of mixed sine tones + noise — offline smoke data."""
    clips = []
    t = np.linspace(0, CLIP_SECS, int(sr * CLIP_SECS), endpoint=False)
    for _ in range(n):
        f1, f2 = np.random.uniform(110, 880, size=2)
        wav = (0.5 * np.sin(2 * np.pi * f1 * t)
               + 0.3 * np.sin(2 * np.pi * f2 * t)
               + 0.05 * np.random.randn(t.size))
        clips.append(wav.astype(np.float32))
    return clips


def load_wavs(audio_dir: str, n: int, sr: int) -> list:
    paths = [p for p in Path(audio_dir).rglob("*")
             if p.suffix.lower() in (".wav", ".flac", ".ogg")]
    return [load_audio(p, sr) for p in paths[:n]]


def to_mel_batch(clips, idx) -> mx.array:
    """Stack a batch of clips into [B, T, N_MELS], cropping to the shortest."""
    mels = [logmel(clips[i]) for i in idx]
    t = min(m.shape[0] for m in mels)
    return mx.array(np.stack([m[:t] for m in mels], axis=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", default=None, help="Directory of .wav clips")
    ap.add_argument("--n", type=int, default=2000, help="Clips to load")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    if args.audio_dir:
        print(f"Loading audio from {args.audio_dir} …")
        clips = load_wavs(args.audio_dir, args.n, SAMPLE_RATE)
        if not clips:
            raise SystemExit("No audio files found in --audio-dir.")
    else:
        print("No --audio-dir given → generating synthetic smoke corpus.")
        clips = synthetic_corpus(args.n, SAMPLE_RATE)
    print(f"  {len(clips)} clips | sr={SAMPLE_RATE} | n_mels={N_MELS}")

    model = AudioVQVAE()
    opt = optim.AdamW(learning_rate=args.lr)
    lg = nn.value_and_grad(model, lambda m, mel: m.loss(mel))

    n = len(clips)
    print(f"Training {args.steps} steps (batch {args.batch}) …")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = np.random.randint(0, n, size=min(args.batch, n))
        mel = to_mel_batch(clips, idx)
        loss, grads = lg(model, mel)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        if step % 100 == 0 or step == 1:
            print(f"  step {step:5d} | loss {float(loss):.4f} | "
                  f"{step/(time.time()-t0):.1f} it/s")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"Saved → {args.out}")

    ids = model.encode_ids(clips[0])
    print(f"Round-trip: {len(ids)} tokens for a {CLIP_SECS}s clip")


if __name__ == "__main__":
    main()
