"""Training loops for the multimodal VQ-VAE tokenizers (image + audio).

Folded out of scripts/train_tokenizer.py so the CLI stays thin: it just parses args
and calls these. Each modality is OPTIONAL — with no data for it, the trainer SKIPS
(returns False) rather than failing, since text is the only required tokenizer.
"""

from __future__ import annotations

import time
from pathlib import Path

AUDIO_SR = 16_000  # mirror of audio.SAMPLE_RATE
AUDIO_CLIP = 1.0  # seconds


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
