#!/usr/bin/env python3
import sys, os
from pathlib import Path
_venv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv", "bin", "python")
if os.path.exists(_venv) and os.path.abspath(sys.executable) != os.path.abspath(_venv):
    os.execv(_venv, [_venv] + sys.argv)

"""
Train the image VQ-VAE tokenizer (RDMCA §7.2).
Maps images → discrete tokens in the unified vocabulary's image range.

Data: a HuggingFace image dataset (default CIFAR-10) OR a directory of images.

Usage:
  python scripts/train_image_tokenizer.py                       # CIFAR-10
  python scripts/train_image_tokenizer.py --images-dir path/    # your images
  python scripts/train_image_tokenizer.py --steps 2000 --batch 64
"""
import argparse
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.modalities.image import ImageVQVAE, DEFAULT_IMG_SIZE

OUT_PATH = "dist/tokenizer/image_vqvae.npz"


def load_images(images_dir, dataset, n, img_size):
    """Return a float32 array [N, img, img, 3] in [0,1]."""
    from src.modalities.image import _resize
    arrs = []
    if images_dir:
        from PIL import Image
        paths = [p for p in Path(images_dir).rglob("*")
                 if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")]
        for p in paths[:n]:
            arrs.append(_resize(np.asarray(Image.open(p).convert("RGB"),
                                           dtype=np.float32) / 255.0, img_size))
    else:
        from datasets import load_dataset
        ds = load_dataset(dataset, split="train", streaming=True)
        key = None
        for ex in ds:
            if key is None:
                key = "img" if "img" in ex else ("image" if "image" in ex else None)
            img = np.asarray(ex[key].convert("RGB"), dtype=np.float32) / 255.0
            arrs.append(_resize(img, img_size))
            if len(arrs) >= n:
                break
    if not arrs:
        raise SystemExit("No images found. Use --images-dir or install `datasets`.")
    return np.stack(arrs, axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default=None, help="Directory of images")
    ap.add_argument("--dataset", default="uoft-cs/cifar10", help="HF image dataset")
    ap.add_argument("--n", type=int, default=5000, help="Images to load")
    ap.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    print(f"Loading up to {args.n} images …")
    data = load_images(args.images_dir, args.dataset, args.n, args.img_size)
    print(f"  {data.shape[0]} images at {args.img_size}×{args.img_size}")

    model = ImageVQVAE(img_size=args.img_size)
    opt = optim.AdamW(learning_rate=args.lr)
    lg = nn.value_and_grad(model, lambda m, x: m.loss(x))

    n = data.shape[0]
    print(f"Training {args.steps} steps (batch {args.batch}) → {model.n_tokens} tokens/image")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = np.random.randint(0, n, size=args.batch)
        x = mx.array(data[idx])
        loss, grads = lg(model, x)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        if step % 100 == 0 or step == 1:
            print(f"  step {step:5d} | loss {float(loss):.4f} | "
                  f"{step/(time.time()-t0):.1f} it/s")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"Saved → {args.out}")

    # round-trip sanity check
    ids = model.encode_ids(data[0])
    rec = model.decode_ids(ids)
    print(f"Round-trip: {len(ids)} tokens, recon shape {rec.shape}")


if __name__ == "__main__":
    main()
