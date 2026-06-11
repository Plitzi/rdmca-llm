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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.backend as backend
# Model module is imported lazily in main() AFTER the backend is selected, so the
# module's classes bind to the chosen backend (importing it binds eagerly).

OUT_PATH = "dist/tokenizer/image_vqvae.npz"
DEFAULT_IMG_SIZE = 32   # CIFAR-scale default (kept here to avoid an early import)


def load_images(images_dir, dataset, n, img_size):
    """Return a float32 array [N, img, img, 3] in [0,1]."""
    from src.modalities.image import _resize
    arrs = []
    if images_dir:
        from PIL import Image
        paths = [p for p in Path(images_dir).rglob("*")
                 if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp",
                                         ".tiff", ".tif", ".webp", ".gif")]
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
                if key is None:
                    raise KeyError(f"Dataset '{dataset}' has no 'img'/'image' field. "
                                   f"Available keys: {list(ex.keys())}")
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
    ap.add_argument("--backend", default=None, choices=["mlx", "torch"],
                    help="Compute backend (default: auto — mlx if available, else torch)")
    args = ap.parse_args()

    if args.backend:
        backend.select(args.backend)
    B = backend.current()
    from src.modalities.image import ImageVQVAE   # binds to the selected backend

    print(f"Loading up to {args.n} images …")
    data = load_images(args.images_dir, args.dataset, args.n, args.img_size)
    print(f"  {data.shape[0]} images at {args.img_size}×{args.img_size}")

    model = ImageVQVAE(img_size=args.img_size)
    B.engine.set_precision(model, "fp32")
    opt = B.engine.make_optimizer(model, lr=args.lr, weight_decay=0.0)
    lg = B.engine.value_and_grad(model, lambda m, x: m.loss(x))

    n = data.shape[0]
    print(f"Training {args.steps} steps (batch {args.batch}) → {model.n_tokens} tokens/image")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = np.random.randint(0, n, size=args.batch)
        # data is NHWC [B,H,W,3]; the model is channels-first (NCHW).
        x = B.ops.array(np.transpose(data[idx], (0, 3, 1, 2)))
        loss, grads = lg(model, x)
        B.engine.optimizer_step(opt, model, grads)
        if step % 100 == 0 or step == 1:
            print(f"  step {step:5d} | loss {B.engine.item(loss):.4f} | "
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
