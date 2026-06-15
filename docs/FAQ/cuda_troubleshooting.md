# CUDA Troubleshooting (PyTorch backend on NVIDIA clusters)

This covers the most common cluster issue: PyTorch **silently falls back to CPU**
because the installed `torch` build was compiled against a newer CUDA than the
node's NVIDIA driver supports. On CPU, training is orders of magnitude slower, so
this looks like a "performance" problem but is really a setup mismatch.

## Symptom

At startup you see a warning like:

```
UserWarning: CUDA initialization: The NVIDIA driver on your system is too old
(found version 12070). Please update your GPU driver ... Alternatively, go to:
https://pytorch.org/ to install a PyTorch version that has been compiled with
your version of the CUDA driver.
  return torch._C._cuda_getDeviceCount() > 0
```

Tell-tale signs it ran on CPU anyway:

- The startup announce reports a large **`available`** memory figure (that is
  **system RAM**, not VRAM) and a tiny `est. train memory`.
- `tokens/sec` is very low and the GPU sits at 0% utilization (`nvidia-smi`).

The backend itself is correct — `src/backend/torch_backend.py` picks the device in
the order **CUDA → MPS → CPU**. When `torch.cuda.is_available()` returns `False`,
it just lands on CPU.

## Diagnose

```bash
# 1. What CUDA version does the DRIVER support? (top-right: "CUDA Version: X.Y")
nvidia-smi

# 2. What CUDA was this torch BUILT against, and can it see the GPU?
python -c "import torch; print('torch', torch.__version__, '| built for CUDA', torch.version.cuda, '| cuda available', torch.cuda.is_available())"
```

If `torch.version.cuda` (e.g. `12.8`) is **newer** than the driver's CUDA
(`nvidia-smi`, e.g. `12.0`), that is the mismatch. `cuda available` will be
`False`.

## Fix — install a torch wheel that matches the driver

Pick the wheel index for a CUDA **equal to or older than** the driver's CUDA. When
unsure, `cu118` is the most broadly compatible across 11.8+/12.x drivers.

```bash
# Driver supports CUDA 12.1–12.x:
pip install --index-url https://download.pytorch.org/whl/cu121 torch

# Older / very conservative (works on most 11.8+ and 12.x drivers):
pip install --index-url https://download.pytorch.org/whl/cu118 torch
```

Notes:
- `requirements.txt` only requires `torch>=2.2`. If the latest `torch` has no wheel
  for your CUDA, install an older minor (e.g. `torch==2.5.*`) from the same
  `cuXXX` index — any `>=2.2` build works with this codebase.
- On managed clusters you usually **cannot** update the driver; matching the torch
  wheel to the driver is the correct fix, not upgrading the driver.
- If a module system is in use (`module load cuda/12.1`), load the CUDA module that
  matches the wheel before installing/running.

## Verify

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True  NVIDIA A100-SXM4-80GB   (or your GPU name)
```

Re-run training; the startup announce should now show GPU memory (not system RAM)
and `tokens/sec` should jump by 1–2 orders of magnitude.

## Precision note (after CUDA works)

- **Ampere or newer** (A100, A40, H100, RTX 30/40): keep `precision: bf16`
  (`training.precision` in the level config) — ideal.
- **Volta / Turing** (V100, T4, RTX 20): bf16 is emulated and slow → set
  `precision: fp16` instead.

Check the GPU architecture with `nvidia-smi --query-gpu=name --format=csv`.
