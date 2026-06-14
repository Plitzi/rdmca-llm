"""Device selection for the PyTorch backend — shared by torch_backend and
torch_quant (kept in its own tiny module so both import `DEVICE` without a cycle)."""

from __future__ import annotations

import torch


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _pick_device()
