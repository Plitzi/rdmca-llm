"""Weight-only group-affine quantization (2–8 bit) for the PyTorch backend.

A real quantizer (not a fallback): Linear/Embedding weights are stored as grouped
affine integers and dequantized per forward (standard weight-only scheme), matching
the MLX backend's grouped affine quantization so both backends behave the same
numerically at any bit-width. Storage: 4-bit packs two nibbles per byte (~8× smaller
than fp32); every other width (2,3,5,6,7,8) stores one uint8 per weight (~4× smaller,
same footprint regardless of width — only the numerical precision changes). 4-bit and
8-bit are therefore the memory sweet spots on this backend.

Split out of torch_backend.py; `quantize` is wired into the engine namespace there.
"""

from __future__ import annotations

import torch
import torch.nn as torch_nn

from src.backend.torch_device import DEVICE


def _quantize_affine(w: torch.Tensor, bits: int, group_size: int):
    """w:[rows, feat] → (q uint8 [rows, n_groups, g], scale, zero [rows, n_groups])."""
    rows, feat = w.shape
    g = group_size
    ng = feat // g
    wg = w.reshape(rows, ng, g).to(torch.float32)
    wmin = wg.amin(dim=-1)
    wmax = wg.amax(dim=-1)
    qmax = (1 << bits) - 1
    scale = (wmax - wmin) / qmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    zero = torch.round(-wmin / scale)
    q = torch.clamp(torch.round(wg / scale.unsqueeze(-1) + zero.unsqueeze(-1)), 0, qmax)
    return q.to(torch.uint8), scale, zero


def _dequantize_affine(q: torch.Tensor, scale, zero, bits: int) -> torch.Tensor:
    rows, ng, g = q.shape
    w = (q.to(torch.float32) - zero.unsqueeze(-1)) * scale.unsqueeze(-1)
    return w.reshape(rows, ng * g)


def _pack4(q: torch.Tensor) -> torch.Tensor:
    """Pack a uint8 tensor (values 0-15), even last dim, two nibbles per byte."""
    lo = q[..., 0::2]
    hi = q[..., 1::2]
    return (lo | (hi << 4)).to(torch.uint8)


def _unpack4(p: torch.Tensor) -> torch.Tensor:
    lo = p & 0x0F
    hi = (p >> 4) & 0x0F
    out = torch.stack((lo, hi), dim=-1)
    return out.reshape(*p.shape[:-1], p.shape[-1] * 2)


class _QuantLinear(torch_nn.Module):
    def __init__(self, child: torch_nn.Linear, bits: int, group_size: int):
        super().__init__()
        self.bits, self.group_size = bits, group_size
        self.out_features, self.in_features = child.out_features, child.in_features
        q, scale, zero = _quantize_affine(child.weight.detach(), bits, group_size)
        self.register_buffer(
            "qweight", _pack4(q.reshape(q.shape[0], -1)) if bits == 4 else q.reshape(q.shape[0], -1)
        )
        self.register_buffer("scale", scale)
        self.register_buffer("zero", zero)
        self.register_buffer("bias", child.bias.detach() if child.bias is not None else None)

    def _weight(self) -> torch.Tensor:
        ng = self.in_features // self.group_size
        q = self.qweight
        q = _unpack4(q) if self.bits == 4 else q
        q = q.reshape(self.out_features, ng, self.group_size)
        return _dequantize_affine(q, self.scale, self.zero, self.bits)

    def forward(self, x):
        w = self._weight().to(x.dtype)
        return torch_nn.functional.linear(
            x, w, self.bias.to(x.dtype) if self.bias is not None else None
        )

    __call__ = forward


class _QuantEmbedding(torch_nn.Module):
    def __init__(self, child: torch_nn.Embedding, bits: int, group_size: int):
        super().__init__()
        self.bits, self.group_size = bits, group_size
        self.num_embeddings, self.embedding_dim = child.num_embeddings, child.embedding_dim
        q, scale, zero = _quantize_affine(child.weight.detach(), bits, group_size)
        self.register_buffer(
            "qweight", _pack4(q.reshape(q.shape[0], -1)) if bits == 4 else q.reshape(q.shape[0], -1)
        )
        self.register_buffer("scale", scale)
        self.register_buffer("zero", zero)

    def _table(self) -> torch.Tensor:
        ng = self.embedding_dim // self.group_size
        q = self.qweight
        q = _unpack4(q) if self.bits == 4 else q
        q = q.reshape(self.num_embeddings, ng, self.group_size)
        return _dequantize_affine(q, self.scale, self.zero, self.bits)

    def forward(self, ids):
        return torch_nn.functional.embedding(ids, self._table())

    __call__ = forward


def quantize(model, bits: int = 4, group_size: int = 64, skip_names: tuple = ("embed",)) -> None:
    """In-place weight-only quantization of Linear/Embedding submodules. Layers
    whose feature dim isn't divisible by `group_size`, or whose last path
    component is in `skip_names` (by default `embed` — weight-tied as the output
    projection, sliced by `.weight` for MRL and the most quant-sensitive), are
    left in their float dtype."""
    name_to_mod = dict(model.named_modules())
    targets, skipped = [], 0
    for full_name, child in list(model.named_modules()):
        if isinstance(child, torch_nn.Linear):
            feat, kind = child.in_features, _QuantLinear
        elif isinstance(child, torch_nn.Embedding):
            feat, kind = child.embedding_dim, _QuantEmbedding
        else:
            continue
        if full_name.split(".")[-1] in skip_names:
            continue
        if feat % group_size != 0:
            skipped += 1
            continue
        parent_name, _, cname = full_name.rpartition(".")
        parent = name_to_mod[parent_name] if parent_name else model
        targets.append((parent, cname, child, kind))
    for parent, cname, child, kind in targets:
        setattr(parent, cname, kind(child, bits, group_size).to(DEVICE))
    if skipped:
        print(
            f"  [quant] {skipped} layer(s) not divisible by group_size={group_size} "
            f"kept in float dtype"
        )
