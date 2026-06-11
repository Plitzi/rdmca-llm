# Bug Report: Model Architecture

**File**: `docs/diagnostico-bugs-03-arquitectura.md`
**Scope**: `src/model/transformer.py`, `src/model/moe.py`, `src/model/config.py`, `configs/levels/*.yaml`
**Date**: 2026-06-11

---

## C1 — MRL Loss Starves Upper Head Dimensions

**Severity**: CRITICAL
**Requires retrain**: Yes

**File**: `src/model/transformer.py:272-284`

```python
weights = [1.0 / d for d in self.cfg.mrl_dims]   # 1/d weighting
```

Level 1: `mrl_dims=[128, 256]`, weights normalized:
- dim 128: `(1/128) / (1/128 + 1/256) = 66.7%`
- dim 256: `(1/256) / (1/128 + 1/256) = 33.3%`

`head_at_dim(h, d)` slices `head.weight[:, :d]` — CE@128 only touches columns 0-127, CE@256 touches all 256. The last 128 columns receive **3× less gradient signal** than the first 128.

**Empirical confirmation**:
```
CE@dim128: 2.3225 (ppl=10.20) — 66.7% weight
CE@dim256: 2.2383 (ppl=9.38)  — 33.3% weight
```
Column norms: first 128 dims mean=6.18 (std=0.77), last 128 dims mean=5.59 (std=0.28). Last 128 are **9.5% weaker**, **2.7× less differentiated**.

Inference uses full d_model=256. The under-trained half of the head degrades all output logits.

**Fix**: Uniform weights: `weights = [1.0 for _ in self.cfg.mrl_dims]`. Or `1/sqrt(d)`. Any scheme that gives ≥50% weight to the largest dimension.

---

## H1 — `head_dim = 32` Below Recommended Minimum

**Severity**: HIGH
**Requires retrain**: Yes

**Configs**: All levels with `d_model=256, n_heads=8` → `head_dim=32`

| Config | d_model | n_heads | head_dim |
|--------|---------|---------|----------|
| Level 0 | 64 | 2 | **32** |
| Level 1 | 256 | 8 | **32** |
| Level 2 | 256 | 8 | **32** |
| Level 3 | 384 | 8 | **48** |
| Level 4 | 512 | 8 | **64** |
| Level 5 | 768 | 8 | **96** |

With head_dim=32:
- QK inner product space is tiny → low-resolution attention
- RoPE has only 16 frequency pairs (vs 32+ at dim 64)
- 8 heads likely collapse to redundant patterns

Recommended minimum head_dim for transformer attention is **64**.

**Fix**: Change `n_heads: 4` for levels 1-2 (head_dim=64). Review level 0 (d_model=64, currently n_heads=2 → head_dim=32, may need n_heads=1 → head_dim=64 but single-head loses multi-head benefit; consider removing level 0 or accepting limitation).

---

## H2 — Weight Initialization Uses Kaiming Uniform

**Severity**: HIGH
**Requires retrain**: Yes

**File**: `src/model/transformer.py` (all `nn.Linear` layers)

Both backends use default initialization:
- **MLX**: Kaiming Uniform `U(-1/sqrt(d_in), 1/sqrt(d_in))`
- **Torch**: Kaiming Uniform (same)

Standard LM practice (GPT, Llama) uses `nn.init.normal_(weight, mean=0.0, std=0.02)`. Kaiming Uniform is designed for ReLU networks, not residual streams. Causes:
- Mismatched variance in the residual stream before the head projection
- Slow early convergence

**Fix**: After model construction, apply:
```python
for param in model.parameters():
    if param.ndim >= 2:
        nn.init.normal_(param, mean=0.0, std=0.02)
```
Zero-init the output head (`head.weight`) or use N(0, 0.01).

---

## H3 — Weight Decay Too High

**Severity**: HIGH
**Requires retrain**: Yes

**Configs**: All level YAML files, field `weight_decay: 0.1`

For 10-30M parameter models, standard weight decay is 0.01-0.05. At 0.1:
- Acts as strong regularizer preventing lower loss
- Combines with small token budget to further restrict effective capacity
- For Level 1 (10.5M params, ~80M tokens): ~8:1 token-to-parameter ratio already low; 0.1 WD effectively halves capacity

**Fix**: Reduce to 0.01 across all configs.

---

## M4 — `top_k` Staleness in MoE Expert Growth

**Severity**: MEDIUM
**Requires retrain**: Yes (for the fix to take effect on dynamically grown MoE)

**File**: `src/model/moe.py:34, 47-56`

```python
# __init__:
self.top_k = min(top_k, n_experts)   # capped at init

# grow_experts (line 47-56):
self.n_experts += delta
# self.top_k is NEVER updated
```

If gate initialized with `n_experts=1, top_k=2` → `top_k=1`. After `grow_experts(5)`: `n_experts=6`, but `top_k` stuck at 1. Only 1 of 6 experts is ever used.

**Currently latent**: Default flow registers all 6 experts before creating the gate. Bug activates only if gate is created with fewer than `top_k` experts and then grown dynamically.

**Fix**: In `grow_experts`: `self.top_k = min(self.top_k, self.n_experts)`.

---

## L8 — `apply_rope` Assumes Even `head_dim`

**Severity**: LOW (latent)
**Requires retrain**: N/A

**File**: `src/model/transformer.py:55-58`

```python
half = D // 2
x1 = x[..., :half]   # floor(D/2) elements
x2 = x[..., half:]   # ceil(D/2) elements
```

If `head_dim` is odd, `x2` is one element wider than `cos/sin`, causing broadcast failure. Currently guarded by `d_model % n_heads == 0` but not by `head_dim % 2 == 0`.

**Fix**: Add `assert head_dim % 2 == 0` in attention init.

---

## L9 — RoPE Tables Allocated Every Forward Pass

**Severity**: LOW (performance)
**Requires retrain**: No

**File**: `src/model/transformer.py:108-109`

```python
cos = ops.array(self._rope_cos[:S])
sin = ops.array(self._rope_sin[:S])
```

Stored as numpy arrays. `ops.array(...)` allocates a new backend tensor every forward pass. When S is constant (training), this is wasteful and causes unnecessary memory churn.

**Fix**: Cache the tensor form and slice instead.

---

## M5 — MRL Dims Have No Validation

**Severity**: MEDIUM (API hygiene)
**Requires retrain**: N/A

**File**: `src/model/config.py:21`

`mrl_dims` can be unsorted, contain duplicates, or exceed `d_model`. The code silently produces wrong weighting or silently truncates.

**File**: `src/model/transformer.py:220-223`

```python
def head_at_dim(self, h, d: int):
    w_d = self.head.weight[:, :d]  # silently truncates if d > d_model
    return h[..., :d] @ w_d.T
```

If `d > cfg.d_model`, `head_at_dim` silently uses full weight matrix — MRL premise is broken.

**Fix**: Add validation in `ModelConfig.__post_init__`: assert sorted ascending, no duplicates, max <= d_model.

---

## Architecture Observations (Not Bugs)

| Aspect | RDMCA | Standard Practice | Note |
|--------|-------|-------------------|------|
| Norm | RMSNorm (pre-norm) | RMSNorm or LayerNorm | Standard (Llama-class) |
| FFN | SwiGLU, 4× hidden | SwiGLU (modern) | Standard |
| RoPE | Pre-computed tables | On-the-fly or cached | Both common |
| Weight tying | Not tied | Often tied | Valid, more capacity |
| Residual scaling | None | None for <50 layers | Fine at N=6-8 |
| Attention upcast | fp32 softmax | Same | Good practice |
| LoRA alpha | 1.0 (scale=1/r) | Typically r (scale=1) | Design choice |
| MoE | Top-k softmax | Switch or GShard | Standard |
| Aux loss | GShard: E·sum(f_e·P_e) | Same | Standard |

---

## Summary

| ID | Severity | File(s) | Fix |
|----|----------|---------|-----|
| C1 | CRITICAL | `transformer.py:272-284` | Uniform MRL weights |
| H1 | HIGH | Config files (n_heads) | n_heads=4 (head_dim=64) |
| H2 | HIGH | `transformer.py` | N(0, 0.02) init |
| H3 | HIGH | Config files (weight_decay) | 0.1 → 0.01 |
| M4 | MEDIUM | `moe.py:47-56` | Update top_k on grow |
| M5 | MEDIUM | `config.py:21`, `transformer.py:220` | Validate mrl_dims |
| L8 | LOW | `transformer.py:55-58` | Assert head_dim even |
| L9 | LOW | `transformer.py:108-109` | Cache RoPE tensors |
