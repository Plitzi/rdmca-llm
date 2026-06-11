# Bug Report: Training Pipeline

**File**: `docs/diagnostico-bugs-02-entrenamiento.md`
**Scope**: `train_stage.py`, `src/backend/mlx_backend.py`, `src/backend/torch_backend.py`, `src/backend/base.py`
**Date**: 2026-06-11

---

## C4 — Gradient Accumulation Broken on Both Backends

**Severity**: CRITICAL
**Requires retrain**: No (code fix), but fixes inflate effective batch size so comparison requires retrain

**File**: `train_stage.py:507-515`

```python
for _ in range(grad_acc):
    batch = B.ops.array(data_loader.next_batch())
    loss, g = loss_and_grad_fn(model, batch)
    B.engine.eval(loss)
    acc_loss += B.engine.item(loss)
    grads = g          # <--- OVERWRITES, no acumula!
```

**MLX backend** (`mlx_backend.py:242`): `mlx.nn.value_and_grad` returns fresh gradient tree each call. Only last micro-batch's gradients survive.

**Torch backend** (`torch_backend.py:121-127`):
```python
def _value_and_grad(model, fn):
    def run(*args):
        model.zero_grad(set_to_none=True)   # ← destroys prior micro-batch grads!
        loss = fn(*args)
        loss.backward()
        return loss.detach(), _GRAD_SENTINEL
    return run
```

`zero_grad` called at the start of every micro-batch. Prior micro-batch gradients are destroyed. Only last micro-batch contributes.

**Impact**:
| Level | bs | grad_acc | Effective Batch | Expected Batch | Tokens/Step (counted) | Tokens/Step (actual) |
|-------|----|----------|-----------------|----------------|----------------------|---------------------|
| 1 | 16 | 1 | 16 | 16 (OK) | 4096 | 4096 |
| 2 | 12 | 4 | **12** | 48 | 12288 | **3072** |
| 3 | 8 | 6 | **8** | 48 | 24576 | **4096** |

Loss is correctly averaged across micro-batches (line 514, 533), but gradients driving the update only come from the last micro-batch.

**Fix**: Tree accumulation pattern:
```python
grads = None
for _ in range(grad_acc):
    batch = B.ops.array(data_loader.next_batch())
    loss, g = loss_and_grad_fn(model, batch)
    B.engine.eval(loss)
    acc_loss += B.engine.item(loss)
    if grads is None:
        grads = g
    else:
        grads = B.ops.tree_map(lambda a, b: a + b, grads, g)
grads = B.ops.tree_map(lambda x: x / grad_acc, grads)
```

---

## M6 — LR Schedule Uses Inflated Token Counter

**Severity**: HIGH
**Requires retrain**: Yes (schedule changes)

**File**: `train_stage.py:460-464`

```python
toks_step = bs * seq_len * grad_acc   # multiplied by grad_acc!
total_steps = n_tokens_target // toks_step
```

Level 2: `grad_acc=4`, `toks_step = 12 * 256 * 4 = 12,288` (counted) but only 3,072 actual tokens/step. `total_steps = 110M / 12,288 = 8,951` steps. LR reaches minimum at step 8,951 when model has seen just 27.5M real tokens (25% of budget).

**Fix**: Remove `grad_acc` multiplier: `toks_step = bs * seq_len`.

---

## M1 — Optimizer State Not Saved in Checkpoint

**Severity**: MEDIUM
**Requires retrain**: No (fix checkpoint logic)

**File**: `train_stage.py:155-164`

`load_checkpoint` only loads model weights (`.npz`). AdamW momentum (m) and variance (v) are lost on resume. After resume:
1. Optimizer starts with uninitialized m/v for all parameters
2. Temporary loss spike and slower convergence
3. `cosine_lr` correctly positions at resumed step, but optimizer isn't ready for that LR

**Fix**: Serialize `optimizer.state_dict()` alongside model weights. Load on resume if available.

**MLX note**: `mlx_optim.AdamW` state accessible via `opt.state`. Torch: `optimizer.state_dict()`.

---

## M2 — MoE Aux Loss Never Added to Training Loss

**Severity**: MEDIUM
**Requires retrain**: Yes (loss function changes)

**File**: `train_stage.py:483-485` + `src/model/transformer.py:236-240`

```python
def loss_fn(mdl, toks):
    return mdl.mrl_loss(toks)   # aux_loss never added!
```

MoE load-balance loss is computed in `_route()` and accumulated in `self._aux_accum` / `self._aux_count`, but the training loss function only returns `mrl_loss`. The aux loss receives zero gradient — gate has no incentive to balance expert load.

**Impact**: Only relevant for behavioral stages (7-9) where MoE routing is active. Expert collapse is a real risk without load balancing.

**Fix**: Add aux loss to training loss:
```python
def loss_fn(mdl, toks):
    return mdl.mrl_loss(toks) + 0.01 * mdl.aux_loss()
```
Reset `_aux_accum` and `_aux_count` after each backward pass.

---

## M3 — Corpus Cap Causes Discontinuous LR Jump

**Severity**: MEDIUM
**Requires retrain**: Yes (schedule changes)

**File**: `train_stage.py:537-549`

When corpus cap triggers after first epoch:
1. `total_steps` recalculated with smaller denominator
2. `cosine_lr` progress jumps discontinuously (e.g., 15% → 41%)
3. LR drops abruptly

**Example**: At step 2000 with `total_steps` dropping from 10000 to 4000, progress jumps from 20% to 50%.

**Fix**: Use absolute step-based calculation or maintain original `total_steps` after capping.

---

## C5 — `ops.array(0.0)` DType Mismatch With bf16 (Torch Backend)

**Severity**: CRITICAL
**Requires retrain**: No (code fix)

**File**: `src/model/transformer.py:272`

```python
total = ops.array(0.0)    # float32 scalar
```

When model runs in bf16 on Torch backend, `ops.cross_entropy(...)` returns bf16 scalar. `total + (w / w_sum) * loss_d` adds float32 + bf16 → **RuntimeError: expected scalar type BFloat16 but found Float**.

**Fix**: `total = ops.array(0.0, dtype=logits_d.dtype)` or use `ops.astype(total, logits_d.dtype)`.

---

## L1 — Dashboard Shows avg_loss = 0.0 at Log Boundaries

**Severity**: LOW
**Requires retrain**: No

**File**: `train_stage.py:551-567`

```python
if step % log_interval == 0:
    running_loss = 0.0          # reset BEFORE dashboard reads it
    ...
if step % dash_interval == 0:
    avg_loss = running_loss / max(...)   # reads 0.0!
```

At steps 100, 200, 300 (log_interval=100, dash_interval=10): avg_loss = 0.0 briefly displayed.

**Fix**: Swap order — compute avg first, then reset:
```python
if step % dash_interval == 0:
    avg_loss = running_loss / max(...)
if step % log_interval == 0:
    running_loss = 0.0
```

---

## L2 — Running Loss Window Is Cumulative Average

**Severity**: LOW
**Requires retrain**: No

**File**: `train_stage.py:565`

```python
avg_loss = running_loss / max(step % log_interval or log_interval, 1)
```

At step 10 (log_interval=100): average of steps 1-10. At step 90: average of steps 1-90. Early values dominate the window.

**Fix**: Reset more frequently, or use exponential moving average.

---

## L3 — Checkpoint Write Not Atomic

**Severity**: LOW
**Requires retrain**: No

**Files**: `mlx_backend.py:204-208`, `torch_backend.py:284-289`

```python
np.savez(str(path), **flat)    # writes directly to final path
```

NumPy `savez` writes directly to target file. Crash during write produces truncated/corrupt checkpoint. `latest.json` also written non-atomically.

**Fix**: Write to `path + ".tmp"` then `os.rename()`.

---

## L4 — `strict=False` Silently Ignores Architecture Mismatches

**Severity**: LOW
**Requires retrain**: No

**Files**: `torch_backend.py:295`, `mlx_backend.py:213`

```python
model.load_state_dict(sd, strict=False)   # torch
model.load_weights([...], strict=False)    # mlx
```

Loading level 5 weights into level 1 model would silently ignore extra/missing keys. No guard, no warning.

**Fix**: Add key intersection validation.

---

## L7 — BCF Head Training Uses Constant LR

**Severity**: LOW
**Requires retrain**: No

**File**: `train_stage.py:311`

2-layer MLP probe trained for 30 epochs at constant LR 1e-3, no schedule, no warmup.

**Fix**: Add cosine decay (minor improvement).

---

## H7 — No Held-Out Validation Set

**Severity**: HIGH
**Requires retrain**: Yes (data + training pipeline)

**File**: `train_stage.py:211-223, 492`

```python
val_batches = [data_loader.next_batch() for _ in range(8)]  # from training stream!
```

Every stage uses training data for validation. All HuggingFace dataset loads use `split="train"` exclusively. Gate perplexity measures training fit, not generalization. Overfitting cannot be detected.

**Fix**: Reserve held-out splits during data preparation. Create separate validation data stream and loader.

---

## Summary

| ID | Severity | File(s) | Fix |
|----|----------|---------|-----|
| C4 | CRITICAL | `train_stage.py:507-515` | Tree accumulation of gradients |
| C5 | CRITICAL | `transformer.py:272` | Match dtype in ops.array(0.0) |
| M6 | HIGH | `train_stage.py:460-464` | Remove grad_acc from toks_step |
| H7 | HIGH | `train_stage.py:492` | Add held-out validation split |
| M1 | MEDIUM | `train_stage.py:155-164` | Save/load optimizer state |
| M2 | MEDIUM | `train_stage.py:483`, `transformer.py:236` | Add aux_loss to loss_fn |
| M3 | MEDIUM | `train_stage.py:537-549` | Fix corpus cap LR discontinuity |
| L1 | LOW | `train_stage.py:551-567` | Fix reset order |
| L2 | LOW | `train_stage.py:565` | Use moving average |
| L3 | LOW | Backend save functions | Atomic checkpoint writes |
| L4 | LOW | Backend load functions | Add strict load validation |
| L7 | LOW | `train_stage.py:311` | Add cosine decay for BCF |
