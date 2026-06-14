# Distributed Training Plan — DDP + ZeRO (torch/CUDA)

> Follow-up to `hardware_optimization_report.md` items #3 (ZeRO-1), #19 (DDP),
> #20-22 (pipeline / tensor / sequence parallelism). Confirmed in scope: the project
> WILL run multi-GPU. This is a **plan**, not yet implemented — it needs a real
> multi-GPU CUDA box to build and validate against (single-device dev can't test the
> all-reduce / sharding paths, and shipping untested collective code is worse than
> none). The single-device memory levers it builds on (gradient checkpointing,
> bf16/8-bit optimizer states, the accurate OOM guard) are already in place.

## Scope & ordering (lowest effort / highest payoff first)

| Step | What | Memory/throughput | Effort | Backend |
|---|---|---|---|---|
| 1 | **DDP** (data parallel) | N× throughput | low-med | torch only |
| 2 | **ZeRO-1** (shard optimizer states) | AdamW states ÷ N | med | torch (FSDP `SHARD_GRAD_OP` or ZeroRedundancyOptimizer) |
| 3 | ZeRO-2 (shard grads too) | + grads ÷ N | med | torch FSDP |
| 4 | ZeRO-3 / FSDP full shard | + params ÷ N | high | torch FSDP |
| 5 | tensor / pipeline / sequence parallel | fits >7B | very high | torch, model-surgery |

MLX has no native multi-GPU → all of this lives on the **torch backend only**. Keep
it behind the backend facade so MLX (L0-L3, Apple Silicon) is untouched.

## Where it plugs into the current code

The training loop (`train_stage.py`) and backend engine are already the only seams
that need to change — the model code stays backend-neutral.

1. **Process group / launch.** New `src/core/backend/distributed.py` (torch-only):
   `init_distributed()` reads `RANK`/`WORLD_SIZE`/`LOCAL_RANK` (set by `torchrun`),
   calls `dist.init_process_group("nccl")`, pins `torch.cuda.set_device(local_rank)`.
   `is_main()` / `barrier()` / `rank()` helpers. No-op when `WORLD_SIZE==1` (so the
   single-GPU path is unchanged).

2. **Engine surface additions** (torch backend; MLX gets no-op stubs so the surface
   stays uniform):
   - `wrap_distributed(model, strategy)` → DDP or FSDP-wrapped model.
     - DDP: `torch.nn.parallel.DistributedDataParallel(model, device_ids=[local])`.
     - ZeRO-1: `torch.distributed.optim.ZeroRedundancyOptimizer` wrapping AdamW, OR
       FSDP `ShardingStrategy.SHARD_GRAD_OP`.
     - ZeRO-2/3: `torch.distributed.fsdp.FullyShardedDataParallel` with
       `SHARD_GRAD_OP` / `FULL_SHARD`, `auto_wrap_policy` on `TransformerBlock`,
       `MixedPrecision(param=bf16, reduce=bf16)`, and — composes with what we built —
       `activation_checkpointing` per block (already have `engine.checkpoint`).
   - `all_reduce_mean(value)` for logging the loss/grad-norm across ranks.

3. **Trainer changes** (`train_stage.py`):
   - `init_distributed()` at startup; seed already set per-rank-deterministically
     (`engine.set_seed` is in place — add `+ rank` so shards see different data).
   - Wrap the model via `engine.wrap_distributed(...)` after load, before the optimizer.
   - **Data sharding**: the loader is already seeded + seekable (`DataLoader.skip`).
     Add a `shard=(rank, world_size)` so each rank draws a disjoint slice — simplest:
     `skip(rank)` then stride by `world_size` in `next_batch`, or a per-rank seed
     offset. Token accounting multiplies by `world_size`.
   - Gate eval, checkpoint save, dashboard: **main-rank only** (`if is_main()`),
     `barrier()` before/after save. Checkpoint format is unchanged (save the
     unwrapped `model.module` state; FSDP needs `FullStateDictConfig` gather).
   - The NaN guard, resume-skip, and `optimizer_states=int8` already implemented all
     compose with this unchanged.

4. **Config** (`configs/levels/level{4,5}.yaml`, the CUDA levels):
   ```yaml
   distributed:
     strategy: ddp        # off | ddp | zero1 | zero2 | zero3
     # launched with: torchrun --nproc_per_node=N train_stage.py --level 5 --stage K
   ```
   Default `off` → single-GPU path, zero change.

## Validation (requires the multi-GPU box)

- 1-GPU vs 2-GPU DDP: same loss curve (within noise) at the same *effective* batch;
  ~2× tokens/s.
- ZeRO-1/2/3: per-GPU memory drops ~as the table; loss curve unchanged.
- Checkpoint saved under DDP/FSDP re-loads and runs single-GPU (cross-config load).
- Resume under DDP lands at the right data shard per rank (extends the seeded-skip
  test to the sharded loader).

## Why deferred (not coded now)

Collective code (all-reduce, sharding, FSDP wrap policies) is only meaningfully
testable on ≥2 GPUs; writing it blind on a single Mac and shipping it untested would
be a regression risk. Everything it *depends on* is already done and tested
(checkpointing, low-precision optimizer states, seeded+seekable loader, accurate OOM
guard), so this is a clean, isolated next task to pick up on the CUDA hardware.
