# RDMCA — Single step-by-step guide

From an empty repo to a trained, multimodal model you can chat with and that keeps
learning daily through consolidation. One linear read. All commands run from the
project root.

> **Core idea:** a cognitive *core* (language, perception, abstraction/math, causality,
> reasoning, memory, ethics — stages 1–7) is trained **once** and then **frozen forever**.
> Continual learning happens in the **LoRA sectors** (tool/MCP/skills — stages 8–10) via
> **daily consolidation** of real experiences. Text, image and audio share **one unified
> token space** (Era 3b).

Contents:
1. [Requirements](#1-requirements)
2. [Setup](#2-setup-once)
3. [Backend & precision](#3-backend--precision)
4. [Choose languages](#4-choose-languages)
5. [Data](#5-data)
6. [Tokenizers (text / image / audio)](#6-tokenizers)
7. [Train the cognitive base](#7-train-the-cognitive-base)
8. [Freeze + BCF](#8-freeze--bcf)
9. [Chat (text / image / audio)](#9-chat)
10. [Daily consolidation](#10-daily-consolidation)
11. [Quick test (level 1)](#11-quick-test-level-1)
12. [Cleanup](#12-cleanup)
13. [Scaling up (T3/T4)](#13-scaling-up-t3t4)

---

## 1. Requirements

- **Apple Silicon Mac** (M1/M2/M3/M4) with the **MLX** backend (unified GPU), **or**
  **Linux/cloud with NVIDIA CUDA** (or CPU/MPS) with the **PyTorch** backend.
- Python 3.10 (macOS: Homebrew).
- ~40 GB free (data + weights + venv) for a real run; level 1 needs little.

## 2. Setup (once)

```bash
/opt/homebrew/bin/python3.10 -m venv .venv      # (any Python 3.10 venv on Linux)
source .venv/bin/activate

# One install works everywhere: base + PyTorch, plus MLX automatically on Apple
# Silicon only (an environment marker makes pip skip MLX on Linux/Windows — no crash).
pip install -r requirements.txt

# Sanity-check the backend(s) you got:
python -c "import mlx.core as mx; print(mx.default_device())"          # MLX → Device(gpu, 0)
python -c "import torch; print(torch.cuda.is_available(), torch.backends.mps.is_available())"
```

`pillow`/`soundfile` are only needed for the multimodal parts (loading images/audio).
The main scripts (`train_stage.py`, `uses/chat/run_chat.py`, `consolidation_daemon.py`) re-exec
themselves with the venv's Python if you run them without activating it.

## 3. Backend & precision

Both are set in the level config.

```yaml
backend: mlx          # mlx | torch  (both fully supported)
training:
  precision: bf16     # fp32 | bf16 | fp16
```

- **Backend** — the **same model code** runs on either backend (one source of truth
  behind `src/backend/`); `require_backend()` selects it at startup. Use `mlx` on
  Apple Silicon (fastest there) and `torch` on Linux/cloud CUDA (or Mac MPS/CPU). The
  higher levels (`level4`/`level5`) default to `torch`; the lower ones to `mlx` — change
  the `backend:` key in any level config. PyTorch auto-picks the device: CUDA → MPS → CPU.
  **Checkpoints are cross-backend** for the text foundational model: a core trained on
  MLX loads into the PyTorch model and vice-versa (identical parameter names).
  *Exception:* the image/audio VQ-VAE weight checkpoints are **not** cross-backend
  (conv weight layouts differ) — train and load those on the same backend.
- **Precision** — `bf16` is the default (paper default, fast and stable on Apple
  Silicon and CUDA). `fp32` is the most stable. `fp16` is the fastest for quick smoke
  tests but has no loss-scaling, so it can produce NaNs on a real run — use it only to
  sanity-check the pipeline. Note: `bf16` on Mac **MPS** (torch) is slower and less
  precise than on MLX/CUDA — prefer MLX on Mac, or `fp32` for MPS sanity runs.
- **Lower precision → bigger levels fit.** The resource guard's memory estimate is
  precision-aware, so dropping `fp32 → bf16` roughly halves weights/grads/activations
  and a heavier level may now fit on the same hardware. Override per run without
  editing the config: `python train_stage.py --level 4 --stage 1 --precision bf16`
  (the guard recomputes with the chosen dtype; the announce prints it).
- **Inference quantization** — for running (not training) on limited hardware, chat/agent
  take `--quant N` for any 2–8 bit width (e.g. `int4`, `8`): real grouped-affine
  quantization via `engine.quantize` on both backends. MLX packs at the true width;
  torch packs nibbles at 4-bit (≈⅛) and stores a byte per weight otherwise (≈¼), so
  4-/8-bit are its memory sweet spots. See [uses/chat/](../uses/chat/).

## 4. Choose languages

Languages are **config-driven**: the single source of truth is `model.languages`.
Everything (data download, tokenizer, training, chat) respects it.

```yaml
model:
  languages: ["en", "es"]     # ← edit this. e.g. ["en"], ["en","es","fr"]
```

- The token budget is split **evenly** across languages.
- Changing languages means **re-training the tokenizer** (the `<lang:xx>` ids are baked
  into the SentencePiece model) and re-training the model.
- One-off override without touching the config: `--lang en,es` on `prepare_data.py` and
  `train_tokenizer.py`.

The chosen languages are persisted to `dist/tokenizer/tokenizer_info.json`, which the
tokenizer and model read at runtime.

## 5. Data

Downloads Wikipedia (one dump per language) + per-stage task datasets. It is
**resumable**: if it stops, re-run the same command.

```bash
# Per stage (recommended; you can pause between stages)
python scripts/prepare_data.py --level 3 --stage 1
python scripts/prepare_data.py --level 3 --stage 2
# … 3, 4, 5

# All at once
python scripts/prepare_data.py --level 3 --stage all

# Most basic level (small, fast)
python scripts/prepare_data.py --level 1 --stage 1
```

Output: `data/level{N}/stage{1..5}/` as `.jsonl` (`{"text": "...", "lang": "<code>"}`),
one file per source. Level 5 reuses the full unfiltered `data/stage{1..5}_*/` dirs.
A real high-level bilingual run is roughly ~18 GB downloaded, ~36 GB on disk.

## 6. Tokenizers

### 6.1 Text (required)

Train it **after** you have Stage-1 data. It creates the SentencePiece model and the
**unified vocabulary** (text ∪ image ∪ audio) in `tokenizer_info.json`.

```bash
python scripts/train_tokenizer.py --level 3 --vocab_size 65536 --sample_mb 500
```

### 6.2 Image and audio (optional, for multimodal)

VQ-VAEs trained from scratch (on the active backend, MLX or PyTorch). They map
image/audio to discrete tokens in the matching range of the unified vocabulary. Pick the
backend with `--backend mlx|torch` (default: auto). Note their weight checkpoints are not
cross-backend — train and load on the same backend.

```bash
# Image (CIFAR-10 by default; or --images-dir with your own images)
python scripts/train_image_tokenizer.py --steps 1500

# Audio (dir of .wav; with no data it generates a synthetic smoke corpus)
python scripts/train_audio_tokenizer.py --audio-dir path/to/wavs
```

Output: `dist/tokenizer/image_vqvae.npz` and `dist/tokenizer/audio_vqvae.npz`.

## 7. Train the cognitive base

Progressive curriculum in **natural order**: cognition (1–6) → values (7, freeze) →
behavioral interfaces (8–10). **All 10 stages run at every level** (each stage's
`entry_level ≤ 1`); levels differ only in size/data/context, not in which stages run.
Each stage must pass its graduation gate before the next (L0/L1 skip it). Checkpoints
are saved automatically; `--resume` continues.

```bash
# Frozen cognitive core (stages 1-7):
python train_stage.py --level 4 --stage 1      # Language
python train_stage.py --level 4 --stage 2      # Perception / patterns
python train_stage.py --level 4 --stage 3      # Abstraction / arithmetic
python train_stage.py --level 4 --stage 4      # Causal / procedural
python train_stage.py --level 4 --stage 5      # Reasoning (chain-of-thought)
python train_stage.py --level 4 --stage 6      # Memory management (recall & use context)
python train_stage.py --level 4 --stage 7      # Ethics + BCF  → freezes the core
# Behavioral interfaces, trained on the frozen core as LoRA sectors (stages 8-10):
python train_stage.py --level 4 --stage 8      # Tool use
python train_stage.py --level 4 --stage 9      # MCP
python train_stage.py --level 4 --stage 10     # Skills

# Resume after Ctrl+C (fast-forwards the data stream to where it stopped)
python train_stage.py --level 4 --stage 1 --resume

# Plain scrolling logs (selectable/copyable, no flicker); a full train.log is
# written to the stage's checkpoint dir regardless of mode.
python train_stage.py --level 4 --stage 1 --plain
```

Same 10 stages at every level — the smaller levels just learn each faculty more
shallowly. `train_stage.py` suggests the next stage and enforces prerequisites.

| Stage | Gate metric | Threshold |
|---|---|---|
| 1 Language | val perplexity (BLiMP proxy) | per level |
| 2 Perception | val perplexity (ARC proxy) | per level |
| 3 Abstraction | val perplexity (GSM8K proxy) | per level |
| 4 Causal | val perplexity | per level |
| 5 Reasoning | val perplexity (chain-of-thought) | per level |
| 6 Memory | val perplexity (recall & use of injected memory) | per level |
| 7 Ethics | val perplexity + **BCF probe ≥ 0.90** | per level |
| 8-10 Tool/MCP/Skills | val perplexity (post-freeze LoRA sectors) | per level |

> Gates use **validation perplexity** as the operative proxy. To use real benchmarks
> (BLiMP/ARC/GSM8K), replace `evaluate_gate` in `train_stage.py`.

Checkpoints live in `dist/checkpoints/level<N>/stage<N>/` (`step_*.npz`, `latest.json`,
`final.npz`, `stage_complete.json`).

## 8. Freeze + BCF

When the **Ethics + BCF stage (stage 7)** completes, the foundational core (stages 1-7:
cognition + memory + values) is **frozen permanently** and saved to
`dist/checkpoints/level<N>/foundational/theta_f_frozen.npz`. If
`data/benchmarks/bcf_probes.jsonl` exists (one `{"text": "...", "label": 0|1}` per line),
the BCF safety head is also trained and saved as `bcf_head.npz`. This happens
automatically inside `train_stage.py --stage 7` (driven by `BCF_STAGE = 7`).

From here the core is never touched again: all learning is through consolidation.

## 9. Chat

```bash
python uses/chat/run_chat.py --level 3 --stage 10                # most complete (core + behavioral)
python uses/chat/run_chat.py --level 3 --stage 1 --lang es       # Spanish session
python uses/chat/run_chat.py --level 3 --stage 10 --think medium # show <think> reasoning
python uses/chat/run_chat.py --level 3 --stage 10 --image foto.png  # visual grounding
python uses/chat/run_chat.py --level 3 --stage 10 --audio clip.wav  # audio grounding
```

In-chat commands: `/lang es` · `/temp 0.7` · `/topp 0.9` · `/maxtok 512`
· `/think off|low|medium|high` · `/format text|json` · `/stats` · `/reset` · `/quit`.

**Thinking / reasoning** — stage 5 teaches a `<think>…</think>` scratchpad
register (real GSM8K chain-of-thought). `--think` / `/think` is an effort dial
(off · low · medium · high, **default medium** — more thinking ≈ better answers)
that sets how big a token budget the scratchpad gets; the chat shows the
scratchpad above the answer. Tokens **stream live by default** (`--no-stream` to
batch). The agent (`uses/agent/run_agent.py`) runs several think→act→observe
rounds until it answers, surfacing each round. See [uses/chat/](../uses/chat/).

With `--image`/`--audio`, the **perception layer** converts the file to unified-vocab
tokens and prepends them to the context (text output, Era 3a). It requires the matching
modality tokenizer (step 6.2).

Each turn is recorded as an **experience** in `data/runtime/experiences.jsonl` for daily
consolidation.

## 10. Daily consolidation

The daemon runs while the system is idle (CPU < 20% for 5+ min), drains the accumulated
experiences and consolidates them: BCF filter → adversarial filter (R⁺<0) → MRF
(promote/retain/expire) → **MoE joint update** of the gate + expert sectors S1–S6 (routed
per token, top-k) → PGQ (sector growth) → snapshot/rollback → audit log in
`logs/cycle_*.json`. The frozen core stays intact; the **per-token gate** means one
experience updates several sectors (multi-sectorial), while **S7 (safety) stays isolated**
(never trained here). Configure routing in each level's `moe:` block (`experts`, `top_k`,
`aux_loss_weight`).

```bash
python consolidation_daemon.py --level 3 --once    # one cycle, then exit
python consolidation_daemon.py --level 3           # daemon (waits for idle)
```

Updated sectors are saved to `dist/checkpoints/level<N>/sectors.npz`; long-term memory
to `data/runtime/ltss.db`.

## 11. Start here — the Level 1 experiment

Level 1 (`configs/levels/level1.yaml`) is the smallest usable base: it trains fast on a
laptop and already **holds a basic conversation and does simple arithmetic**. For what
each level is and **exactly what it adds** over the previous one (sizes, active stages,
the freeze point), see **[levels.md](levels.md)**. Train the stages **in order** (each
starts from the previous one's weights):

```bash
# 1) Graded data → data/level1/stage*/   (ALL 10 stages, at small budgets)
python scripts/prepare_data.py    --level 1 --stage all

# 2) Child-sized tokenizer (vocab auto-caps to the corpus size)
python scripts/train_tokenizer.py --level 1

# 3) Train, in order — the SAME 10 stages as every level (just smaller)
python train_stage.py --level 1 --stage 1      # Language / conversation
python train_stage.py --level 1 --stage 2      # Perception / patterns
python train_stage.py --level 1 --stage 3      # Abstraction / arithmetic
python train_stage.py --level 1 --stage 4      # Causal / procedural
python train_stage.py --level 1 --stage 5      # Reasoning (chain-of-thought)
python train_stage.py --level 1 --stage 6      # Memory management
python train_stage.py --level 1 --stage 7      # Ethics + BCF  → freezes the core
python train_stage.py --level 1 --stage 8      # Tool use      (LoRA sector)
python train_stage.py --level 1 --stage 9      # MCP           (LoRA sector)
python train_stage.py --level 1 --stage 10     # Skills        (LoRA sector)

# 4) Chat with it (most complete checkpoint)
python uses/chat/run_chat.py --level 1 --stage 10
```

On start each command prints the **announce** (what the model is learning + estimated
memory) and runs the **resource guard** (aborts a level that won't fit; `--force` to
override). Data lands in `data/level1/...`, checkpoints in `dist/checkpoints/level1/...`.

**Faster first pass:** the `n_tokens` budgets in `level1.yaml` control run length — lower
them (e.g. 8–10M per stage) for a few-minute end-to-end run, then raise for a fuller model.

**Heads-up:** L1 runs the full 10-stage curriculum like every level — the cognitive
*sectors*, the MoE gate and daily consolidation activate at the **freeze point**
(ethics+BCF, **stage 7**), so they're available once you train through stage 7 even at L1
(they just have little to route at this scale). To exercise MoE routing + consolidation,
train through stage 7+ and run `consolidation_daemon.py --level N --once`.

## 12. Cleanup / fresh start

Use `scripts/purge.py` to wipe generated artifacts and train from zero. It only
removes things the pipeline produces (checkpoints, tokenizer, prepared corpora,
runtime memory, logs) — never your inputs (configs, `.env`, `src/`,
`data/benchmarks/`, the HF cache). It previews what it will delete (with sizes)
and asks for confirmation; `--dry-run` previews only, `--yes` skips the prompt.

```bash
python scripts/purge.py --all --dry-run            # preview a full wipe
python scripts/purge.py --all --yes                # full fresh start
python scripts/purge.py --checkpoints --data --level 1   # redo level 1 only
python scripts/purge.py --tokenizer --checkpoints  # keep prepared data, retrain
```

Targets: `--checkpoints` (`dist/checkpoints/` + `dist/snapshots/`), `--tokenizer`
(`dist/tokenizer/` + VQ-VAE + `*.bak`), `--data` (`data/level*/` corpora),
`--runtime` (`data/runtime/` — `experiences.jsonl`, `ltss.db`), `--logs`. Scope
checkpoints/data to one level with `--level N`.

The shared HuggingFace download cache is left alone by `--all` (re-downloading is
slow and it's shared across projects). To wipe it too, opt in explicitly with
`--hf-cache` (honors `HF_HOME` / `HF_DATASETS_CACHE` / `HF_HUB_CACHE`):

```bash
python scripts/purge.py --hf-cache --dry-run       # preview just the HF cache
python scripts/purge.py --all --hf-cache --yes     # wipe everything incl. HF cache
```

## 13. Scaling up (T3/T4)

The model uses MRL (nested embeddings). A large model can be **truncated down** to a
smaller tier at inference (not the other way around): train at the size you will use.
**Levels** (`configs/levels/`) set the size from the information taught; a startup
**resource guard** refuses a level that won't fit your hardware (`--force` overrides).
For the per-level breakdown — sizes, active stages and exactly what each level adds —
see **[levels.md](levels.md)**; architecture details in
[reference/architecture.md](reference/architecture.md).

---

### Validate the core hypothesis (no data, no GPU)

Shows in seconds that sectorized consolidation does not forget, vs. sequential
fine-tuning and EWC:

```bash
python experiments/continual_learning.py --domains 5 --steps 250
```

Expected ordering: `naive` worst → `ewc` middle → `rdmca` BWT≈0.
