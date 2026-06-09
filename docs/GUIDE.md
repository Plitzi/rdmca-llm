# RDMCA — Single step-by-step guide

From an empty repo to a trained, multimodal model you can chat with and that keeps
learning daily through consolidation. One linear read. All commands run from the
project root.

> **Core idea:** a cognitive *core* (language, abstraction, math, causality, ethics)
> is trained **once** and then **frozen forever**. Continual learning happens in the
> **LoRA sectors** via **daily consolidation** of real experiences. Text, image and
> audio share **one unified token space** (Era 3b).

Contents:
1. [Requirements](#1-requirements)
2. [Setup](#2-setup-once)
3. [Backend & precision](#3-backend--precision)
4. [Choose languages](#4-choose-languages)
5. [Data](#5-data)
6. [Tokenizers (text / image / audio)](#6-tokenizers)
7. [Train the 5 stages](#7-train-the-5-stages)
8. [Freeze + BCF](#8-freeze--bcf)
9. [Chat (text / image / audio)](#9-chat)
10. [Daily consolidation](#10-daily-consolidation)
11. [Quick test (the `test` profile)](#11-quick-test-the-test-profile)
12. [Cleanup](#12-cleanup)
13. [Scaling up (T3/T4)](#13-scaling-up-t3t4)

---

## 1. Requirements

- **Apple Silicon Mac** (M1/M2/M3/M4) with the **MLX** backend (unified GPU), **or**
  **Linux/cloud with NVIDIA CUDA** (or CPU/MPS) with the **PyTorch** backend.
- Python 3.10 (macOS: Homebrew).
- ~40 GB free (data + weights + venv) for a real run; the `test` profile needs little.

## 2. Setup (once)

```bash
/opt/homebrew/bin/python3.10 -m venv .venv      # (any Python 3.10 venv on Linux)
source .venv/bin/activate
pip install sentencepiece pyyaml numpy tqdm datasets pytest rich pillow soundfile

# + exactly ONE compute backend:
pip install mlx mlx-lm     # Apple Silicon — fastest on Mac
pip install torch          # Linux/cloud (CUDA) or Mac (MPS/CPU)

# Sanity-check the backend you installed:
python -c "import mlx.core as mx; print(mx.default_device())"          # MLX → Device(gpu, 0)
python -c "import torch; print(torch.cuda.is_available(), torch.backends.mps.is_available())"
```

`pillow`/`soundfile` are only needed for the multimodal parts (loading images/audio).
The main scripts (`train_stage.py`, `chat.py`, `consolidation_daemon.py`) re-exec
themselves with the venv's Python if you run them without activating it.

## 3. Backend & precision

Both are set in the config / profile.

```yaml
backend: mlx          # mlx | torch  (both fully supported)
training:
  precision: bf16     # fp32 | bf16 | fp16
```

- **Backend** — the **same model code** runs on either backend (one source of truth
  behind `src/backend/`); `require_backend()` selects it at startup. Use `mlx` on
  Apple Silicon (fastest there) and `torch` on Linux/cloud CUDA (or Mac MPS/CPU). The
  cloud profiles `a100`/`cluster` default to `torch`; `test`/`nano`/`m2max` to `mlx`.
  PyTorch auto-picks the device: CUDA → MPS → CPU.
  **Checkpoints are cross-backend** for the text foundational model: a core trained on
  MLX loads into the PyTorch model and vice-versa (identical parameter names).
  *Exception:* the image/audio VQ-VAE weight checkpoints are **not** cross-backend
  (conv weight layouts differ) — train and load those on the same backend.
- **Precision** — `bf16` is the default (paper default, fast and stable on Apple
  Silicon and CUDA). `fp32` is the most stable. `fp16` is the fastest for quick smoke
  tests but has no loss-scaling, so it can produce NaNs on a real run — use it only to
  sanity-check the pipeline. Note: `bf16` on Mac **MPS** (torch) is slower and less
  precise than on MLX/CUDA — prefer MLX on Mac, or `fp32` for MPS sanity runs.

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
python scripts/prepare_data.py --profile m2max --stage 1
python scripts/prepare_data.py --profile m2max --stage 2
# … 3, 4, 5

# All at once
python scripts/prepare_data.py --profile m2max --stage all

# Small slice for testing (50 MB per language)
python scripts/prepare_data.py --profile test --stage 1 --limit 50
```

Output: `data/stage{1..5}_*/` as `.jsonl` (`{"text": "...", "lang": "<code>"}`).
A real bilingual run is roughly ~18 GB downloaded, ~36 GB on disk.

## 6. Tokenizers

### 6.1 Text (required)

Train it **after** you have Stage-1 data. It creates the SentencePiece model and the
**unified vocabulary** (text ∪ image ∪ audio) in `tokenizer_info.json`.

```bash
python scripts/train_tokenizer.py --profile m2max --vocab_size 65536 --sample_mb 500
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

## 7. Train the 5 stages

Progressive curriculum. Each stage must pass its graduation gate before the next (the
`test` profile skips it). Checkpoints are saved automatically; `--resume` continues.

```bash
python train_stage.py --profile m2max --stage 1      # Language
python train_stage.py --profile m2max --stage 2      # Patterns
python train_stage.py --profile m2max --stage 3      # Abstraction
python train_stage.py --profile m2max --stage 4      # Causality
python train_stage.py --profile m2max --stage 5      # Ethics + BCF

# Resume after Ctrl+C
python train_stage.py --profile m2max --stage 1 --resume
```

| Stage | Gate metric | Threshold |
|---|---|---|
| 1 Language | val perplexity (BLiMP proxy) | per profile |
| 2 Patterns | val perplexity (ARC proxy) | per profile |
| 3 Abstraction | val perplexity (GSM8K proxy) | per profile |
| 4 Causal | val perplexity | per profile |
| 5 Ethics | val perplexity + **BCF probe ≥ 0.90** | per profile |

> Gates use **validation perplexity** as the operative proxy. To use real benchmarks
> (BLiMP/ARC/GSM8K), replace `evaluate_gate` in `train_stage.py`.

Checkpoints live in `dist/checkpoints/<profile>/stage<N>/` (`step_*.npz`, `latest.json`,
`final.npz`, `stage_complete.json`).

## 8. Freeze + BCF

When **Stage 5** completes, the foundational core is **frozen permanently** and saved to
`dist/checkpoints/<profile>/foundational/theta_f_frozen.npz`. If
`data/benchmarks/bcf_probes.jsonl` exists (one `{"text": "...", "label": 0|1}` per line),
the BCF safety head is also trained and saved as `bcf_head.npz`. This happens
automatically inside `train_stage.py --stage 5`.

From here the core is never touched again: all learning is through consolidation.

## 9. Chat

```bash
python chat.py --profile m2max --stage 5                 # core + sectors
python chat.py --profile m2max --stage 1 --lang es       # Spanish session
python chat.py --profile m2max --stage 5 --image foto.png   # visual grounding
python chat.py --profile m2max --stage 5 --audio clip.wav   # audio grounding
```

In-chat commands: `/lang es` · `/temp 0.7` · `/topp 0.9` · `/maxtok 512` · `/stats`
· `/reset` · `/quit`.

With `--image`/`--audio`, the **perception layer** converts the file to unified-vocab
tokens and prepends them to the context (text output, Era 3a). It requires the matching
modality tokenizer (step 6.2).

Each turn is recorded as an **experience** in `data/runtime/experiences.jsonl` for daily
consolidation.

## 10. Daily consolidation

The daemon runs while the system is idle (CPU < 20% for 5+ min), drains the accumulated
experiences and consolidates them: BCF filter → adversarial filter (R⁺<0) → MRF
(promote/retain/expire) → sector assignment → **masked sector update** (core and other
sectors stay intact) → PGQ (sector growth) → snapshot/rollback → audit log in
`logs/cycle_*.json`.

```bash
python consolidation_daemon.py --profile m2max --once    # one cycle, then exit
python consolidation_daemon.py --profile m2max           # daemon (waits for idle)
```

Updated sectors are saved to `dist/checkpoints/<profile>/sectors.npz`; long-term memory
to `data/runtime/ltss.db`.

## 11. Quick test (the `test` profile)

The same real flow, with **less data and a small model** (no "toy", no synthetic data).
Ideal to verify everything runs end-to-end in ~10 min.

```bash
python scripts/prepare_data.py    --profile test --stage 1 --limit 50
python scripts/train_tokenizer.py --profile test --vocab_size 8000 --sample_mb 20
python train_stage.py             --profile test --stage 1
python chat.py                    --profile test --stage 1
```

The `test` profile (`configs/profiles/test.yaml`) uses `skip_gate: true` and points all
stages at the same corpus, so you can run the 5 stages → freeze → consolidation without
downloading the per-stage datasets. It is a pipeline check only; the weights are not
production-quality.

## 12. Cleanup

```bash
# Downloaded data (regenerable)
rm -rf data/stage*_*/*.jsonl
# HuggingFace cache
rm -rf ~/.cache/huggingface/datasets ~/.cache/huggingface/hub
# Model weights (NOT regenerable without retraining)
rm -rf dist/checkpoints/
# Tokenizers (text + VQ-VAE)
rm -rf dist/tokenizer/
# Sector snapshots, logs and memory
rm -rf snapshots/* logs/* data/runtime/ltss.db data/runtime/experiences.jsonl
```

## 13. Scaling up (T3/T4)

The model uses MRL (nested embeddings). A large model can be **truncated down** to a
smaller tier at inference (not the other way around): train at the size you will use.
Profiles: `test` (smoke), `nano` (~26M), `m2max` (~109M), `a100`, `cluster` (the last two
target NVIDIA GPUs and default to `backend: torch`, so they run on CUDA out of the box).
See [reference/architecture.md](reference/architecture.md).

---

### Validate the core hypothesis (no data, no GPU)

Shows in seconds that sectorized consolidation does not forget, vs. sequential
fine-tuning and EWC:

```bash
python experiments/continual_learning.py --domains 5 --steps 250
```

Expected ordering: `naive` worst → `ewc` middle → `rdmca` BWT≈0.
