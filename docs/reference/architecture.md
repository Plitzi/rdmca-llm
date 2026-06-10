# Architecture and project structure

## Model

Decoder-only transformer (GPT-style) with RoPE, RMSNorm (pre-norm), SwiGLU FFN and an
MRL (Matryoshka) loss over nested dims. The concrete size is set by the **level**
(`configs/levels/level{1..5}.yaml`) — the size follows the *information* the level
teaches, from d_model=128 (level 1) to d_model=768 (level 5).

| Component | Value (base config) |
|---|---|
| Architecture | Decoder-only transformer |
| Positional encoding | RoPE |
| Normalization | RMSNorm (pre-norm) |
| FFN | SwiGLU |
| MRL dims | [64, 128, 256] |
| Core | freezable foundational (Θ_F) + 7 LoRA sectors |
| Backend | MLX or PyTorch (`backend:` key) — one model source, both supported |
| Precision | fp32 / bf16 / fp16 (`training.precision`, default bf16) |

### LoRA sectors

| ID | Name | Domain | Rank |
|---|---|---|---|
| S1 | Linguistic | Conversation, style, discourse | r=16 |
| S2 | Formal | Math, logic, symbolic | r=16 |
| S3 | WorldKnowledge | Factual, encyclopedic | r=8 |
| S4 | Procedural | Planning, tools | r=8 |
| S5 | Social | Pragmatics, social norms | r=8 |
| S6 | Multimodal | Cross-modal (image/audio ↔ text) | r=8 |
| S7 | Behavioral | Ethics, BCF — adversarial buffer only | r=4 |

Sectors are updated **one at a time** during consolidation, with real gradient masking
(`engine.set_trainable`: MLX freeze/unfreeze, or PyTorch `requires_grad_` toggling): the
core and the other sectors stay bit-identical. PGQ can **grow
a sector's rank** (`SectorAdapter.grow_rank`) or **create new sectors**
(`model.add_sector`) at runtime, preserving the output (new components are zero-output at
first).

---

## Backend and precision

- **Backend** (`backend:` top-level key, default `mlx`). Two backends are fully
  supported — **MLX** (Apple Silicon) and **PyTorch** (CUDA/MPS/CPU) — behind a single
  facade in `src/backend/`. The model is written **once** against the active backend's
  three namespaces:
  - `B.nn` — Module + layer factories (`Linear`, `Embedding`, `Conv*`, `Parameter`,
    `ModuleList`, …); convs use channels-first (NCHW/NCL), MLX wrappers permute internally.
  - `B.ops` — tensor functions, normalized to MLX-style signatures (`axis=`, `keepdims=`).
  - `B.engine` — training/runtime glue (`value_and_grad`, optimizer, `set_trainable`,
    `save_weights`/`load_weights`, precision, memory stats).

  `require_backend(cfg)` (`src/config.py`) calls `backend.select(name)` at startup, so
  model modules must be imported **after** selection — the entrypoints do this with
  function-local imports. Adding a third backend = one `Backend` subclass + a line in
  `src/backend/registry.py`; no model code changes.

  **Checkpoints** use a neutral `.npz` of float32 numpy arrays with identical parameter
  names, so the text foundational core is **cross-backend** (train on MLX, load on torch,
  and vice-versa). The image/audio VQ-VAE checkpoints are *not* cross-backend (conv weight
  layouts differ). On Mac, `bf16` over torch **MPS** is slower/less precise than MLX —
  prefer MLX there.
- **Precision** (`training.precision`, default `bf16`). `set_model_precision()`
  (`src/model/transformer.py`, a thin shim over `engine.set_precision`) casts the float
  params to fp32/bf16/fp16 (and, for torch, moves the model to the selected device). RoPE
  and the causal mask are dtype-aware so low precision is not silently promoted to fp32.
  fp16 has no loss-scaling — use it for quick smoke tests, not for real runs.

---

## Unified vocabulary (multimodal, Era 3b)

Text, image and audio share **one embedding table**. The ranges are disjoint and
persisted in `dist/tokenizer/tokenizer_info.json` (`modality_layout`):

```
text  = [0,            Vt)          SentencePiece (Vt = text_vocab_size)
image = [Vt,           Vt+8192)     image VQ-VAE codebook
audio = [Vt+8192,      Vt+8192+4096) audio VQ-VAE codebook
vocab_size (model) = total
```

- Modality tokens (`<mod:text> <mod:image> <mod:audio> <mod_end>`) and language tokens
  (`<lang:xx>`) are user-defined symbols inside the text range.
- Languages are **config-driven** (`model.languages`); the tokenizer bakes in the chosen
  `<lang:xx>` and stores `lang_token_ids` in `tokenizer_info.json`.
- The **perception layer** (`src/modalities/perception.py`) detects modality, tokenizes
  with the matching tokenizer and assembles the interleaved sequence.
- The `DataLoader` accepts `{"text": ...}` records or pre-tokenized `{"tokens": [...]}`
  (multimodal); the next-token LM objective is the same for every modality.

---

## Project structure

```
rdmca-llm/
├── src/
│   ├── config.py               Config + languages + backend/precision + tokenizer_info
│   ├── backend/                Compute-backend facade (one model, many backends)
│   │   ├── __init__.py          select(name) / current()
│   │   ├── base.py              Backend interface (nn / ops / engine) + surface check
│   │   ├── registry.py          name → backend builder (lazy import)
│   │   ├── mlx_backend.py       MLX implementation (reference)
│   │   └── torch_backend.py     PyTorch implementation (CUDA / MPS / CPU)
│   ├── model/
│   │   ├── config.py            ModelConfig + LoRAConfig (backend-neutral dataclasses)
│   │   ├── transformer.py       RDMCAFoundational + precision shim + add_sector
│   │   ├── lora.py              7 LoRA sectors + grad masking + grow_rank
│   │   └── bcf.py              Behavioral Constraint Function head
│   ├── memory/
│   │   ├── episodic_buffer.py  T1 buffer + Experience
│   │   ├── ltss.py             SQLite (embeddings persisted) + numpy search
│   │   ├── mrf.py              Memory Reevaluation Function
│   │   └── experience_log.py   Experience queue chat → daemon
│   ├── relevance/
│   │   ├── engine.py           R+(e,s): N, U, C, Rep − λ·P
│   │   └── penalty.py          Attack taxonomy (adversarial filter)
│   ├── routing/
│   │   ├── semantic_router.py  STR: segmentation + affinity classifier
│   │   └── sector_router.py    Sector assignment s* for consolidation
│   ├── consolidation/
│   │   ├── pipeline.py         Full consolidation cycle
│   │   ├── snapshot.py         7-day snapshots + rollback + CAT
│   │   ├── ambiguity.py        Deferral + human review queue
│   │   └── pgq.py              Parametric Growth Quantifier (expand / new sector)
│   ├── modalities/
│   │   ├── vocab.py            Unified vocab layout (offsets)
│   │   ├── text.py             SentencePiece wrapper (config-driven languages)
│   │   ├── image.py            ImageVQVAE (conv VQ-VAE, NCHW, backend-neutral)
│   │   ├── audio.py            AudioVQVAE (log-mel VQ-VAE, NCL, backend-neutral)
│   │   ├── vq.py               Shared VectorQuantizer
│   │   └── perception.py       Multimodal Perception Layer (MPL)
│   ├── data/loader.py          DataLoader (text + pre-tokenized multimodal)
│   └── training/dashboard.py   Training dashboard (rich)
├── scripts/
│   ├── prepare_data.py         Download corpus per language + per-stage datasets
│   ├── train_tokenizer.py      SentencePiece + unified vocab
│   ├── train_image_tokenizer.py  Train the image VQ-VAE
│   ├── train_audio_tokenizer.py  Train the audio VQ-VAE
│   └── prepare_multimodal.py   Interleaved image/audio-text grounding data
├── configs/
│   └── levels/                 level1..5 (preescolar..universidad) — size + data + resources
├── src/resources.py            Memory estimate + OOM guard + level announce
├── src/data/graded.py          Graded sources, readability filter, synthetic generators
├── tests/                      test_phase1..4 (model, consolidation, multimodal, PGQ)
├── experiments/continual_learning.py   Hypothesis validation (no-forgetting)
├── train_stage.py              Stage training + freeze + BCF
├── chat.py                     Interactive chat (text / --image / --audio)
├── consolidation_daemon.py     Daily consolidation daemon (wired)
└── docs/
    ├── GUIDE.md                Single step-by-step guide
    ├── reference/architecture.md   This file
    └── papers/                 Theory paper + implementation guide
```

Checkpoints: `dist/checkpoints/level<N>/stage<N>/`, frozen core at
`.../foundational/theta_f_frozen.npz`, sectors at `.../sectors.npz`. Tokenizers in
`dist/tokenizer/`. Long-term memory in `data/runtime/ltss.db`.

---

## Levels (educational curriculum)

Levels replace the old hardware profiles. A level (`configs/levels/level{1..5}.yaml`,
selected with `--level N`) sets the model size, the graded data complexity and the
resource budget — the **information** drives the size, the **hardware** only caps how high
a level you can run. The 5 cognitive **stages** (Language, Patterns, Arithmetic, Causal,
Ethics) each declare an `entry_level`: language/patterns/arithmetic from level 1, causal at
3, ethics at 4. Data is graded per level via `src/data/graded.py` (a Flesch-Kincaid
readability gate + synthetic arithmetic/dialogue/analogy/causal generators + simple
graded corpora like TinyStories / Simple-English-Wikipedia); **level 5 applies no filter**
and reuses the full `data/stage*_*` corpora.

`src/resources.py` estimates a level's parameter count and peak memory from its config,
compares against available RAM/VRAM, and **aborts before an OOM** (with `--force` to
override) — plus an `announce` that prints what the model is learning and from which areas.

---

## Consolidation (daemon)

`consolidation_daemon.py` loads the frozen core + sectors, drains
`data/runtime/experiences.jsonl` and runs `ConsolidationPipeline`: BCF filter → adversarial
filter (R⁺<0) → LTSS consistency → MRF → sector assignment (STR + SectorRouter) → masked
per-sector update → PGQ → snapshot/rollback → audit log in `logs/cycle_*.json`. It saves
the sectors to `dist/checkpoints/level<N>/sectors.npz`.

---

## Scaling up (T3 / T4)

The model uses MRL: embeddings are trained over nested dims, so a large model can be
**truncated down** to a smaller tier at inference (not the other way around). Train at the
size you will use.

```python
import numpy as np                       # checkpoints are neutral .npz (numpy)
w = np.load("dist/checkpoints/level5/foundational/theta_f_frozen.npz")
emb_t3 = w["embed.weight"][:, :512]       # 512-dim prefix
```

| Level | Grade | d_model | ~params | Default backend |
|---|---|---|---|---|
| 1 | Preescolar  | 128 | ~2M   | mlx |
| 2 | Primaria    | 256 | ~11M  | mlx |
| 3 | Secundaria  | 384 | ~32M  | mlx |
| 4 | Bachillerato| 512 | ~76M  | torch |
| 5 | Universidad | 768 | ~200M | torch |
