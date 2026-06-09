# Architecture and project structure

## Model

Decoder-only transformer (GPT-style) with RoPE, RMSNorm (pre-norm), SwiGLU FFN and an
MRL (Matryoshka) loss over nested dims. The concrete size is set by the profile
(`configs/profiles/*.yaml`); the base config defaults to d_model=256, 8 layers, 4 heads,
FFN 1024, context 2048.

| Component | Value (base config) |
|---|---|
| Architecture | Decoder-only transformer |
| Positional encoding | RoPE |
| Normalization | RMSNorm (pre-norm) |
| FFN | SwiGLU |
| MRL dims | [64, 128, 256] |
| Core | freezable foundational (Θ_F) + 7 LoRA sectors |
| Backend | MLX (PyTorch selectable in config, not implemented yet) |
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
(MLX freeze/unfreeze): the core and the other sectors stay bit-identical. PGQ can **grow
a sector's rank** (`SectorAdapter.grow_rank`) or **create new sectors**
(`model.add_sector`) at runtime, preserving the output (new components are zero-output at
first).

---

## Backend and precision

- **Backend** (`backend:` top-level key, default `mlx`). `mlx` is the only implemented
  backend. `torch` is accepted but `require_backend()` (`src/config.py`) fails fast with
  a clear error — no silent fallback. The selector exists so the PyTorch backend can be
  wired in later without changing configs.
- **Precision** (`training.precision`, default `bf16`). `set_model_precision()`
  (`src/model/transformer.py`) casts the float params to fp32/bf16/fp16. RoPE and the
  causal mask are dtype-aware so low precision is not silently promoted to fp32. fp16 has
  no loss-scaling — use it for quick smoke tests, not for real runs.

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
│   ├── model/
│   │   ├── transformer.py       RDMCAFoundational + ModelConfig + precision + add_sector
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
│   │   ├── image.py            ImageVQVAE (conv VQ-VAE in MLX)
│   │   ├── audio.py            AudioVQVAE (log-mel VQ-VAE in MLX)
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
│   ├── rdmca_t2.yaml           Base config
│   └── profiles/               test · nano · m2max · a100 · cluster
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

Checkpoints: `dist/checkpoints/<profile>/stage<N>/`, frozen core at
`.../foundational/theta_f_frozen.npz`, sectors at `.../sectors.npz`. Tokenizers in
`dist/tokenizer/`. Long-term memory in `data/runtime/ltss.db`.

---

## The `test` profile

`configs/profiles/test.yaml` replaces the old "toy": the **same real flow** with a small
model, little data and `skip_gate: true`. It points all stages at the same corpus so you
can run the 5 stages → freeze → consolidation without downloading the per-stage datasets.
It is only for verifying the pipeline; the weights are not production-quality.

---

## Consolidation (daemon)

`consolidation_daemon.py` loads the frozen core + sectors, drains
`data/runtime/experiences.jsonl` and runs `ConsolidationPipeline`: BCF filter → adversarial
filter (R⁺<0) → LTSS consistency → MRF → sector assignment (STR + SectorRouter) → masked
per-sector update → PGQ → snapshot/rollback → audit log in `logs/cycle_*.json`. It saves
the sectors to `dist/checkpoints/<profile>/sectors.npz`.

---

## Scaling up (T3 / T4)

The model uses MRL: embeddings are trained over nested dims, so a large model can be
**truncated down** to a smaller tier at inference (not the other way around). Train at the
size you will use.

```python
import mlx.core as mx
w = mx.load("dist/checkpoints/<profile>/foundational/theta_f_frozen.npz")
emb_t3 = w["embed.weight"][:, :512]   # 512-dim prefix
```

| Profile | approx d_model | Target hardware |
|---|---|---|
| test | 256 (4 layers) | smoke test |
| nano | 384 | MacBook M2/M3 |
| m2max | 512 | MacBook M2/M3 Max 64 GB |
| a100 | 768 | 1× A100 (needs the torch/CUDA backend) |
| cluster | 1024 | multi-GPU (needs the torch/CUDA backend) |
