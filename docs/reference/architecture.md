# Architecture and project structure

## Model

Decoder-only transformer (GPT-style) with RoPE, RMSNorm (pre-norm), SwiGLU FFN and an
MRL (Matryoshka) loss over nested dims. The concrete size is set by the **level**
(`configs/levels/level{0..5}.yaml`) — the size follows the *information* the level
teaches, from d_model=256 (level 1) to d_model=768 (level 5) (plus a tiny d_model=64
level 0 for smoke tests). The output projection is **weight-tied** to the input
embedding (a single nested `[vocab, d_model]` matrix serves input lookup and output
at every MRL prefix), so a tier truncation `embed.weight[:, :d]` stays consistent on
both ends.

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

**Mixture-of-Experts (MoE) over sectors.** The sectors S1–S6 are **experts** routed
**per token** by a learned gate (`src/model/moe.py` `SectorGate`): each token activates
only its **top-k** sectors (default k=2) — like the brain, not every expert fires. So the
*active* sector compute stays bounded as the sector pool grows (PGQ), letting modest
hardware keep running the model as it accumulates knowledge. The gate + experts train
**jointly** in consolidation (`pipeline._moe_update`) on the LM loss + a load-balance aux
loss; because routing is per token, **one experience updates several sectors**
(multi-sectorial: a new equation's terminology → Linguistic, its method → Formal).

**S7 (Behavioral/BCF) is excluded from the MoE** — it is always-on and isolated, never in
the consolidation trainable set, shaped only by the BCF probe training. This preserves the
safety guarantee.

**Dispatch** (`RDMCAFoundational._moe_combine`) is **sparse on both backends** — each expert
runs only on its routed tokens, so expert compute is ~O(top_k·T) and stays bounded as the
pool grows (the saving that lets modest hardware keep up):
  - PyTorch (`_moe_sparse`): dynamic gather/scatter (`nonzero`/`index_select`/`index_add`) —
    exact, no token drops.
  - MLX (`_moe_capacity`): GShard-style **fixed-capacity** dispatch (static shapes via
    `cumsum` + index gather/scatter), each expert processing C = ⌈factor·k·T/E⌉ slots;
    overflow tokens are dropped (rare at the default capacity factor 1.25). Routing indices
    are `stop_gradient`-ed (non-differentiable) while the combine weights stay differentiable
    so the gate still learns.
With a large capacity factor (no drops) the capacity path equals the exact path — a parity
check covered by the tests.

PGQ can **grow a sector's rank** (`SectorAdapter.grow_rank`) or **create new experts**
(`model.add_sector`, which grows the gate by a zero-init column) at runtime, preserving the
output (new components are zero-output at first).

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
├── uses/                       Ways to consume a trained model
│   ├── chat/run_chat.py        Interactive chat (text / --image / --audio)
│   └── agent/run_agent.py      Agentic tool loop (Action/Observation)
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

> **Single source of truth for what each level is and exactly what it adds:
> [../levels.md](../levels.md).** This section covers only the *architectural*
> mechanics behind levels; the per-level sizes/stages/data live in that doc.

A level (`configs/levels/level{0..5}.yaml`, `--level N`) sets the model size from the
**information** it teaches (the hardware only caps how high you can run). The **frozen
cognitive core** is seven developmental **stages** (1 Language · 2 Perception ·
3 Abstraction · 4 Causal · 5 Reasoning · 6 Memory · 7 Ethics+BCF), present at every level
(`entry_level ≤ 1`); the core **freezes after the ethics/BCF stage** (`BCF_STAGE = 7`),
so neither competence nor values drift. Three **behavioral** stages (8 tool use · 9 MCP ·
10 skills) then train as **LoRA sectors** on the frozen core — swappable without retraining it.
Reasoning *effort* is a runtime dial (`--think off|low|medium|high`) in `src/agent.py`
(see [uses/chat/](../../uses/chat/)). Data is graded per level via `src/data/graded.py`.

`src/resources.py` estimates a level's parameter count and peak memory from its config,
compares against available RAM/VRAM, and **aborts before an OOM** (with `--force` to
override) — plus an `announce` that prints what the model is learning and from which areas.
The estimate is **precision-aware**, so a lower training precision shrinks it and a heavier
level may fit; `train_stage.py --precision {fp32,bf16,fp16}` overrides the config per run.
For inference on limited hardware, chat/agent accept `--quant {int8,int4}` → real
grouped-affine weight quantization via `engine.quantize` (MLX native; torch weight-only,
packed nibbles at 4-bit; the output head stays in float). Generation is bounded by
`max_new_tokens`, a degenerate-loop detector, and a wall-clock deadline (`--max-seconds`)
— anti-logic-bomb guards that stop only on stuck/repeating output, never genuine reasoning.

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

The per-level ladder (sizes, layers, vocab, backend and what each level adds) is in
**[../levels.md](../levels.md)** — the single source of truth.
