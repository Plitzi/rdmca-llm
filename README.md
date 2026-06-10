# RDMCA — Relevance-Driven Modular Cognitive Architecture

**MLX + PyTorch backends · configurable languages · multimodal (text + image + audio) · 5-stage curriculum**

An adaptive language model trained from scratch: a cognitive *core* that is frozen
permanently + modular LoRA sectors that keep learning daily through **consolidation** of
real experiences. Text, image and audio share a single token space (Era 3b). Behavioral
Constraint Function (BCF) built in.

---

## Quick start

```bash
# 1. Environment — one install works on Mac and Linux/cloud
/opt/homebrew/bin/python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # base + torch; MLX auto-added only on Apple Silicon

# 2. Exercise the WHOLE real pipeline at the most basic level (1 = preescolar)
python scripts/prepare_data.py    --level 1 --stage 1
python scripts/train_tokenizer.py --level 1
python train_stage.py             --level 1 --stage 1
python uses/chat/run_chat.py      --level 1 --stage 1
```

The level sets the model size, the data complexity and the resource use — pick the
highest your hardware can run (a startup guard refuses a level that won't fit).

➡️ **Full step-by-step guide: [docs/GUIDE.md](docs/GUIDE.md)** (setup → backend/precision
→ languages → data → tokenizers → 5 stages → freeze/BCF → chat text/image/audio → daily
consolidation).

---

## The model at a glance

| | |
|---|---|
| Architecture | Decoder-only · RoPE · RMSNorm · SwiGLU · MRL |
| Freezable core | foundational (Θ_F) + 7 LoRA sectors |
| Backend | `mlx` or `torch` (`backend:` key) — same model code, one source of truth |
| Precision | `fp32 / bf16 / fp16` (`training.precision`, default bf16) |
| Languages | **configurable** (`model.languages`), EN+ES by default |
| Multimodal | text ∪ image (VQ-VAE) ∪ audio (log-mel VQ-VAE), unified vocab |
| Curriculum | 5 stages → frozen core → daily consolidation |
| Safety | BCF + attack taxonomy (adversarial filter in consolidation) |
| Hardware | Apple Silicon (MLX) · NVIDIA/CUDA & CPU/MPS (PyTorch) |

**Educational levels** in `configs/levels/` (the size follows the *information*,
not the hardware — your hardware just caps how high you can go):

| Level | Grade | ~params | Learns |
|---|---|---|---|
| 1 | Preescolar | ~2M | basic conversation, simple words, counting & single-digit +/− |
| 2 | Primaria | ~11M | sentences/paragraphs, 2-digit + − ×, simple patterns |
| 3 | Secundaria | ~32M | general knowledge, multi-digit/algebra, basic causal reasoning |
| 4 | Bachillerato | ~76M | advanced text, word-problem math (GSM8K), causal + ethics |
| 5 | Universidad | ~200M | everything, **no filters** (full Wikipedia, MATH, full ethics) |

Cognitive stages (Language, Patterns, Arithmetic, Causal, Ethics) each enter at a
level: language/patterns/arithmetic from level 1; causal at 3; ethics at 4.

---

## Documentation

| Doc | Contents |
|---|---|
| [docs/GUIDE.md](docs/GUIDE.md) | Single step-by-step guide (init → train → use → consolidate) |
| [docs/reference/architecture.md](docs/reference/architecture.md) | Model, sectors, unified vocab, structure, scaling |
| [docs/papers/](docs/papers/) | Theory paper + implementation guide (reference) |

## Tests

```bash
python -m pytest tests/ -v
```
