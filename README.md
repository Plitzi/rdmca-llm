# RDMCA — Relevance-Driven Modular Cognitive Architecture

**Apple MLX · configurable languages · multimodal (text + image + audio) · 5-stage curriculum**

An adaptive language model trained from scratch: a cognitive *core* that is frozen
permanently + modular LoRA sectors that keep learning daily through **consolidation** of
real experiences. Text, image and audio share a single token space (Era 3b). Behavioral
Constraint Function (BCF) built in.

---

## Quick start

```bash
# 1. Environment
/opt/homebrew/bin/python3.10 -m venv .venv && source .venv/bin/activate
pip install mlx mlx-lm sentencepiece pyyaml numpy tqdm datasets pytest rich pillow soundfile

# 2. Exercise the WHOLE real pipeline on a little data (~10 min, `test` profile)
python scripts/prepare_data.py    --profile test --stage 1 --limit 50
python scripts/train_tokenizer.py --profile test --vocab_size 8000 --sample_mb 20
python train_stage.py             --profile test --stage 1
python chat.py                    --profile test --stage 1
```

➡️ **Full step-by-step guide: [docs/GUIDE.md](docs/GUIDE.md)** (setup → backend/precision
→ languages → data → tokenizers → 5 stages → freeze/BCF → chat text/image/audio → daily
consolidation).

---

## The model at a glance

| | |
|---|---|
| Architecture | Decoder-only · RoPE · RMSNorm · SwiGLU · MRL |
| Freezable core | foundational (Θ_F) + 7 LoRA sectors |
| Backend | `mlx` (PyTorch backend selectable in config, not implemented yet) |
| Precision | `fp32 / bf16 / fp16` (`training.precision`, default bf16) |
| Languages | **configurable** (`model.languages`), EN+ES by default |
| Multimodal | text ∪ image (VQ-VAE) ∪ audio (log-mel VQ-VAE), unified vocab |
| Curriculum | 5 stages → frozen core → daily consolidation |
| Safety | BCF + attack taxonomy (adversarial filter in consolidation) |
| Hardware | Apple MLX (M-series) |

Hardware profiles in `configs/profiles/`: `test` (smoke), `nano` (~26M),
`m2max` (~109M), `a100`, `cluster`.

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
