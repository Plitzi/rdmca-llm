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
python scripts/train.py             --level 1 --stage 1
python models/cognition/uses/chat/run_chat.py      --level 1 --stage 1
```

The level sets the model size, the data complexity and the resource use — pick the
highest your hardware can run (a startup guard refuses a level that won't fit).

➡️ **Full step-by-step guide: [docs/GUIDE.md](docs/GUIDE.md)** (setup → backend/precision
→ languages → data → tokenizers → cognitive core (stages 1-7) → freeze/BCF →
behavioral stages (8 tool/9 MCP/10 skills) → chat text/image/audio → daily consolidation).

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
| Curriculum | cognitive base (6 stages, freeze at ethics) → behavioral (tool/MCP/skills) → daily consolidation |
| Safety | BCF + attack taxonomy (adversarial filter in consolidation) |
| Hardware | Apple Silicon (MLX) · NVIDIA/CUDA & CPU/MPS (PyTorch) |

**Educational levels** in `configs/levels/` — the size follows the *information*, not
the hardware (your hardware just caps how high you can go). From **0 Pruebas** (smoke)
through **1 Preescolar · 2 Primaria · 3 Secundaria · 4 Bachillerato · 5 Universidad**,
each level grows the model and/or activates a new cognitive faculty (Language/Patterns/
Arithmetic/Reasoning from L1, Causal at L3, Ethics+freeze at L4).

➡️ **See [docs/levels.md](docs/levels.md)** — the single source of truth for per-level
sizes, active stages and **exactly what each level adds** over the previous one.

---

## Documentation

| Doc | Contents |
|---|---|
| [docs/GUIDE.md](docs/GUIDE.md) | Single step-by-step guide (init → train → use → consolidate) |
| [docs/reference/architecture.md](docs/reference/architecture.md) | Model, sectors, unified vocab, structure, scaling |
| [docs/papers/](docs/papers/) | Theory paper + implementation guide (reference) |

## Tests

Framework tests live under `src/`; each model's tests live with the model
(`models/<model>/.../tests/`). `pytest.ini` collects both — just run:

```bash
python -m pytest -q
```
