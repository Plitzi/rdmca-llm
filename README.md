# RDMCA — Relevance-Driven Modular Cognitive Architecture

**Apple MLX · idiomas configurables · multimodal (texto + imagen + audio) · currículum de 5 stages**

Modelo de lenguaje adaptativo entrenado desde cero: un *core* cognitivo que se
congela permanentemente + sectores LoRA modulares que aprenden a diario por
**consolidación** de experiencias reales. Texto, imagen y audio comparten un único
espacio de tokens (Era 3b). Behavioral Constraint Function (BCF) integrada.

---

## Quick start

```bash
# 1. Entorno
/opt/homebrew/bin/python3.10 -m venv .venv && source .venv/bin/activate
pip install mlx mlx-lm sentencepiece pyyaml numpy tqdm datasets pytest rich pillow soundfile

# 2. Probar TODO el pipeline real con poca data (~10 min, perfil `test`)
python scripts/prepare_data.py    --profile test --stage 1 --limit 50
python scripts/train_tokenizer.py --profile test --vocab_size 8000 --sample_mb 20
python train_stage.py             --profile test --stage 1
python chat.py                    --profile test --stage 1
```

➡️ **Guía completa, paso a paso: [GUIDE.md](GUIDE.md)** (setup → idiomas → datos →
tokenizers → 5 stages → freeze/BCF → chat texto/imagen/audio → consolidación diaria).

---

## El modelo en un vistazo

| | |
|---|---|
| Arquitectura | Decoder-only · RoPE · RMSNorm · SwiGLU · MRL |
| Core congelable | foundational (Θ_F) + 7 sectores LoRA |
| Idiomas | **configurables** (`model.languages`), por defecto EN+ES |
| Multimodal | texto ∪ imagen (VQ-VAE) ∪ audio (VQ-VAE log-mel), vocab unificado |
| Currículum | 5 stages → core congelado → consolidación diaria |
| Seguridad | BCF + taxonomía de ataques (filtro adversarial en consolidación) |
| Hardware | Apple MLX (M-series) |

Perfiles de hardware en `configs/profiles/`: `test` (smoke), `nano` (~26M),
`m2max` (~109M), `a100`, `cluster`.

---

## Documentación

| Doc | Contenido |
|---|---|
| [GUIDE.md](GUIDE.md) | Guía única paso a paso (init → entrenar → usar → consolidar) |
| [docs/reference/architecture.md](docs/reference/architecture.md) | Modelo, sectores, vocab unificado, estructura, migración T3/T4 |
| [docs/papers/](docs/papers/) | Paper teórico + guía de implementación (referencia) |

## Tests

```bash
python -m pytest tests/ -v
```
