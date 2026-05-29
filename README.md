# RDMCA — Relevance-Driven Modular Cognitive Architecture
**MacBook M2 Max 64GB · Apple MLX · Bilingüe EN+ES · 5-Stage Curriculum**

Modelo de lenguaje adaptativo de 73M parámetros, entrenado desde cero con
curriculum progresivo, consolidación diaria, sectores LoRA modulares y
Behavioral Constraint Function integrada.

---

## Quick start

```bash
# Instalar entorno
/opt/homebrew/bin/python3.10 -m venv .venv && source .venv/bin/activate
pip install mlx mlx-lm sentencepiece pyyaml numpy tqdm datasets pytest rich

# Probar el pipeline sin descargar nada (~10 min)
python scripts/make_toy_data.py
python scripts/train_tokenizer.py --sample_mb 5
python train_stage.py --stage 1 --config configs/rdmca_t2_toy.yaml

# Chat para ver los resultados
python chat.py --stage 1
```

---

## Modelo en un vistazo

| | |
|---|---|
| Arquitectura | Decoder-only, RoPE, RMSNorm, SwiGLU |
| Tamaño | ~31M foundational + 7 sectores LoRA ≈ 73M total |
| Idiomas | EN + ES (vocab 65 536) |
| MRL | Dims anidadas [64, 128, 256] — migrable a T3/T4 sin reentrenar |
| Curriculum | 5 etapas → 4.5B tokens → core congelado permanentemente |
| Hardware | M2 Max 64GB, ~20–25h entrenamiento total |

---

## Documentación — orden de ejecución

| # | Doc | Qué hace |
|---|---|---|
| 1 | [docs/guides/1-setup.md](docs/guides/1-setup.md) | Instalar entorno Python + MLX |
| 2 | [docs/guides/2-data.md](docs/guides/2-data.md) | Descargar corpus Wikipedia EN+ES (~18 GB) |
| 3 | [docs/guides/3-training.md](docs/guides/3-training.md) | Entrenamiento real por stages (4–6 h/stage) |
| 4 | [docs/guides/4-chat.md](docs/guides/4-chat.md) | Probar el modelo con el chat interactivo |
| 5 | [docs/guides/5-eval.md](docs/guides/5-eval.md) | Prompts de evaluación por stage + respuestas esperadas |
| 6 | [docs/guides/6-cleanup.md](docs/guides/6-cleanup.md) | Parar, limpiar datos y cache |

**Experimentos** — fuera del flujo principal:

| Doc | Contenido |
|---|---|
| [docs/experiments/quickstart.md](docs/experiments/quickstart.md) | Pipeline completo en ~15 min sin descargar nada (toy data) |

**Referencia:**

| Doc | Contenido |
|---|---|
| [docs/reference/architecture.md](docs/reference/architecture.md) | Modelo, sectores LoRA, estructura del proyecto, migración T3/T4 |
| [docs/papers/](docs/papers/) | Papers de referencia (.docx) |
