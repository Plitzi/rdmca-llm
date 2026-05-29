← [5-eval.md](5-eval.md) | → [README](../../README.md)

# Parar y limpiar

## Parar el entrenamiento

`Ctrl+C` en cualquier momento. El trainer guarda un checkpoint automáticamente
cada `save_every` pasos (configurable en `configs/rdmca_t2.yaml`).
Para retomar exactamente donde quedó:

```bash
python train_stage.py --stage 1 --config configs/rdmca_t2.yaml --resume
```

---

## Limpiar por partes

**1. Solo los datos descargados** (~36 GB en `data/`)
```bash
rm -rf data/stage1_language data/stage2_patterns \
        data/stage3_abstraction data/stage4_causal data/stage5_ethics
```

**2. Solo el cache de HuggingFace** (~2–5 GB en `~/.cache/huggingface/`)
```bash
rm -rf ~/.cache/huggingface/datasets
rm -rf ~/.cache/huggingface/hub
```

**3. Solo los checkpoints** (pesos del modelo entrenado)
```bash
rm -rf dist/checkpoints/
```

**4. Solo el tokenizador entrenado**
```bash
rm -rf dist/tokenizer/
```

**5. Solo logs y snapshots** (generados durante consolidación)
```bash
rm -rf logs/* dist/snapshots/*
```

**6. Solo el entorno virtual** (~2 GB en `.venv/`)
```bash
rm -rf .venv/
```

---

## Limpiar todo (reset completo)

Borra datos, pesos, cache de HF, tokenizador y venv.
El código fuente y los docs quedan intactos.

```bash
# Desde la raíz del proyecto
rm -rf data/stage1_language data/stage2_patterns \
        data/stage3_abstraction data/stage4_causal data/stage5_ethics \
        dist/checkpoints/ dist/tokenizer/ logs/* dist/snapshots/* .venv/

# Cache de HuggingFace (fuera del proyecto)
rm -rf ~/.cache/huggingface/datasets ~/.cache/huggingface/hub
```

Para volver a empezar desde cero después del reset:
```bash
/opt/homebrew/bin/python3.10 -m venv .venv
source .venv/bin/activate
pip install mlx mlx-lm sentencepiece pyyaml numpy tqdm datasets pytest rich
python scripts/prepare_data.py --stage 1
python scripts/train_tokenizer.py
python train_stage.py --stage 1 --config configs/rdmca_t2.yaml
```

---

## Mapa de espacio en disco

| Carpeta | Contenido | Tamaño aprox. | Regenerable |
|---|---|---|---|
| `data/` | Corpus JSONL EN+ES | ~36 GB | Sí — `prepare_data.py` |
| `~/.cache/huggingface/` | Cache temporal HF | ~2–5 GB | Sí — se recrea al descargar |
| `dist/checkpoints/` | Pesos del modelo | ~500 MB–2 GB | **No** — requiere reentrenar |
| `dist/tokenizer/` | SentencePiece model | ~5 MB | Sí — `train_tokenizer.py` |
| `dist/snapshots/` | Backups de sectores LoRA | variable | **No** — historial de consolidación |
| `logs/` | Audit logs de ciclos | pequeño | No (pero son solo logs) |
| `.venv/` | Entorno Python | ~2 GB | Sí — pip install |

> ⚠️ **Antes de borrar `dist/checkpoints/`**: un Stage 1 a mitad de camino puede
> representar varios días de cómputo. Verificar que no hay trabajo valioso antes de borrar.
