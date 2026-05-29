← [2-data.md](2-data.md) | → [4-chat.md](4-chat.md)

# Entrenamiento

## Verificación rápida antes de empezar

```bash
# Tests unitarios del modelo y los módulos (sin datos, ~1 seg)
python -m pytest tests/ -v

# Smoke test del training loop con pesos random (Ctrl+C después de 5 seg)
python train_stage.py --stage 1 --config configs/rdmca_t2.yaml
# Esperado: "[data] Tokenizer not found — usando dummy batches" + logs de loss
```

---

## Opción A — Entrenamiento toy (~10 min, solo para probar el pipeline)

Usa datos sintéticos locales sin descargar nada.
Los pesos resultantes no sirven para producción, pero confirman que todo funciona.

```bash
# 1. Generar corpus toy (instantáneo, sin internet)
python scripts/make_toy_data.py

# 2. Entrenar tokenizador sobre el corpus toy (~30 seg)
python scripts/train_tokenizer.py --sample_mb 5

# 3. Entrenar Stage 1 toy (~10 min en M2 Max)
python train_stage.py --stage 1 --config configs/rdmca_t2_toy.yaml
```

Diferencias del config toy vs real:

| Parámetro | Toy | Real |
|---|---|---|
| `context_len` | 128 | 2048 |
| `batch_size` | 4 | 8 |
| `n_tokens` Stage 1 | 500 K | 1.5 B |
| Tiempo estimado | ~10 min | 4–6 h |

Al terminar, probá el chat:
```bash
python chat.py --stage 1
```
El output va a tener palabras reales (tokenizador funcionando) pero sin coherencia
semántica — normal para 500 K tokens de datos toy.

---

## Opción B — Entrenamiento real (4–6 h por stage)

Requiere haber descargado los datos (ver [2-data.md](2-data.md)).

```bash
# Stage 1 — Language (~1.5B tokens)
python train_stage.py --stage 1 --config configs/rdmca_t2.yaml

# Retomar si se interrumpe con Ctrl+C
python train_stage.py --stage 1 --config configs/rdmca_t2.yaml --resume

# Stages 2–5 (correr en orden, cada uno después de pasar el gate)
python train_stage.py --stage 2 --config configs/rdmca_t2.yaml
python train_stage.py --stage 3 --config configs/rdmca_t2.yaml
python train_stage.py --stage 4 --config configs/rdmca_t2.yaml
python train_stage.py --stage 5 --config configs/rdmca_t2.yaml
```

### Graduation gates

Cada stage tiene una métrica que debe alcanzar antes de pasar al siguiente.
El trainer las evalúa automáticamente cada `eval_every` pasos.

| Stage | Métrica | Umbral | Tiempo estimado |
|---|---|---|---|
| 1 Language | BLiMP grammaticality | ≥ 70% | 4–6 h |
| 2 Patterns | ARC Easy accuracy | ≥ 60% | 2–3 h |
| 3 Abstraction | GSM8K accuracy | ≥ 15% | 4–5 h |
| 4 Causal | Causal reasoning bench | ≥ 65% | 4–5 h |
| 5 Ethics | BCF probe set | ≥ 90% | 2–3 h |
| **Total** | | | **~20–25 h** |

### Checkpoints

Guardados automáticamente en `dist/checkpoints/stage{N}/`:
- `step_XXXXXXXX.npz` — checkpoint periódico
- `latest.json` — apunta al más reciente (usado por `--resume`)
- `final.npz` — guardado al pasar el gate
- `stage_complete.json` — marca la etapa como completa

---

## Después del Stage 5

```bash
# El core queda congelado permanentemente en:
# dist/checkpoints/foundational/theta_f_frozen.npz

# Iniciar daemon de consolidación diaria (Phase 2+)
python consolidation_daemon.py --config configs/rdmca_t2.yaml

# Una sola ejecución de prueba
python consolidation_daemon.py --config configs/rdmca_t2.yaml --once
```

---

## Tests por fase

```bash
python -m pytest tests/test_phase1.py -v   # smoke tests Phase 1
python -m pytest tests/test_phase2.py -v   # smoke tests Phase 2
python -m pytest tests/ -v                 # todos (16 skipped hasta entrenar)
```
