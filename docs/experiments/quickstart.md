← [README](../../README.md) | → [1-setup.md](../guides/1-setup.md)

# Entrenamiento mínimo — Stage 1 (hola mundo)

Entrena el modelo con datos sintéticos locales en ~10 minutos.
No requiere descargar nada. El resultado no es un modelo útil, pero
confirma que todo el pipeline funciona de punta a punta.

---

## Los 3 comandos

```bash
# 1. Generar corpus sintético (~6 MB, instantáneo)
python scripts/make_toy_data.py

# 2. Entrenar el tokenizador sobre ese corpus (~30 seg)
python scripts/train_tokenizer.py --sample_mb 5

# 3. Entrenar Stage 1 con config reducida (~10 min)
python train_stage.py --stage 1 --config configs/rdmca_t2_toy.yaml
```

---

## Qué esperar durante el entrenamiento

El trainer imprime una línea cada 100 pasos:

```
============================================================
  Stage 1: Language and communication
  Target: 0.00B tokens
============================================================
  Model: 31.0M params | d_model=256 | layers=8
  [data] Real data loader: data/stage1_language
  Starting from step 0 | tokens 0.0M

  step=    100 |     0.5M tok (  9.6%) | loss=8.2341 | lr=3.00e-04 | 12.3K tok/s
  step=    200 |     1.0M tok ( 19.2%) | loss=7.1204 | lr=2.98e-04 | 13.1K tok/s
  step=    300 |     1.5M tok ( 28.8%) | loss=6.4891 | lr=2.95e-04 | 13.4K tok/s
  ...
  step=    977 |     5.0M tok (100.0%) | loss=5.1XXX | lr=3.00e-05 | XX.XK tok/s
```

**Señales de que va bien:**

| Qué observar | Valor esperado |
|---|---|
| Loss en step 100 | 8–10 (random init cerca de `log(65536) ≈ 11`) |
| Loss en step 500 | 6–7 |
| Loss al final (~977 steps) | 4–6 |
| Velocidad | 10–15 K tok/s en M2 Max |
| Memoria GPU | < 4 GB |

Si el loss **no baja** en los primeros 200 pasos, algo está mal
(ver sección de troubleshooting más abajo).

Para interrumpir en cualquier momento: `Ctrl+C`.
El checkpoint queda guardado automáticamente.

---

## Verificar que funcionó

```bash
python chat.py --stage 1
```

Escribí estos prompts y observá el output:

**Test 1 — el tokenizador decodifica correctamente**
```
hi
```
> Esperado: palabras reales en inglés (aunque incoherentes)
> ❌ Si ves `[token IDs — tokenizer not trained]: [23847, ...]` → el tokenizador no se entrenó correctamente

**Test 2 — el modelo aprendió algo del corpus**
```
The sun
```
> Esperado: continuación con palabras relacionadas al corpus (rises, is, shines...)
> ⚠️ Output completamente random: normal con solo 500K tokens

**Test 3 — soporte español**
```
/lang es
El sol
```
> Esperado: palabras en español (aunque sin coherencia)
> ❌ Si solo genera en inglés: el corpus ES no se procesó

**Test 4 — velocidad**
```
/stats
```
> Esperado: 40–80 tok/s
> ❌ Si dice `cpu` en vez de `gpu`: MLX no está usando Apple Silicon

---

## Diferencia con el entrenamiento real

| | Toy | Real |
|---|---|---|
| Config | `rdmca_t2_toy.yaml` | `rdmca_t2.yaml` |
| Tokens Stage 1 | 500 K | 1.5 B |
| context_len | 128 | 2048 |
| Tiempo | ~10 min | 4–6 h |
| Datos | Sintéticos locales | Wikipedia EN+ES |
| Loss final esperado | 4–6 (no converge) | < 2.5 (convergido) |
| ¿Pasa el graduation gate? | No | Sí (si corpus suficiente) |
| ¿Útil para producción? | No | Sí |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'mlx'`**
```bash
source .venv/bin/activate   # activar el venv manualmente
```
O simplemente correr con `python` directamente — el script se auto-redirige al venv.

**`FileNotFoundError: No .jsonl files found in data/stage1_language`**
```bash
python scripts/make_toy_data.py   # olvidaste el paso 1
```

**`RuntimeError: Tokenizer not found`**
```bash
python scripts/train_tokenizer.py --sample_mb 5   # olvidaste el paso 2
```

**El loss no baja (se mantiene en ~10–11)**
- Verificar que MLX usa GPU: `python -c "import mlx.core as mx; print(mx.default_device())"`
- Esperado: `Device(gpu, 0)`. Si dice `cpu`: reinstalar con `pip install --upgrade mlx`

**El chat muestra token IDs en vez de texto**
- El tokenizador no está en `dist/tokenizer/rdmca_spm.model`
- Correr: `python scripts/train_tokenizer.py --sample_mb 5`

---

## Próximo paso

Una vez confirmado que el pipeline funciona, pasá al entrenamiento real:

```bash
python scripts/prepare_data.py --stage 1   # descarga ~8 GB Wikipedia EN+ES
python scripts/train_tokenizer.py          # re-entrena sobre datos reales
python train_stage.py --stage 1 --config configs/rdmca_t2.yaml
```

Ver [4-training.md](../guides/4-training.md) para el flujo completo.
