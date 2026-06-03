# 0 · Primer modelo y perfiles de hardware

Esta guía te lleva del repo vacío a **un modelo con el que puedes chatear**, y
explica los perfiles de entrenamiento para distintos equipos.

## Perfiles disponibles

Cada perfil es un `configs/profiles/<nombre>.yaml`. Selecciónalo con
`--profile <nombre>` en `train_stage.py` y `chat.py`.

| Perfil    | Hardware objetivo            | d_model / capas | Params | Para qué sirve |
|-----------|------------------------------|-----------------|--------|----------------|
| `nano`    | MacBook M2/M3 (16–64 GB)     | 384 / 6         | ~26M   | **Tu primer modelo chateable**, rápido |
| `m2max`   | MacBook M2/M3 Max 64 GB      | 512 / 10        | ~109M  | Primer modelo "de verdad" en tu equipo |
| `a100`    | 1× A100 40/80 GB (CUDA)      | 768 / 12        | ~214M  | Referencia fuerte (requiere backend CUDA) |
| `cluster` | multi-GPU A100/H100          | 1024 / 16       | ~403M  | Modelo grande T4 (requiere backend distribuido) |

> **Nota de hardware honesta:** el repo entrena con **Apple MLX** (Apple
> Silicon). Los perfiles `a100`/`cluster` definen la arquitectura y el
> presupuesto objetivo, pero correrlos en NVIDIA requiere portar el bucle de
> entrenamiento a PyTorch/CUDA (aún no incluido). En tu Mac usa `nano` o `m2max`.

> **MRL y migración entre tiers:** cada perfil entrena un modelo independiente.
> El MRL (dims anidadas) permite *truncar hacia abajo* un modelo grande a un
> tier menor en inferencia, **no** ampliar uno pequeño. Entrena al tamaño que
> vayas a usar (o el mayor que tu hardware permita y trunca para desplegar).

---

## Ruta A — "Quiero ver el chat funcionando YA" (~10 min, en cualquier Mac)

Pesos sin sentido, pero verifica que todo el pipeline corre:

```bash
source .venv/bin/activate
python scripts/make_toy_data.py                 # corpus sintético, sin descargas
python scripts/train_tokenizer.py --vocab_size 8000 --sample_mb 5
python train_stage.py --config configs/rdmca_t2_toy.yaml --stage 1   # ~10 min
python chat.py --config configs/rdmca_t2_toy.yaml --stage 1
```

Esperado: el modelo genera texto incoherente. Sirve para confirmar que
tokenizer + entrenamiento + checkpoints + chat funcionan end-to-end.

---

## Ruta B — Primer modelo con coherencia básica (`nano`, unas horas en M2 Max)

```bash
source .venv/bin/activate

# 1. Datos reales EN+ES (empieza pequeño: ~1 GB). Ver docs/guides/2-data.md
python scripts/prepare_data.py --stage 1 --limit 1000   # ~1 GB

# 2. Tokenizer 16k (rápido y suficiente para nano)
python scripts/train_tokenizer.py --vocab_size 16000 --sample_mb 200

# 3. Entrena el Stage 1 (lenguaje). Puedes parar cuando quieras: hay checkpoints.
python train_stage.py --profile nano --stage 1

# 4. Chatea con el último checkpoint (no hace falta pasar el gate)
python chat.py --profile nano --stage 1 --lang es
```

Notas:
- `nano` usa `vocab_size` del tokenizer entrenado automáticamente.
- Cada `save_every` pasos se guarda un checkpoint en
  `dist/checkpoints/nano/stage1/`. `chat.py` carga el más reciente.
- Para más calidad: añade más datos y continúa con `--stage 2..5`.

---

## Ruta C — Modelo "de verdad" en tu equipo (`m2max`, 1–2 días)

```bash
python scripts/prepare_data.py --stage all          # ~18 GB EN+ES
python scripts/train_tokenizer.py --vocab_size 65536 --sample_mb 500
for s in 1 2 3 4 5; do
  python train_stage.py --profile m2max --stage $s
done
python chat.py --profile m2max --stage 5
```

Tras el Stage 5 el core foundacional se **congela** y (si existe
`data/benchmarks/bcf_probes.jsonl`) se entrena el head BCF. A partir de ahí el
aprendizaje continuo ocurre por consolidación (`consolidation_daemon.py`).

---

## Checkpoints por perfil

Los checkpoints se separan por perfil para que no colisionen:

```
dist/checkpoints/<perfil>/stage<N>/   ← step_*.npz, latest.json, final.npz
dist/checkpoints/<perfil>/foundational/theta_f_frozen.npz   (tras Stage 5)
```

---

## Validar la hipótesis central (no necesita datos ni GPU)

Demuestra en ~30 s que la consolidación sectorizada de RDMCA no olvida, frente
a fine-tuning secuencial (naive) y EWC:

```bash
python experiments/continual_learning.py --domains 5 --steps 250
```

Salida esperada (orden): `naive` peor → `ewc` intermedio → `rdmca` BWT≈0.
Este es el experimento que convierte el paper de "propuesta" en "contribución".
Para el paper real, repítelo sobre tareas NLP secuenciales (Split-AG News, etc.,
§17), no sobre las tareas sintéticas de este script.
