# RDMCA — Guía única, paso a paso

Del repo vacío a un modelo entrenado, multimodal, con el que podés chatear y que
aprende a diario por consolidación. Una sola lectura lineal. Todos los comandos
se corren desde la raíz del proyecto.

> **Idea central del proyecto:** se entrena **una vez** un *core* cognitivo
> (lenguaje, abstracción, matemáticas, causalidad, ética) que luego se **congela
> para siempre**. El aprendizaje continuo ocurre en los **sectores LoRA** mediante
> la **consolidación diaria** de las experiencias reales. Texto, imagen y audio
> comparten **un único espacio de tokens** (Era 3b).

Índice:
1. [Requisitos](#1-requisitos)
2. [Setup](#2-setup-una-sola-vez)
3. [Elegir idiomas](#3-elegir-idiomas)
4. [Datos](#4-datos)
5. [Tokenizers (texto / imagen / audio)](#5-tokenizers)
6. [Entrenar los 5 stages](#6-entrenar-los-5-stages)
7. [Congelar + BCF](#7-congelar--bcf)
8. [Chatear (texto / imagen / audio)](#8-chatear)
9. [Consolidación diaria](#9-consolidación-diaria)
10. [Probar rápido (perfil `test`)](#10-probar-rápido-perfil-test)
11. [Limpieza](#11-limpieza)
12. [Migración a hardware mayor (T3/T4)](#12-migración-t3t4)

---

## 1. Requisitos

- MacBook con Apple Silicon (M1/M2/M3/M4) — el entrenamiento usa **Apple MLX** (GPU unificada).
- macOS con Homebrew y Python 3.10.
- ~40 GB libres (datos + pesos + venv) para un run real; el perfil `test` necesita poco.

## 2. Setup (una sola vez)

```bash
/opt/homebrew/bin/python3.10 -m venv .venv
source .venv/bin/activate
pip install mlx mlx-lm sentencepiece pyyaml numpy tqdm datasets pytest rich pillow soundfile

# Verificar que MLX usa el GPU de Apple Silicon
python -c "import mlx.core as mx; print(mx.default_device())"   # Device(gpu, 0)
```

`pillow`/`soundfile` solo hacen falta para la parte multimodal (cargar imágenes/audio).
Los scripts principales (`train_stage.py`, `chat.py`, `consolidation_daemon.py`) se
auto-reinician con el Python del venv si los corrés sin activarlo.

## 3. Elegir idiomas

Los idiomas son **config-driven**: hay una sola fuente de verdad, la lista
`model.languages` del perfil/config. Todo (descarga de datos, tokenizer,
entrenamiento, chat) la respeta.

```yaml
# en configs/profiles/<perfil>.yaml  o  configs/rdmca_t2.yaml
model:
  languages: ["en", "es"]     # ← editá esto. p.ej. ["en"], ["en","es","fr"]
```

- El presupuesto de tokens se reparte **equitativamente** entre los idiomas.
- Cambiar idiomas implica **re-entrenar el tokenizer** (los IDs `<lang:xx>` se hornean
  en el modelo SentencePiece) y re-entrenar el modelo.
- Override puntual sin tocar el config: `--lang en,es` en `prepare_data.py` y
  `train_tokenizer.py`.

Los idiomas elegidos quedan persistidos en `dist/tokenizer/tokenizer_info.json`, que
es lo que leen el tokenizer y el modelo en runtime.

## 4. Datos

Descarga Wikipedia (un dump por idioma) + datasets de tarea por stage. Es
**resumible**: si se corta, volvé a correr el mismo comando.

```bash
# Por stage (recomendado, podés pausar entre etapas)
python scripts/prepare_data.py --profile m2max --stage 1
python scripts/prepare_data.py --profile m2max --stage 2
# … 3, 4, 5

# Todo de una
python scripts/prepare_data.py --profile m2max --stage all

# Slice chico para probar (50 MB por idioma)
python scripts/prepare_data.py --profile test --stage 1 --limit 50
```

Salida: `data/stage{1..5}_*/` en `.jsonl` (`{"text": "...", "lang": "<code>"}`).
Tamaño aproximado de un run real bilingüe: ~18 GB descargados, ~36 GB en disco.

## 5. Tokenizers

### 5.1 Texto (obligatorio)

Se entrena **después** de tener datos del Stage 1. Crea el modelo SentencePiece y
el **vocabulario unificado** (texto ∪ imagen ∪ audio) en `tokenizer_info.json`.

```bash
python scripts/train_tokenizer.py --profile m2max --vocab_size 65536 --sample_mb 500
```

### 5.2 Imagen y audio (opcional, para multimodal)

VQ-VAE entrenados desde cero en MLX. Mapean imagen/audio a tokens discretos en el
rango correspondiente del vocabulario unificado.

```bash
# Imagen (por defecto CIFAR-10; o --images-dir con tus imágenes)
python scripts/train_image_tokenizer.py --steps 1500

# Audio (dir de .wav; sin datos genera un corpus sintético para smoke test)
python scripts/train_audio_tokenizer.py --audio-dir path/a/wavs
```

Salida: `dist/tokenizer/image_vqvae.npz` y `dist/tokenizer/audio_vqvae.npz`.

## 6. Entrenar los 5 stages

Currículum progresivo. Cada stage debe pasar su *graduation gate* antes del siguiente
(el perfil `test` lo omite). Se guardan checkpoints automáticamente; `--resume` retoma.

```bash
python train_stage.py --profile m2max --stage 1      # Lenguaje
python train_stage.py --profile m2max --stage 2      # Patrones
python train_stage.py --profile m2max --stage 3      # Abstracción
python train_stage.py --profile m2max --stage 4      # Causalidad
python train_stage.py --profile m2max --stage 5      # Ética + BCF

# Retomar tras Ctrl+C
python train_stage.py --profile m2max --stage 1 --resume
```

| Stage | Métrica del gate | Umbral |
|---|---|---|
| 1 Lenguaje | perplejidad val (proxy de BLiMP) | según perfil |
| 2 Patrones | perplejidad val (proxy de ARC) | según perfil |
| 3 Abstracción | perplejidad val (proxy de GSM8K) | según perfil |
| 4 Causal | perplejidad val | según perfil |
| 5 Ética | perplejidad val + **BCF probe ≥ 0.90** | según perfil |

> Los gates usan **perplejidad de validación** como proxy operativo. Para usar
> benchmarks reales (BLiMP/ARC/GSM8K), reemplazá `evaluate_gate` en `train_stage.py`.

Checkpoints: `dist/checkpoints/<perfil>/stage<N>/` (`step_*.npz`, `latest.json`,
`final.npz`, `stage_complete.json`).

## 7. Congelar + BCF

Al completar el **Stage 5**, el core foundacional se **congela permanentemente** y
se guarda en `dist/checkpoints/<perfil>/foundational/theta_f_frozen.npz`. Si existe
`data/benchmarks/bcf_probes.jsonl` (`{"text": "...", "label": 0|1}` por línea), se
entrena además el head BCF (clasificador de seguridad) y se guarda `bcf_head.npz`.
Esto ocurre automáticamente dentro de `train_stage.py --stage 5`.

A partir de acá el core no se vuelve a tocar: todo el aprendizaje es por consolidación.

## 8. Chatear

```bash
python chat.py --profile m2max --stage 5                 # core + sectores
python chat.py --profile m2max --stage 1 --lang es       # sesión en español
python chat.py --profile m2max --stage 5 --image foto.png   # grounding visual
python chat.py --profile m2max --stage 5 --audio clip.wav   # grounding de audio
```

Comandos dentro del chat: `/lang es` · `/temp 0.7` · `/topp 0.9` · `/maxtok 512`
· `/stats` · `/reset` · `/quit`.

Con `--image`/`--audio`, la **capa de percepción** convierte el archivo a tokens del
vocabulario unificado y los antepone al contexto (salida en texto, Era 3a). Requiere
haber entrenado el tokenizer de esa modalidad (paso 5.2).

Cada turno se registra como **experiencia** en `data/experiences.jsonl` para la
consolidación diaria.

## 9. Consolidación diaria

El daemon corre en tiempo de inactividad (CPU < 20% por 5+ min), drena las
experiencias acumuladas y las consolida: filtro BCF → filtro adversarial (R⁺<0) →
MRF (promover/retener/expirar) → asignación de sector → **update enmascarado del
sector** (el core y los demás sectores quedan intactos) → PGQ (crecimiento de
sectores) → snapshot/rollback → audit log en `logs/cycle_*.json`.

```bash
python consolidation_daemon.py --profile m2max --once    # un ciclo y termina
python consolidation_daemon.py --profile m2max           # daemon (espera idle)
```

Sectores actualizados se guardan en `dist/checkpoints/<perfil>/sectors.npz`; la
memoria de largo plazo en `data/ltss.db`.

## 10. Probar rápido (perfil `test`)

Mismo flujo real, con **menos datos y un modelo chico** (sin "toy", sin datos
sintéticos). Ideal para verificar que todo corre punta a punta en ~10 min.

```bash
python scripts/prepare_data.py   --profile test --stage 1 --limit 50
python scripts/train_tokenizer.py --profile test --vocab_size 8000 --sample_mb 20
python train_stage.py            --profile test --stage 1
python chat.py                   --profile test --stage 1
```

El perfil `test` (`configs/profiles/test.yaml`) usa `skip_gate: true` y apunta todos
los stages al mismo corpus, para poder correr los 5 stages → freeze → consolidación
sin descargar los datasets por etapa.

## 11. Limpieza

```bash
# Datos descargados (regenerables)
rm -rf data/stage*_*/*.jsonl
# Cache de HuggingFace
rm -rf ~/.cache/huggingface/datasets ~/.cache/huggingface/hub
# Pesos del modelo (NO regenerables sin reentrenar)
rm -rf dist/checkpoints/
# Tokenizers (texto + VQ-VAE)
rm -rf dist/tokenizer/
# Snapshots de sectores, logs y memoria
rm -rf snapshots/* logs/* data/ltss.db data/experiences.jsonl
```

## 12. Migración T3/T4

El modelo usa MRL (embeddings anidados). Un modelo grande se puede **truncar hacia
abajo** a un tier menor en inferencia (no al revés): entrená al tamaño que vayas a
usar. Perfiles disponibles: `nano` (~26M), `m2max` (~109M), `a100`, `cluster`
(estos dos definen arquitectura objetivo; correrlos en NVIDIA requiere portar el loop
a CUDA). Ver `docs/reference/architecture.md`.

---

### Validar la hipótesis central (sin datos ni GPU)

Demuestra en segundos que la consolidación sectorizada no olvida, frente a
fine-tuning secuencial y EWC:

```bash
python experiments/continual_learning.py --domains 5 --steps 250
```

Esperado: `naive` peor → `ewc` intermedio → `rdmca` BWT≈0.
