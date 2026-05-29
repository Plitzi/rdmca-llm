← [1-setup.md](1-setup.md) | → [3-training.md](3-training.md)

# Preparación de datos

## Dónde se guardan los archivos

| Ubicación | Qué hay | Cuándo se crea |
|---|---|---|
| `data/stage{1-5}_*/` | Corpus JSONL procesados para training | `prepare_data.py` |
| `~/.cache/huggingface/` | Cache temporal de HuggingFace | Durante la descarga |
| `dist/tokenizer/` | Modelo SentencePiece entrenado | `train_tokenizer.py` |

---

## Opción A — Datos de prueba rápida (sin descarga, instantáneo)

Genera un corpus sintético de ~6 MB en menos de 1 segundo.
Útil para verificar que el pipeline funciona antes de descargar nada.

```bash
python scripts/make_toy_data.py
# Resultado: data/stage1_language/toy_en.jsonl + toy_es.jsonl
```

Luego seguir con el tokenizador y el config toy (ver [4-training.md](../guides/4-training.md)).

---

## Opción B — Datos reales desde HuggingFace (~18 GB)

Wikipedia EN + ES es el backbone. Se descarga una vez y se filtra por etapa.

### Tamaños por stage

| Stage | Datasets | Download | En disco |
|---|---|---|---|
| 1 Language | Wikipedia EN + ES (todo) | ~8 GB | ~16 GB |
| 2 Patterns | Wikipedia EN+ES (ciencia/lógica) + ARC | ~3 GB | ~6 GB |
| 3 Abstraction | Wikipedia EN+ES (mat.) + GSM8K + mGSM-ES + MATH | ~3 GB | ~6 GB |
| 4 Causal | Wikipedia EN+ES (ingeniería/medicina) | ~3 GB | ~6 GB |
| 5 Ethics | Wikipedia EN+ES (filosofía) + ethics EN+ES | ~1 GB | ~2 GB |
| **Total** | | **~18 GB** | **~36 GB** |

### Comandos de descarga

```bash
# Stage por stage — recomendado, podés pausar entre etapas
python scripts/prepare_data.py --stage 1   # descarga EN + ES por defecto
python scripts/prepare_data.py --stage 2
python scripts/prepare_data.py --stage 3
python scripts/prepare_data.py --stage 4
python scripts/prepare_data.py --stage 5

# Todos juntos
python scripts/prepare_data.py --stage all

# Solo inglés (mitad del tamaño, misma lógica)
python scripts/prepare_data.py --stage 1 --lang en

# Test rápido con 100 MB por idioma (sin descargar todo)
python scripts/prepare_data.py --stage 1 --limit 100
```

El script es **resumible**: si se interrumpe, saltea los archivos que ya existen.

---

## Entrenar el tokenizador

Debe correrse **después** de tener datos del Stage 1 (toy o reales).

```bash
python scripts/train_tokenizer.py
# Con datos toy (rápido, ~30 seg):
python scripts/train_tokenizer.py --sample_mb 5

# Resultado:
# dist/tokenizer/rdmca_spm.model
# dist/tokenizer/rdmca_spm.vocab
```

El tokenizador es bilingüe EN+ES, vocab=65 536, con soporte para diacríticos
españoles (á é í ó ú ü ñ ¡ ¿).

---

## Formato de los archivos de datos

Todos los archivos son `.jsonl` — una línea por documento:
```json
{"text": "El sol sale por el este.", "lang": "es"}
{"text": "The sun rises in the east.", "lang": "en"}
```

También se aceptan `.txt` (un documento por línea) en `data/stage{N}_*/`.
