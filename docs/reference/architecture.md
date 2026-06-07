# Arquitectura y estructura del proyecto

## Modelo

Decoder-only transformer (GPT-style) con RoPE, RMSNorm (pre-norm), FFN SwiGLU y
pérdida MRL (Matryoshka) sobre dims anidadas. El tamaño concreto lo fija el perfil
(`configs/profiles/*.yaml`); los valores por defecto del config base son d_model=256,
8 capas, 4 heads, FFN 1024, contexto 2048.

| Componente | Valor (config base) |
|---|---|
| Arquitectura | Decoder-only transformer |
| Positional encoding | RoPE |
| Normalización | RMSNorm (pre-norm) |
| FFN | SwiGLU |
| MRL dims | [64, 128, 256] |
| Core | foundational congelable (Θ_F) + 7 sectores LoRA |
| Precisión | BF16 (entrenamiento) |

### Sectores LoRA

| ID | Nombre | Dominio | Rango |
|---|---|---|---|
| S1 | Linguistic | Conversación, estilo, discurso | r=16 |
| S2 | Formal | Matemáticas, lógica, simbólico | r=16 |
| S3 | WorldKnowledge | Factual, enciclopédico | r=8 |
| S4 | Procedural | Planificación, herramientas | r=8 |
| S5 | Social | Pragmática, normas sociales | r=8 |
| S6 | Multimodal | Cross-modal (imagen/audio ↔ texto) | r=8 |
| S7 | Behavioral | Ética, BCF — solo adversarial buffer | r=4 |

Los sectores se actualizan **uno a la vez** por consolidación, con *gradient masking*
real (MLX freeze/unfreeze): el core y los demás sectores quedan bit-idénticos. El PGQ
puede **crecer el rango** de un sector (`SectorAdapter.grow_rank`) o **crear sectores
nuevos** (`model.add_sector`) en runtime, preservando la salida (componentes nuevos a
cero al inicio).

---

## Vocabulario unificado (multimodal, Era 3b)

Texto, imagen y audio comparten **una sola tabla de embeddings**. Los rangos son
disjuntos y se persisten en `dist/tokenizer/tokenizer_info.json` (`modality_layout`):

```
text  = [0,            Vt)          SentencePiece (Vt = text_vocab_size)
image = [Vt,           Vt+8192)     codebook del VQ-VAE de imagen
audio = [Vt+8192,      Vt+8192+4096) codebook del VQ-VAE de audio
vocab_size (modelo) = total
```

- Tokens de modalidad (`<mod:text> <mod:image> <mod:audio> <mod_end>`) y de idioma
  (`<lang:xx>`) son *user-defined symbols* dentro del rango de texto.
- Los idiomas son **config-driven** (`model.languages`); el tokenizer hornea los
  `<lang:xx>` elegidos y guarda `lang_token_ids` en `tokenizer_info.json`.
- La **capa de percepción** (`src/modalities/perception.py`) detecta modalidad,
  tokeniza con el tokenizer correspondiente y ensambla la secuencia interleaved.
- El `DataLoader` acepta registros `{"text": ...}` o pre-tokenizados `{"tokens": [...]}`
  (multimodal); el objetivo LM de next-token es el mismo para toda modalidad.

---

## Estructura del proyecto

```
rdmca-llm/
├── src/
│   ├── config.py               Config + idiomas + tokenizer_info (fuente única)
│   ├── model/
│   │   ├── transformer.py       RDMCAFoundational + ModelConfig + add_sector
│   │   ├── lora.py              7 sectores LoRA + grad masking + grow_rank
│   │   └── bcf.py              Behavioral Constraint Function head
│   ├── memory/
│   │   ├── episodic_buffer.py  T1 buffer + Experience
│   │   ├── ltss.py             SQLite (embeddings persistidos) + búsqueda numpy
│   │   ├── mrf.py              Memory Reevaluation Function
│   │   └── experience_log.py   Cola de experiencias chat → daemon
│   ├── relevance/
│   │   ├── engine.py           R+(e,s): N, U, C, Rep − λ·P
│   │   └── penalty.py          Taxonomía de ataques (filtro adversarial)
│   ├── routing/
│   │   ├── semantic_router.py  STR: segmentación + affinity classifier
│   │   └── sector_router.py    Asignación de sector s* para consolidación
│   ├── consolidation/
│   │   ├── pipeline.py         Ciclo completo de consolidación
│   │   ├── snapshot.py         Snapshots 7 días + rollback + CAT
│   │   ├── ambiguity.py        Deferral + human review queue
│   │   └── pgq.py              Parametric Growth Quantifier (expand / new sector)
│   ├── modalities/
│   │   ├── vocab.py            Layout del vocab unificado (offsets)
│   │   ├── text.py             SentencePiece wrapper (idiomas config-driven)
│   │   ├── image.py            ImageVQVAE (conv VQ-VAE en MLX)
│   │   ├── audio.py            AudioVQVAE (log-mel VQ-VAE en MLX)
│   │   ├── vq.py               VectorQuantizer compartido
│   │   └── perception.py       Multimodal Perception Layer (MPL)
│   ├── data/loader.py          DataLoader (texto + pre-tokenizado multimodal)
│   └── training/dashboard.py   Dashboard de entrenamiento (rich)
├── scripts/
│   ├── prepare_data.py         Descarga corpus por idioma + datasets por stage
│   ├── train_tokenizer.py      SentencePiece + vocab unificado
│   ├── train_image_tokenizer.py  Entrena el VQ-VAE de imagen
│   ├── train_audio_tokenizer.py  Entrena el VQ-VAE de audio
│   └── prepare_multimodal.py   Grounding interleaved imagen/audio-texto
├── configs/
│   ├── rdmca_t2.yaml           Config base
│   └── profiles/               test · nano · m2max · a100 · cluster
├── tests/                      test_phase1..4 (modelo, consolidación, multimodal, PGQ)
├── experiments/continual_learning.py   Validación de la hipótesis (no-forgetting)
├── train_stage.py              Entrenamiento por stages + freeze + BCF
├── chat.py                     Chat interactivo (texto / --image / --audio)
├── consolidation_daemon.py     Daemon de consolidación diaria (cableado)
├── GUIDE.md                    Guía única paso a paso
└── docs/{papers,reference}/    Paper + esta referencia
```

Checkpoints: `dist/checkpoints/<perfil>/stage<N>/`, core congelado en
`.../foundational/theta_f_frozen.npz`, sectores en `.../sectors.npz`.
Tokenizers en `dist/tokenizer/`. Memoria de largo plazo en `data/ltss.db`.

---

## Perfil `test`

`configs/profiles/test.yaml` reemplaza al viejo "toy": **mismo flujo real** con un
modelo chico, pocos tokens y `skip_gate: true`. Apunta todos los stages al mismo
corpus para correr los 5 stages → freeze → consolidación sin descargar los datasets
por etapa. Es solo para verificar el pipeline; los pesos no sirven para producción.

---

## Consolidación (daemon)

`consolidation_daemon.py` carga el core congelado + sectores, drena
`data/experiences.jsonl` y ejecuta `ConsolidationPipeline`: filtro BCF → filtro
adversarial (R⁺<0) → consistencia LTSS → MRF → asignación de sector (STR + SectorRouter)
→ update enmascarado por sector → PGQ → snapshot/rollback → audit log en
`logs/cycle_*.json`. Guarda los sectores en `dist/checkpoints/<perfil>/sectors.npz`.

---

## Migración a hardware mayor (T3 / T4)

El modelo usa MRL: los embeddings están entrenados en dims anidadas, lo que permite
**truncar hacia abajo** un modelo grande a un tier menor en inferencia (no ampliar uno
chico). Entrená al tamaño que vayas a usar.

```python
import mlx.core as mx
w = mx.load("dist/checkpoints/<perfil>/foundational/theta_f_frozen.npz")
emb_t3 = w["embed.weight"][:, :512]   # prefijo de 512 dims
```

| Perfil | d_model aprox | Hardware objetivo |
|---|---|---|
| test | 256 (4 capas) | smoke test |
| nano | 384 | MacBook M2/M3 |
| m2max | 512 | MacBook M2/M3 Max 64 GB |
| a100 | 768 | 1× A100 (requiere backend CUDA) |
| cluster | 1024 | multi-GPU (requiere backend CUDA) |
