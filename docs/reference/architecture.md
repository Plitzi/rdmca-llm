# Arquitectura y estructura del proyecto

## Modelo — T2 Edge (256 dims)

| Componente | Valor |
|---|---|
| Arquitectura | Decoder-only transformer (GPT-style) |
| d_model | 256 (T2 Edge) |
| Capas | 8 |
| Attention heads | 4 |
| FFN dim | 1024 (SwiGLU) |
| Context length | 2048 tokens |
| Vocab size | 65 536 (bilingüe EN+ES) |
| Positional encoding | RoPE |
| Normalización | RMSNorm (pre-norm) |
| MRL dims | [64, 128, 256] |
| Parámetros foundational | ~31 M |
| Sectores LoRA (7×) | ~42 M |
| **Total** | **~73 M** |
| Training precision | BF16 |
| Inference | INT8 (~2 GB activos) |

### Sectores LoRA

| ID | Nombre | Dominio | Rango |
|---|---|---|---|
| S1 | Linguistic | Conversación, estilo, discurso | r=16 |
| S2 | Formal | Matemáticas, lógica, simbólico | r=16 |
| S3 | WorldKnowledge | Factual, enciclopédico | r=8 |
| S4 | Procedural | Planificación, herramientas | r=8 |
| S5 | Social | Pragmática, normas sociales | r=8 |
| S6 | Multimodal | Cross-modal (Phase 3+) | r=8 |
| S7 | Behavioral | Ética, BCF — solo adversarial buffer | r=4 |

---

## Estructura del proyecto

```
rdmca-llm/
├── src/
│   ├── model/
│   │   ├── transformer.py      RDMCAFoundational + ModelConfig
│   │   ├── lora.py             7 sectores LoRA + gradient masking
│   │   └── bcf.py              Behavioral Constraint Function head
│   ├── memory/
│   │   ├── episodic_buffer.py  T1 buffer en memoria
│   │   ├── ltss.py             SQLite + FAISS long-term store
│   │   └── mrf.py              Memory Reevaluation Function
│   ├── relevance/
│   │   ├── engine.py           R+(e,s): N, U, C, Rep, P
│   │   └── penalty.py          Taxonomía de ataques adversariales
│   ├── routing/
│   │   ├── semantic_router.py  STR: segmentación + affinity classifier
│   │   └── sector_router.py    Asignación de sector s* para consolidación
│   ├── consolidation/
│   │   ├── pipeline.py         Ciclo completo de 9 pasos
│   │   ├── snapshot.py         Snapshots rolling 7 días + rollback + CAT
│   │   ├── ambiguity.py        Deferral + human review queue
│   │   └── pgq.py              Parametric Growth Quantifier
│   ├── modalities/
│   │   ├── text.py             SentencePiece wrapper
│   │   ├── image.py            VQVAE stub (Phase 3)
│   │   └── audio.py            EnCodec stub (Phase 4)
│   ├── inference/
│   │   └── generate.py         Generación autoregresiva + nucleus sampling
│   └── data/
│       └── loader.py           Streaming JSONL DataLoader
├── scripts/
│   ├── make_toy_data.py        Corpus sintético local (sin descarga)
│   ├── prepare_data.py         Descarga Wikipedia EN+ES + datasets por stage
│   └── train_tokenizer.py      Entrena SentencePiece BPE bilingüe
├── configs/
│   ├── rdmca_t2.yaml           Config de producción (4.5B tokens)
│   └── rdmca_t2_toy.yaml       Config de prueba rápida (~10 min)
├── docs/
│   ├── papers/                 Papers de referencia (.docx)
│   ├── 1-setup.md
│   ├── 2-data.md
│   ├── 3-training.md
│   ├── 4-chat.md
│   ├── 5-eval.md
│   ├── 6-cleanup.md
│   └── 7-architecture.md      ← este archivo
├── tests/
│   ├── test_phase1.py          10 tests activos, 6 skipped
│   ├── test_phase2.py
│   ├── test_phase3.py          Todos skipped (Phase 3)
│   └── test_phase4.py          Todos skipped (Phase 4)
├── data/stage{1-5}_*/          Corpus generado por prepare_data.py
├── dist/tokenizer/                  rdmca_spm.model (generado)
├── dist/checkpoints/                Pesos por etapa
├── dist/snapshots/                  Backups de sectores LoRA (Phase 2+)
├── logs/                       Audit logs de consolidación
├── train_stage.py              Entrenamiento por etapas
├── chat.py                     Chat interactivo
└── consolidation_daemon.py     Daemon de consolidación (Phase 2+)
```

---

## Migrar a hardware mayor (T3 / T4)

El modelo usa MRL (Matryoshka Representation Learning). Los embeddings están
entrenados en dims anidadas [64, 128, 256], lo que permite migrar sin reentrenar.

```python
import mlx.core as mx
weights = mx.load("dist/checkpoints/foundational/theta_f_frozen.npz")

# T3 Standard (512 dims): usar prefijo de 512 dimensiones
embedding_t3 = weights["embedding"][:, :512]

# T4 Large (1024 dims): usar prefijo de 1024 dimensiones
embedding_t4 = weights["embedding"][:, :1024]
```

| Tier | d_model | Params | Hardware target |
|---|---|---|---|
| T2 Edge | 256 | ~73 M | M2 Max 64 GB |
| T3 Standard | 512 | ~225 M | M3 Ultra / A100 |
| T4 Large | 1024 | ~800 M | H100 / multi-GPU |
