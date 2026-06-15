# RDMCA documentation

The docs are split **framework vs model** — the same separation as the code (`src/` is the
task-agnostic framework; each model lives under `models/<model>/`).

The **framework** itself (the model/stage plugin system, the `ModelSpec` seam, how to add a
model) is documented in [CLAUDE.md](../CLAUDE.md). Everything model-specific — including the
network/model architecture and the theory papers — lives with its model.

## Framework (task-agnostic)

- [FAQ/](FAQ/) — troubleshooting (e.g. CUDA).
- [future-features-reports/](future-features-reports/) — design notes (e.g. distributed training).

## Models (each carries its own docs)

- **cognition** (the conversational/agentic LLM) — [GUIDE](../models/cognition/docs/GUIDE.md)
  (setup → backend → data → tokenizer → train → chat/agent → consolidation),
  [levels](../models/cognition/docs/levels.md) (the per-level curriculum),
  [architecture](../models/cognition/docs/reference/architecture.md) (model, sectors, unified
  vocab, scaling) and [papers](../models/cognition/docs/papers/) (theory + implementation guide).
- **hands_recognition** (hand-pose / skeleton, a non-text model) —
  [GUIDE](../models/hands_recognition/docs/GUIDE.md) (camera use with FPS/skeleton overlay,
  training the pose net).

Discover what exists with `rdmca info [--model M] [--level L]`.
