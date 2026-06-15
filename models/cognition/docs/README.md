# cognition — docs

`cognition` is the conversational/agentic LLM (the 10-stage curriculum, the default model).
These docs are cognition-specific; framework-level docs live in
[../../../docs/README.md](../../../docs/README.md).

- [GUIDE.md](GUIDE.md) — the full step-by-step: setup → backend/precision → languages →
  data → tokenizer → train the cognitive base → freeze/BCF → chat/agent → daily consolidation.
- [levels.md](levels.md) — what each level (0–5) is and exactly what it adds (size, data,
  context); the single source of truth for the per-level curriculum.
- [reference/architecture.md](reference/architecture.md) — the model architecture: the
  transformer, MoE sectors, unified vocab, checkpoints and scaling.
- [papers/](papers/) — the RDMCA theory paper + implementation guide.

Run it with `rdmca <prepare|tokenizer|train|chat|agent> --level N` (model defaults to
`cognition`).
