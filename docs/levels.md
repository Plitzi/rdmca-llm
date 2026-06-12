# RDMCA Levels — what each level is and exactly what it adds

A **level** (`configs/levels/level{0..5}.yaml`, selected with `--level N`) sets the
model size, the graded-data complexity and the active curriculum stages. The guiding
rule: **a level's size follows the *information* it must learn, not the hardware** —
the hardware only caps *how high* a level you can run (the startup resource guard
aborts a level that won't fit; `--force` overrides).

Each higher level does one or more of: **(a)** grow the model, **(b)** raise data
complexity / budgets, **(c)** activate a NEW cognitive faculty (a new stage). This doc
makes the "(c) — what's genuinely new" explicit, since it's otherwise buried in the
configs.

## The ladder at a glance

| Lvl | Grade | d_model | layers | vocab | ctx | ~params* | backend | gate |
|----|-------|---------|--------|-------|-----|----------|---------|------|
| 0 | Pruebas (smoke) | 64  | 2  | 4 096  | 64   | ~0.4M  | mlx   | skip |
| 1 | Preescolar      | 256 | 6  | 8 192  | 512  | ~8.4M  | mlx   | skip |
| 2 | Primaria        | 256 | 8  | 8 192  | 512  | ~10.5M | mlx   | yes  |
| 3 | Secundaria      | 384 | 8  | 16 384 | 512  | ~25M   | mlx   | yes  |
| 4 | Bachillerato    | 512 | 12 | 24 576 | 1 024| ~63M   | torch | yes  |
| 5 | Universidad     | 768 | 16 | 32 768 | 2 048| ~176M  | torch | yes  |

\* params are **weight-tied** (input embedding = output head; one `[vocab, d_model]`
matrix counted once). L1 and L2 share the d_model=256 tier; L1 is the lighter 6-layer
variant (capacity was never L1's bottleneck — data/training was), L2 carries the tier's
full 8-layer spec. Distinct sizes resume at L3.

## Curriculum stages (the faculties a level can teach)

The **frozen cognitive base** is six stages in developmental order; three **behavioral**
stages then train as LoRA sectors on the frozen core. A stage runs at a level only if
its `entry_level ≤ level`.

| Stage | Faculty | Enters at |
|-------|---------|-----------|
| 1 | **Language** & communication (converse) | L1 |
| 2 | **Patterns** / perception (analogies, sequences) | L1 |
| 3 | **Arithmetic** / symbolic composition | L1 |
| 4 | **Causal** & procedural reasoning | **L3** |
| 5 | **Reasoning** — chain-of-thought | L1 |
| 6 | **Ethics + BCF** — values · **freeze point** | **L4** |
| 7 | **Tool use** (behavioral / LoRA sector) | all |
| 8 | **MCP** integration (behavioral) | all |
| 9 | **Skills** composition (behavioral) | all |

**Freeze point (stage 6, L4+):** after the last cognitive stage the base is frozen
forever; from then on the behavioral stages (7–9) train as **LoRA sectors**, the **MoE
gate** routes per token, and **daily consolidation** (learning from operational
experience, the confidence-gated validation, the human-review queue) turns on. **At
L1–L3 there are no sectors — you chat with the pure dense base.**

---

## Level 0 — Pruebas (smoke)
- **Model:** 64d · 2L · 4 096 vocab · ~0.4M params. Trains end-to-end in **minutes**.
- **Stages:** 1, 2, 3, 5, 7, 8, 9 — all at tiny budgets (1–4M tokens each).
- **Purpose:** validate the full pipeline (prepare → tokenize → train → chat) works.
  **Not a quality model** — it's a wiring test. `skip_gate`.

## Level 1 — Preescolar
- **Model:** 256d · 6L · 8 192 vocab · ctx 512 · ~8.4M. `skip_gate`. Pure dense base.
- **Stages:** 1 Language, 2 Patterns, 3 Arithmetic (1-digit +/−), 5 Reasoning, + behavioral 7/8/9.
- **Data:** rich-but-basic 4-register Stage-1 (TinyStories + everyday dialogue + short
  instruction Q&A + Simple-Wikipedia), readability grade ≤ 4; single-digit arithmetic.
- **Can do:** hold a **basic coherent conversation**, answer simple requests, count and
  do single-digit add/subtract, think step-by-step on easy problems, basic tool/MCP/skill format.
- **Adds vs L0:** the first *real* (still tiny) usable base — actual conversational
  competence and arithmetic, on broad varied data, instead of a smoke test.

## Level 2 — Primaria
- **Model:** 256d · **8L** · 8 192 vocab · ~10.5M. **Graduation gate ON** (`skip_gate:false`).
- **Stages:** same faculties as L1 (1, 2, 3, 5, 7/8/9) — **no new faculty**.
- **Data:** larger budgets; harder content — sentences/paragraphs, everyday vocabulary,
  prose grade ≤ 5, **two-digit +, −, ×**, more varied analogies.
- **Adds vs L1:** **capacity + complexity, not new faculties.** Two more layers (the full
  256-tier spec), more/harder data → more fluent conversation and bigger arithmetic, plus
  a real **graduation gate** (the stage must clear a validation-perplexity bar to advance).

## Level 3 — Secundaria
- **Model:** **384d** · 8L · 16 384 vocab · ~25M. Gate ON.
- **Stages:** 1, 2, 3, **4 (NEW)**, 5, 7/8/9.
- **Data:** general knowledge (**full Wikipedia**, grade ≤ 9), arithmetic level 3
  (multi-digit / fractions / simple algebra), ARC-easy patterns.
- **Adds vs L2:** 🆕 **Causal & procedural reasoning (stage 4)** — the first genuinely new
  faculty since L1. Wider model, real-world knowledge breadth, harder math.

## Level 4 — Bachillerato
- **Model:** **512d · 12L** · 24 576 vocab · ctx 1 024 · ~63M. **`torch` (CUDA/cluster).**
- **Stages:** 1, 2, 3, 4, **6 (NEW)**, 5, 7/8/9.
- **Data:** advanced text (grade ≤ 13), **GSM8K word problems**, ARC-challenge,
  causal + **ethics**.
- **Adds vs L3:** 🆕 **Ethics + BCF (stage 6) — the FREEZE POINT.** This is the structural
  turning point: after L4's cognitive stages the **base freezes**, and behavioral stages
  (7–9) become **LoRA sectors** with **MoE routing** and **daily consolidation** (the
  operational learning-from-experience loop, confidence-gated validation, human-review
  queue) — none of which exist below L4. Plus a big capacity jump and the move to PyTorch.

## Level 5 — Universidad
- **Model:** **768d · 16L** · 32 768 vocab · ctx 2 048 · ~176M. `torch`.
- **Stages:** all (1–9) — **no new faculty vs L4**.
- **Data:** **no filters** — full corpora at every stage, plus **competition math (MATH)**.
- **Adds vs L4:** mastery, not new faculties: removes every readability/complexity filter
  (full-difficulty text and math), adds competition-level math, and the largest model.

---

## Notes
- **Train in order:** each stage starts from the previous stage's weights; cognitive
  stages 1→…→(freeze), then behavioral 7→8→9 on the frozen core.
- **MRL truncation:** embeddings are trained over nested dims, so a larger model can be
  **truncated down** to a smaller tier at inference (`embed.weight[:, :d]`) — not the
  other way around. Train at the size you will use.
- **Backends:** L0–L3 default to MLX (Apple Silicon laptop); L4–L5 to PyTorch (CUDA).
  Either backend runs any level; the default is just the typical hardware.
- Commands to prepare/tokenize/train/chat a level are in [GUIDE.md](GUIDE.md);
  architecture details in [reference/architecture.md](reference/architecture.md).
