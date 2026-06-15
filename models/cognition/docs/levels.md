# RDMCA Levels — what each level is and exactly what it adds

A **level** (`models/cognition/configs/levels/level{0..5}.yaml`, selected with `--level N`) sets the
model size, the graded-data complexity/budget and the context window. The guiding
rule: **a level's size follows the *information* it must learn, not the hardware** —
the hardware only caps *how high* a level you can run (the startup resource guard
aborts a level that won't fit; `--force` overrides).

**Every level runs the SAME 10 curriculum stages** (the same faculties, the same
structure, the same freeze point). A level is NOT "a new faculty turned on" — it is
the *same model, grown*. Concretely, a higher level changes only three things:

1. **More parameters** (wider `d_model`, more `n_layers`, bigger `vocab`).
2. **More / harder data** (bigger token budgets, looser readability filters).
3. **A longer context window** (`context_len`, grown Fibonacci-style).

So the smallest model already works the *same way* as the largest; growing a level
just makes every faculty deeper. This keeps training and debugging uniform — a bug or
an improvement applies identically at every scale.

## The ladder at a glance

| Lvl | Grade | d_model | layers | vocab | ctx | ~params* | backend | gate |
|----|-------|---------|--------|-------|-----|----------|---------|------|
| 0 | Pruebas (smoke) | 64  | 2  | 4 096  | 64    | ~0.6M  | mlx   | skip |
| 1 | Preescolar      | 256 | 6  | 8 192  | 512   | ~11M   | mlx   | skip |
| 2 | Primaria        | 256 | 8  | 8 192  | 768   | ~14M   | mlx   | yes  |
| 3 | Secundaria      | 384 | 8  | 16 384 | 1 280 | ~36M   | mlx   | yes  |
| 4 | Bachillerato    | 512 | 12 | 24 576 | 2 048 | ~98M   | torch | yes  |
| 5 | Universidad     | 768 | 16 | 32 768 | 3 328 | ~260M  | torch | yes  |

\* params are **weight-tied** (input embedding = output head; one `[vocab, d_model]`
matrix counted once) and now **include the per-level MTP heads + PLE lookup tables**
(the optional efficiency knobs — lookup-heavy, near-zero extra FLOPs). The dense
compute-core is smaller (e.g. L1's core ≈ 8.4M). The context window follows a
**Fibonacci ladder** (512 · 768 · 1280 · 2048 · 3328). L1 and L2 share the d_model=256
tier; L1 is the lighter 6-layer variant (capacity was never L1's bottleneck —
data/training was), L2 carries the tier's full 8-layer spec. Distinct sizes resume at L3.

## Curriculum stages (the faculties — present at EVERY level)

Seven **cognitive** stages train into the dense **core** in developmental order;
after the ethics/BCF stage the core is **frozen forever**, and three **behavioral**
stages train as **LoRA sectors** on top of it. This is identical at every level —
only the data/params/context differ.

| Stage | Faculty | Type | Gate metric |
|-------|---------|------|-------------|
| 1 | **Language** & communication (converse) | cognitive (core) | BLiMP grammaticality |
| 2 | **Perception** & pattern recognition (analogies, sequences) | cognitive (core) | ARC-easy |
| 3 | **Abstraction** & symbolic composition (arithmetic) | cognitive (core) | GSM8K |
| 4 | **Causal** & procedural reasoning | cognitive (core) | causal |
| 5 | **Reasoning** — chain-of-thought | cognitive (core) | GSM8K CoT |
| 6 | **Memory management** — recall & USE injected memory | cognitive (core) | memory recall |
| 7 | **Cognitive ethics + BCF** — values · **FREEZE POINT** | cognitive (core) | BCF probe |
| 8 | **Action & tool use** | behavioral (LoRA sector) | — |
| 9 | **Model Context Protocol (MCP)** | behavioral (LoRA sector) | — |
| 10 | **Skills** composition (SKILL.md) | behavioral (LoRA sector) | — |

**Freeze point — after stage 7, at EVERY level:** once ethics/BCF (stage 7) completes,
the cognitive core is frozen permanently. From then on the behavioral stages (8–10)
train as **LoRA sectors**, the **MoE gate** routes per token (safety sector S7 always
on, isolated), and **daily consolidation** (learning from operational experience, the
confidence-gated validation, the human-review queue) is available. The freeze point
(`bcf_stage()` from `src.plugins`, the stage whose plugin declares `is_freeze_point` — 7
for cognition) drives the cognitive-vs-behavioral split and the freeze — there is no
per-level special-casing.

> **Memory (stage 6)** is the cognitive faculty that learns to *recall a fact given in
> context and use it* — trained on `<mem>…</mem>` blocks of facts + distractors. This is
> the capability behind "answer based on what the user said / the conversation"; it does
> NOT exist after stage 1 alone — it is taught at stage 6.

---

## What each level adds (same 10 stages throughout)

## Level 0 — Pruebas (smoke)
- **Model:** 64d · 2L · 4 096 vocab · ctx 64 · ~0.6M. Trains end-to-end in **minutes**.
- **Purpose:** validate the FULL pipeline (prepare → tokenize → train all 10 stages →
  freeze after 7 → LoRA sectors 8–10 → chat/agent) runs without breaking. **Not a quality
  model** — a wiring test, at tiny per-stage budgets (≈1M tokens). `skip_gate`.

## Level 1 — Preescolar
- **Model:** 256d · 6L · 8 192 vocab · ctx 512 · ~11M. `skip_gate`.
- **Data:** rich-but-basic 4-register Stage-1 (TinyStories + everyday dialogue + short
  instruction Q&A + Simple-Wikipedia), readability grade ≤ 4; single-digit arithmetic;
  graded-small versions of every later faculty.
- **Can do:** a tiny but COMPLETE model — basic coherent conversation, simple answers,
  single-digit add/subtract, easy step-by-step reasoning, basic memory recall, basic
  tool/MCP/skill format. Every faculty is present but **shallow** (it's a preescolar).
- **vs L0:** the first *usable* base — real (still basic) competence on broad varied
  data instead of a smoke test.

## Level 2 — Primaria
- **Model:** 256d · **8L** · 8 192 vocab · ctx **768** · ~14M. **Graduation gate ON**.
- **Data:** larger budgets; harder content — sentences/paragraphs, everyday vocabulary,
  prose grade ≤ 5, **two-digit +, −, ×**, more varied analogies.
- **vs L1:** **capacity + complexity + context, not new faculties.** Two more layers (the
  full 256-tier spec), a longer window, more/harder data, plus a real **graduation gate**
  (each stage must clear a validation bar to advance).

## Level 3 — Secundaria
- **Model:** **384d** · 8L · 16 384 vocab · ctx **1 280** · ~36M. Gate ON.
- **Data:** general knowledge (**full Wikipedia**, grade ≤ 9), multi-digit / fractions /
  simple algebra, ARC-easy patterns.
- **vs L2:** wider model, real-world knowledge breadth, harder math, longer context — the
  *same faculties*, deeper.

## Level 4 — Bachillerato
- **Model:** **512d · 12L** · 24 576 vocab · ctx **2 048** · ~98M. **`torch` (CUDA).**
  Gradient checkpointing + 8-bit optimizer states ON (fits the deeper stack on less VRAM).
- **Data:** advanced text (grade ≤ 13), **GSM8K word problems**, ARC-challenge, richer
  ethics.
- **vs L3:** a big capacity jump and the move to PyTorch/CUDA, longer context — still the
  same 10 stages, deeper.

## Level 5 — Universidad
- **Model:** **768d · 16L** · 32 768 vocab · ctx **3 328** · ~260M. `torch`.
- **Data:** **no filters** — full corpora at every stage, plus **competition math (MATH)**.
- **vs L4:** mastery, not new faculties — removes every readability/complexity filter
  (full-difficulty text and math), competition-level math, the largest model and window.

---

## Notes
- **Train in order:** each stage starts from the previous stage's weights; cognitive
  stages **1→2→3→4→5→6→7 (freeze)**, then behavioral **8→9→10** as LoRA sectors on the
  frozen core. This order is identical at every level.
- **All-stages-everywhere:** because each stage's `entry_level ≤ 1`, the full curriculum
  runs even at L1 — so the smallest model is a complete (if shallow) RDMCA, and the
  10-stage pipeline can be debugged cheaply before scaling up.
- **MRL truncation:** embeddings are trained over nested dims, so a larger model can be
  **truncated down** to a smaller tier at inference (`embed.weight[:, :d]`) — not the
  other way around. Train at the size you will use.
- **Backends:** L0–L3 default to MLX (Apple Silicon laptop); L4–L5 to PyTorch (CUDA).
  Either backend runs any level; the default is just the typical hardware.
- Commands to prepare/tokenize/train/chat a level are in [GUIDE.md](GUIDE.md);
  architecture details in [reference/architecture.md](reference/architecture.md).
