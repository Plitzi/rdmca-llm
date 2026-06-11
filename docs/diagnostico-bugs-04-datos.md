# Bug Report: Data Pipeline

**File**: `docs/diagnostico-bugs-04-datos.md`
**Scope**: `scripts/prepare_data.py`, `src/data/loader.py`, `src/data/graded.py`, `src/modalities/text.py`, tokenizer
**Date**: 2026-06-11

---

## H5 — Dialogue Data Unfiltered With 3× Oversampling

**Severity**: HIGH
**Requires retrain**: Yes

**File**: `scripts/prepare_data.py:230` (`_READABILITY_FILTERED`)

```python
_READABILITY_FILTERED = {"tinystories", "simple_wikipedia", "wikipedia"}  # dialogue missing!
```

TinyStories passes Flesch-Kincaid readability filter (max_grade=2). Dialogue does not.

**File**: `configs/levels/level1.yaml:85-89`

```yaml
source_weights: {dialogue: 3.0}
```

Dialogue data quality issues found in sample inspection:
- Single-word turns: "User: 5 Assistant: 4 User: 5 Assistant: 7"
- Non-sequiturs: "User: What languages is Open Assistant written in? Assistant: Math."
- Grammar errors: "kiil", emotional outbursts
- Garbage: "HdhdebhdrbduwbeuebhdrvwjshHabHgVJsndbBbabdbdbsbdjdjqjsbrjebjejshajJJjJ"
- Duplicates: "sassy teenage daughter" prompt appears 5+ times in first 20k
- Very short: "hi" / "hi" / "hai,hello" / "how are you" / "fine you?"

**Effective token mix** (with weight 3.0):
- tinystories: 40M × 1.0 = 40M (46%)
- dialogue: 14M × 3.0 = 42M (54%)

The model spends more than half its training on low-quality dialogue.

**Fix**: Add `dialogue` to `_READABILITY_FILTERED`. Add minimum utterance/turn filter (>2 utterances, >10 words). Reduce weight to 1.0 or lower.

---

## H4 — `context_len = 256` Too Small for Documents

**Severity**: HIGH
**Requires retrain**: Yes

**File**: `configs/levels/level1.yaml:37`

**File**: `src/data/loader.py:157-175`

Documents are concatenated into a flat stream and sliced into `[B, S+1]` batches. Documents exceeding `context_len` are truncated across batch boundaries.

**TinyStories overflow analysis** (142,354 docs, avg 255 words ≈ 331 BPE tokens):
| Threshold | Documents exceeding | % |
|-----------|-------------------|--|
| 256 tokens | 89,881 | 63.1% |
| 128 tokens | 138,195 | 97.1% |
| 64 tokens | 142,331 | 100.0% |

Only 2.9% of TinyStories documents fit entirely within 128 tokens. The model never sees complete stories.

**Other sources**:
| Source | % exceeding 256 tokens | Mean tokens | Max tokens |
|--------|----------------------|-------------|------------|
| agentic/tools | 100% | 1,047 | 2,300 |
| mcp | 100% | 1,177 | 2,466 |
| skills | 97% | 514 | 3,201 |
| reasoning | ~35% | 413 | 2,366 |
| tinystories | 63% | 331 | 961 |
| dialogue | ~1% | 94 | 315 |

**Fix**: Increase `context_len` to 512 (level1/2) or 1024+. Increase level 0 from 64 to 128+.

---

## H8 — Document Concatenation Without Attention Masking

**Severity**: HIGH
**Requires retrain**: Yes (changes data format)

**File**: `src/data/loader.py:157-175`

Documents are tokenized with BOS+lang+EOS and concatenated into a flat buffer. Batches are contiguous `[B, S+1]` slices without document boundary isolation. A single batch typically contains:

```
End of doc A (20 tokens), BOS, <lang:en>, tokens(doc_B), EOS, BOS, ...
```

The model is trained to predict transitions like:
- Last token of doc A → BOS
- BOS → `<lang:en>`
- EOS → BOS (cross-document)

These spurious continuations consume training capacity without teaching useful language patterns.

**Fix**: Implement document-level attention masking (block out cross-document attention), or align batch boundaries with document boundaries. For current sizes, document as known limitation.

---

## H6 — CHARS_PER_TOKEN Underestimates Actual Token Count by 37%

**Severity**: HIGH
**Requires retrain**: Yes (budget changes)

**File**: `scripts/prepare_data.py:138`

```python
CHARS_PER_TOKEN = 4.5
```

Measured actual ratios across Level 1 sources:

| Source | Actual chars/token | Error |
|--------|-------------------|-------|
| arithmetic | 1.17 | +285% |
| analogies | 2.31 | +95% |
| agentic/tools | 3.05 | +48% |
| reasoning | 2.97 | +52% |
| skills | 3.59 | +25% |
| dialogue | 3.84 | +17% |
| tinystories | 3.92 | +15% |
| **Weighted avg** | **3.29** | **+37%** |

The data preparation script miscalculates token budgets by 37% on average, and up to 285% for structured data. This means:
- Sources appear to exhaust their budgets faster than expected
- Structured sources (arithmetic, analogies) contribute far more tokens than intended
- The "budget-based" curatorial control is inaccurate

**Fix**: Measure empirically per source. Set `CHARS_PER_TOKEN` to ~3.3 for general text, or compute per-source conversion factors.

---

## H7 — No Held-Out Validation Set

**Severity**: HIGH
**Requires retrain**: Yes (data + pipeline)

**File**: `train_stage.py:211-223, 492`

All HuggingFace dataset loads use `split="train"` exclusively. Validation batches are sampled from the training data stream. Gate perplexity measures training fit, not generalization.

**All affected source files** in `src/data/graded.py`: lines 95, 153, 263, 385, 463, 504, 539, 611, 654, 671 — all `split="train"`.

**Fix**: During data preparation (`scripts/prepare_data.py`), reserve 5% of each source for held-out validation. Create separate validation `DataLoader` in `train_stage.py`.

---

## M5 — GSM8K Data Leakage Between Stages

**Severity**: MEDIUM
**Requires retrain**: Yes (data selection)

**Files**: `configs/levels/level4.yaml:83` + `src/data/graded.py:374-415`

At levels 4+:
- **Stage 3** includes `gsm8k` as a direct source (raw QA format)
- **Stage 5** internally loads GSM8K as part of the `reasoning` stream (CoT format)

Same underlying question pool, same samples, different formatting. The model sees the same questions twice at different training stages.

Additionally, the rehearsal mechanism (`train_stage.py:183-193`, 15% replay from prior stages) mixes data without deduplication, causing further overlap.

**Fix**: Add `gsm8k` to an exclusion list when building the stage 5 reasoning stream. Or tag datasets by source ID and filter during rehearsal.

---

## Ethics Source Is Only 12 Hardcoded Snippets

**Severity**: LOW
**Requires retrain**: N/A

**File**: `scripts/prepare_data.py:258-275`

The `ethics` source is not a real dataset — it's 12 hardcoded bilingual EN/ES text snippets. For a task labeled "Cognitive ethics and BCF" (Stage 6), this is a placeholder that will not produce meaningful BCF probe training.

**Fix**: Replace with a real ethics dataset (e.g., ETHICS, CommonsenseQA, or curated moral scenarios).

---

## Analogies Source Is 8 Hardcoded Pairs

**Severity**: LOW
**Requires retrain**: N/A

**File**: `src/data/graded.py:117-133`

The `analogies` source is not a real dataset — it's 8 hardcoded synthetic analogy pairs (e.g., "king:queen :: man:woman"). Despite producing a 183 MB file (likely massively duplicated or synthetic expansion), the actual semantic content is trivial.

**Fix**: Replace with a proper analogy dataset (e.g., Google Analogy Questions, BATS, or SAT analogies).

---

## L10 — `.meta.json` Sidecar Files Missing

**Severity**: LOW
**Requires retrain**: No

**File**: `scripts/prepare_data.py:413-414`

The `.meta.json` sidecar files that track source exhaustion are not present in the prepared data directory. Re-running `prepare_data` will re-process all sources instead of skipping complete ones.

---

## Summary

| ID | Severity | File(s) | Fix |
|----|----------|---------|-----|
| H5 | HIGH | `prepare_data.py:230`, level configs | Add dialogue filter, reduce weight |
| H4 | HIGH | Configs (context_len), `loader.py` | Increase to 512+ |
| H8 | HIGH | `loader.py:157-175` | Document masking or alignment |
| H6 | HIGH | `prepare_data.py:138` | Fix CHARS_PER_TOKEN to ~3.3 |
| H7 | HIGH | `train_stage.py:492`, `graded.py` (all) | Add held-out validation |
| M5 | MEDIUM | Configs, `graded.py:374-415` | Exclude GSM8K from stage 5 |
| — | LOW | `prepare_data.py:258-275` | Replace ethics placeholder |
| — | LOW | `graded.py:117-133` | Replace analogies placeholder |
| L10 | LOW | `prepare_data.py:413-414` | Write meta files on data prep |
