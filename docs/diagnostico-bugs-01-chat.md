# Bug Report: Chat & Reasoning System

**File**: `docs/diagnostico-bugs-01-chat.md`
**Scope**: `uses/chat/run_chat.py`, `src/agent.py`, `src/modalities/text.py` (tokenizer interaction)
**Date**: 2026-06-11

---

## C2 — `<think>` Not a User-Defined Symbol

**Severity**: CRITICAL
**Requires retrain**: Yes (tokenizer + all models)

The `<think>` and `</think>` delimiters are plain text in the prompt, NOT registered as SentencePiece user-defined symbols:

- **File**: `scripts/train_tokenizer.py:135-136` — `user_symbols` list includes `<lang:en>`, `<mod:text>` etc. but not `<think>`
- **File**: `src/agent.py:94-95` — `THINK_INSTRUCTION = " First reason step by step inside <think> </think>, then give the final answer."`
- **File**: `src/data/graded.py:286` — Stage 5 reasoning data uses `<think>` as plain text

**Effect**: `<think>` tokenizes as 4 BPE pieces: `▁<` (ID 1541) + `th` (454) + `ink` (411) + `>` (8125). This 4-piece sequence never appears in training data (TinyStories + dialogue have zero angle brackets).

**Fix**: Add `"<think>"` and `"</think>"` to `user_defined_symbols` in tokenizer training. Update Stage 5 data generation to use the new token IDs.

---

## C3 — Language Token Leak in `generate_thinking`

**Severity**: CRITICAL
**Requires retrain**: No (code fix only)

**File**: `uses/chat/run_chat.py:243`

```python
enc = lambda s: tokenizer.encode(s, lang=lang, add_bos=False, add_eos=False)
```

**Root cause**: `TextTokenizer.encode` (`src/modalities/text.py:62-76`) always inserts `<lang:XX>` for known languages regardless of `add_bos`. This produces:

```
Actual:   ...Assistant:<lang:en><think>...   (IDs: 199, 8077, 4, 8126, 454, 411, 8125)
Expected: ...Assistant:<think>...             (IDs: 199, 8077, 1541, 454, 411, 8125)
```

The `<lang:en>` token (ID 4) shifted from the start of the sequence into the middle, changing the BPE segmentation of `<think>` from `▁<` (1541) to standalone `<` (8126). The resulting token sequence has zero probability under the training distribution → model degenerates into HTML-like tag patterns.

**Empirical**: With `--think off`, no HTML appears. Bug is 100% attributable to the thinking phase.

**Fix**: Add `encode_raw(self, text)` to `TextTokenizer` that calls `self._sp.EncodeAsIds(text)` directly without language prefix. Replace the `enc` lambda with `tokenizer.encode_raw()`.

---

## H4 — context_len = 256 Too Small for Chat

**Severity**: HIGH
**Requires retrain**: Yes

**File**: `configs/levels/level1.yaml:37`

**File**: `uses/chat/run_chat.py:536-538`

```python
max_hist = max(64, mcfg.context_len - max_tokens)
history = history[-max_hist:]
```

With `context_len=256` and `max_tokens=256`: `max_hist = 64`. History truncates to 64 tokens — barely 1-2 turns of conversation before everything is cut off. The model has no memory of earlier context.

**Fix**: Increase `context_len` to 512+.

---

## L5 — `sample_top_p` Uses Global NumPy RNG

**Severity**: LOW
**Requires retrain**: No

**File**: `uses/chat/run_chat.py:103`

```python
return int(np.random.choice(len(probs), p=probs))
```

Uses unseeded global numpy RNG. Generation is not reproducible across runs.

**Fix**: Accept optional `rng` parameter or seed before each generation.

---

## L6 — No top-k or Repetition Penalty in Sampling

**Severity**: LOW
**Requires retrain**: No

**File**: `uses/chat/run_chat.py:88-103`

Only top-p (nucleus) sampling is implemented. No top-k filtering, no repetition penalty, no frequency penalty. The only anti-repetition mechanism is the loop detector (`_MAX_TOKEN_REPEAT = 32`).

**Fix**: Add top-k parameter. Add configurable repetition penalty.

---

## Agent Path: `safe_stream_len` May Hold Back Too Much

**Severity**: LOW
**Requires retrain**: No

**File**: `src/agent.py:174-188`

For role names like "User", checks the longest suffix that could be the start of `"User:"`. If text ends in `"U"`, holds back 1 character. Edge case: single-character suffix matching first char of multiple role tags causes conservative (but correct) truncation.

---

## Agent Path: `clean_answer` Case-Sensitive Role Match

**Severity**: LOW
**Requires retrain**: No

**File**: `src/agent.py:157-167`

`_ROLE_BOUNDARY_RE` is case-sensitive. If the model emits lowercase `user:` or `assistant:`, it won't be caught.

---

## Agent Path: `parse_action` May False-Positive on Plain JSON

**Severity**: LOW
**Requires retrain**: No

**File**: `src/agent.py:233-253`

After stripping thinking, searches for JSON objects. Any JSON with a `name` field is treated as an action, even without the `Action:` prefix. Could cause false positives.

---

## Agent Path: Tool Loop History Grows Unbounded

**Severity**: MEDIUM
**Requires retrain**: No

**File**: `src/agent.py:270-302` (`run_agent`)

The `transcript` list grows unbounded across multi-step tool interactions. For long tool loops, this can exceed `context_len`. No truncation logic.

**Fix**: Add context window management in `run_agent`.

---

## Todo Tool Uses Mutable Global State

**Severity**: LOW
**Requires retrain**: No

**File**: `uses/agent/tools/todo.py:21`

`_PLAN` is a process-global mutable list. Not thread-safe, state shared across runs.

---

## Summary

| ID | Severity | File(s) | Fix |
|----|----------|---------|-----|
| C2 | CRITICAL | `train_tokenizer.py:135`, `agent.py:94`, `graded.py:286` | Register `<think>` as user-defined symbol |
| C3 | CRITICAL | `run_chat.py:243`, `text.py:62-76` | Add `encode_raw()`, use in `enc` lambda |
| H4 | HIGH | `level*.yaml:37`, `run_chat.py:536` | Increase context_len to 512+ |
| L5 | LOW | `run_chat.py:103` | Seed numpy RNG |
| L6 | LOW | `run_chat.py:88-103` | Add top-k + repetition penalty |
| — | LOW | `agent.py:174-188` | Accept known limitation |
| — | MEDIUM | `agent.py:270-302` | Add history truncation in tool loop |
| — | LOW | `todo.py:21` | Use instance-level state |
