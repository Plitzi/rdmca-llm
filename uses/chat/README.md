# Chat usage

Interactive conversation with a trained model. Implemented by
[`run_chat.py`](run_chat.py).

```bash
python uses/chat/run_chat.py --level 1 --stage 9                 # streamed, reasoning=medium (defaults)
python uses/chat/run_chat.py --level 1 --stage 9 --format json   # JSON output
python uses/chat/run_chat.py --level 1 --stage 9 --think high     # more reasoning
python uses/chat/run_chat.py --level 1 --stage 9 --no-stream      # batched (no live tokens)
python uses/chat/run_chat.py --level 1 --stage 9 --quant int4     # 4-bit weights (limited hardware)
python uses/chat/run_chat.py --dummy                              # random weights (plumbing only)
```

Runtime commands inside the chat:

| Command            | Effect                                  |
|--------------------|-----------------------------------------|
| `/format text\|json` | switch output format                  |
| `/think off\|low\|medium\|high` | reasoning effort (default medium) |
| `/stream on\|off`  | live token streaming (default on)       |
| `/lang es\|en`     | switch language                         |
| `/temp 0.7`        | sampling temperature                    |
| `/topp 0.9`        | nucleus sampling p                      |
| `/maxtok 256`      | max new tokens per turn                 |
| `/stats`           | last-generation stats                   |
| `/reset`           | clear history                           |
| `/quit`            | exit                                    |

### Thinking / reasoning

The model learned a reasoning register in **stage 9**: it can write a
`<think>…</think>` scratchpad before the answer (mirrors Claude's thinking
blocks). The `--think` flag / `/think` command is an effort dial — whether to
think at all and how big a token budget the scratchpad gets (a fraction of
`--maxtok`):

| Level    | Scratchpad budget | Behaviour                         |
|----------|-------------------|-----------------------------------|
| `off`    | 0                 | answer directly                   |
| `low`    | 0.25 · maxtok     | brief reasoning                   |
| `medium` | 0.5 · maxtok      | moderate reasoning **(default)**  |
| `high`   | 1.0 · maxtok      | reasons until `</think>` or budget|

**Default is `medium`** — more thinking generally means better answers, so
reasoning is on out of the box (drop to `low`/`off` for speed). When thinking is
active the chat **shows the scratchpad** (`💭 thinking: …`) above the answer.
Generation is two-phase: a budget-capped scratchpad is force-closed, then the
answer is generated from a well-formed `… </think>` prefix. Disabled on the
vocab-ID fallback (needs a real tokenizer). The dial is centralized in
[`src/agent.py`](../../src/agent.py) (`normalize_thinking` / `think_budget` /
`split_thinking`), the same hook the agent runner and future API reuse.

### Streaming

Tokens stream live by default (`--stream` / `/stream on`) so the conversation
feels fluid — the scratchpad and answer are decoded and printed as they
generate. Use `--no-stream` (or `/stream off`) for a single batched print.
Streaming needs a real tokenizer; it falls back to batched output otherwise.

### Quantization (limited hardware)

`--quant int8|int4` quantizes the weights at load time so the model fits in less
memory: **int8 ≈ ¼** and **int4 ≈ ⅛** of fp32 resident size. It's real
grouped-affine quantization on both backends (MLX native; torch weight-only with
packed nibbles at 4-bit). The output head stays in float (most quant-sensitive,
and the MRL logic slices its weight directly). 4-bit trades some accuracy for the
memory win — fine for testing on small machines.

### Anti-loop guards

Reasoning can't wedge a turn. Generation is bounded by `--maxtok`, and on top of
that a **loop detector** stops a degenerate repeating cycle early and a
wall-clock deadline (`--max-seconds`, default 90s, `0` = off) caps runaway turns.
These fire only on *stuck* output (repetition / stalls), so genuine long
reasoning is never cut short — a looping `<think>` scratchpad is detected and
closed, and the model still produces its answer. The scratchpad token budget
(`think_budget`) is a generous resource bound (clamped to `MAX_THINK_TOKENS`),
not the loop defense.

Output format is centralized in [`src/agent.py`](../../src/agent.py)
(`wrap_prompt` / `parse_output`) — the same hook the future API will reuse.

Re-run from the repo root (the script puts the repo on `sys.path` itself).
