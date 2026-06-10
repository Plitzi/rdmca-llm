# Chat usage

Interactive conversation with a trained model. Implemented by
[`run_chat.py`](run_chat.py).

```bash
python uses/chat/run_chat.py --level 1 --stage 9                 # text output (default)
python uses/chat/run_chat.py --level 1 --stage 9 --format json   # JSON output
python uses/chat/run_chat.py --level 1 --stage 9 --think medium  # show reasoning first
python uses/chat/run_chat.py --dummy                              # random weights (plumbing only)
```

Runtime commands inside the chat:

| Command            | Effect                                  |
|--------------------|-----------------------------------------|
| `/format text\|json` | switch output format                  |
| `/think off\|low\|medium\|high` | reasoning effort (see below) |
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
| `off`    | 0                 | answer directly (default)         |
| `low`    | 0.25 · maxtok     | brief reasoning                   |
| `medium` | 0.5 · maxtok      | moderate reasoning                |
| `high`   | 1.0 · maxtok      | reasons until `</think>` or budget|

When thinking is active the chat **shows the scratchpad** (`💭 thinking: …`)
above the answer. Generation is two-phase: a budget-capped scratchpad is
force-closed, then the answer is generated from a well-formed `… </think>`
prefix. Disabled on the vocab-ID fallback (needs a real tokenizer). The dial is
centralized in [`src/agent.py`](../../src/agent.py) (`normalize_thinking` /
`think_budget` / `split_thinking`), the same hook the agent runner and future
API reuse.

Output format is centralized in [`src/agent.py`](../../src/agent.py)
(`wrap_prompt` / `parse_output`) — the same hook the future API will reuse.

Re-run from the repo root (the script puts the repo on `sys.path` itself).
