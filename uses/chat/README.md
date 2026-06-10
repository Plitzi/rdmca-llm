# Chat usage

Interactive conversation with a trained model. Implemented by
[`run_chat.py`](run_chat.py).

```bash
python uses/chat/run_chat.py --level 1 --stage 8                 # text output (default)
python uses/chat/run_chat.py --level 1 --stage 8 --format json   # JSON output
python uses/chat/run_chat.py --dummy                              # random weights (plumbing only)
```

Runtime commands inside the chat:

| Command            | Effect                                  |
|--------------------|-----------------------------------------|
| `/format text\|json` | switch output format                  |
| `/lang es\|en`     | switch language                         |
| `/temp 0.7`        | sampling temperature                    |
| `/topp 0.9`        | nucleus sampling p                      |
| `/maxtok 256`      | max new tokens per turn                 |
| `/stats`           | last-generation stats                   |
| `/reset`           | clear history                           |
| `/quit`            | exit                                    |

Output format is centralized in [`src/agent.py`](../../src/agent.py)
(`wrap_prompt` / `parse_output`) — the same hook the future API will reuse.

Re-run from the repo root (the script puts the repo on `sys.path` itself).
