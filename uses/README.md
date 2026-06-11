# RDMCA — Usage modes

Ways to consume a trained RDMCA model:

- **[chat/](chat/)** — interactive conversation (text or JSON output, optional
  `<think>…</think>` reasoning).
- **[agent/](agent/)** — agentic tool loop (Claude Code-style): the model emits
  `Action: {…}`, a tool runs, an `Observation` is fed back, repeat. Ships two
  example tools (a clock + a `todo` planning aid) and one example skill.
- **api/** — HTTP serving. _TODO (later)._

---

## Which level should I use for real testing?

**Use level 1.** It is the smallest *real* (non-smoke) level — ~2M params,
trains on a laptop in a reasonable time, and exercises every real data source
including the agentic stages (tool use, MCP, skills).

- **Level 0** is a throwaway *smoke* tier (2 layers, gibberish output) — only for
  verifying the pipeline runs, not for real behavior.
- **Levels 2–5** are progressively larger and need much more data/compute
  (level 5 is cluster-grade, ~201M params). Move up once level 1 works.

> Honesty note: at ~2M params, level 1 output is limited and imperfect. The
> *pipeline and behaviors* are real; *quality* scales with level/compute.

---

## Full steps to test (level 1)

```bash
# 0. (optional) higher HF download limits — put HF_TOKEN in .env (see .env.example)
cp .env.example .env        # then edit HF_TOKEN=...

# (to start completely fresh: python scripts/purge.py --all --dry-run, then --yes)

# 1. Prepare the data for every stage in the level (downloads real corpora)
python scripts/prepare_data.py --level 1 --stage all

# 2. Train the tokenizer for the level
python scripts/train_tokenizer.py --level 1

# 3. Train the stages — IN ORDER (each starts from the previous one's weights).
#    Natural order: cognition (1-5) → values (6, freeze) → behavioral (7-9).
#    Level 1's active stages are: 1, 2, 3, 5, 7, 8, 9 (causal/ethics enter at L3/L4).
python train_stage.py --level 1 --stage 1     # language
python train_stage.py --level 1 --stage 2     # patterns
python train_stage.py --level 1 --stage 3     # arithmetic
python train_stage.py --level 1 --stage 5     # reasoning (chain-of-thought) — capstone of the base
python train_stage.py --level 1 --stage 7     # tool use
python train_stage.py --level 1 --stage 8     # MCP
python train_stage.py --level 1 --stage 9     # skills

# 4. Chat — streamed live, reasoning=medium by default
python uses/chat/run_chat.py --level 1 --stage 9
python uses/chat/run_chat.py --level 1 --stage 9 --format json
python uses/chat/run_chat.py --level 1 --stage 9 --think high --no-stream

# 5. Agent (multi-round tool loop with the example tool + skill)
python uses/agent/run_agent.py --level 1 --stage 9 --query "What time is it?"
```

### What to test
- **Conversation / arithmetic**: `run_chat.py` — ask simple questions and sums
  (the model should do arithmetic itself; that's the stage-3 skill).
- **Output format**: `run_chat.py --format json` (or `/format json` at runtime).
- **Reasoning**: on by default at `medium` — the model writes a `<think>…</think>`
  scratchpad (shown in the chat) before answering. Effort/budget dial:
  off · low · medium · high (`--think` / `/think`).
- **Streaming**: tokens stream live by default (`--no-stream` / `/stream off` to
  batch) so chat and agent feel fluid.
- **Multi-round agent**: the agent runs several think→act→observe rounds until it
  answers (Claude Code-style); each round's reasoning + tool call is surfaced.
- **Limited hardware**: `--quant N` loads weights quantized to any 2–8 bit width
  (real on both backends); `--quant int4` (≈⅛ size) is the low-resource testing
  tier, `int8` ≈¼ size. For *training* bigger levels on the same machine, lower the
  precision (`train_stage.py --precision bf16`) — the memory guard is precision-aware.
- **Anti-loop**: a degenerate-thinking loop or runaway turn is stopped by a loop
  detector + `--max-seconds` deadline; genuine long reasoning is never truncated.
- **Tool use**: `run_agent.py` with a date/time question → the model should emit
  an `Action` calling `get_current_time`; the runner executes it and feeds back
  the `Observation`. (The example tool is deliberately *not* a calculator, so it
  never masks the model's own arithmetic.)

> Tip: to verify the plumbing without training, add `--dummy` (random weights).
