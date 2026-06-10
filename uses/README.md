# RDMCA — Usage modes

Ways to consume a trained RDMCA model:

- **[chat/](chat/)** — interactive conversation (text or JSON output, optional
  `<think>…</think>` reasoning).
- **[agent/](agent/)** — agentic tool loop (Claude Code-style): the model emits
  `Action: {…}`, a tool runs, an `Observation` is fed back, repeat. Ships one
  example tool and one example skill.
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

# 1. Prepare the data for every stage in the level (downloads real corpora)
python scripts/prepare_data.py --level 1 --stage all

# 2. Train the tokenizer for the level
python scripts/train_tokenizer.py --level 1

# 3. Train the stages — IN ORDER (each stage starts from the previous one's
#    weights). Level 1's active stages are: 1, 2, 3, 6, 7, 8, 9.
python train_stage.py --level 1 --stage 1     # language
python train_stage.py --level 1 --stage 2     # patterns
python train_stage.py --level 1 --stage 3     # arithmetic
python train_stage.py --level 1 --stage 6     # tool use
python train_stage.py --level 1 --stage 7     # MCP
python train_stage.py --level 1 --stage 8     # skills
python train_stage.py --level 1 --stage 9     # reasoning (chain-of-thought)

# 4. Chat (text / JSON output, optional reasoning)
python uses/chat/run_chat.py --level 1 --stage 9
python uses/chat/run_chat.py --level 1 --stage 9 --format json
python uses/chat/run_chat.py --level 1 --stage 9 --think medium   # show <think> scratchpad

# 5. Agent (tool loop with the example tool + skill)
python uses/agent/run_agent.py --level 1 --stage 9 --query "What time is it?"
```

### What to test
- **Conversation / arithmetic**: `run_chat.py` — ask simple questions and sums
  (the model should do arithmetic itself; that's the stage-3 skill).
- **Output format**: `run_chat.py --format json` (or `/format json` at runtime).
- **Reasoning**: `run_chat.py --think medium` (or `/think medium`) — the model
  writes a `<think>…</think>` scratchpad (shown in the chat) before answering.
  The level is an effort/budget dial: off · low · medium · high.
- **Tool use**: `run_agent.py` with a date/time question → the model should emit
  an `Action` calling `get_current_time`; the runner executes it and feeds back
  the `Observation`. (The example tool is deliberately *not* a calculator, so it
  never masks the model's own arithmetic.)

> Tip: to verify the plumbing without training, add `--dummy` (random weights).
