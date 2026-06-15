# Agent usage (tool loop)

Drives the model through a Claude Code / Anthropic SDK-style loop: the model
emits `Action: {"name","input"}`, the runner executes the matching tool, feeds
back an `Observation`, and repeats until the model answers without an Action.
The loop itself lives in [`models/cognition/uses/common/agent.py`](../common/agent.py) (`run_agent`) and
runs several think→act→observe rounds (reasoning on by default) until it answers.

```bash
python models/cognition/uses/agent/run_agent.py --level 1 --stage 10 --query "What time is it?"
python models/cognition/uses/agent/run_agent.py --dummy --query "hi"        # plumbing only
```

The same cross-surface features as the chat apply here: KV-cache decoding,
`<mem>` memory recall, and the optional STR per-sector context slots (§12) via
`--context-slots` — it routes each `Action`/`Observation` step block to its
sector slot and evicts overflow to memory, while the header (system+tools+skill+
user) stays pinned. Off by default (recent steps are char-windowed instead);
best with trained sectors.

## Layout

```
models/cognition/uses/agent/
  run_agent.py            # wires the model + tools + skill into the loop
  tools/
    current_time.py       # example tool (TOOL: get_current_time)
    todo.py               # example planning aid (TOOL: todo) — used when planning
  skills/
    time-helper/SKILL.md  # example skill (Claude-style frontmatter + steps)
```

## Add a tool
Copy `tools/current_time.py`: expose a `TOOL` of type `models.cognition.uses.common.agent.Tool` with a
`name`, `description`, JSON `input_schema`, and a `run(input: dict)` function
returning JSON-serializable output. Register it in `run_agent.py`'s `TOOLS` list.

## Add a skill
Create `skills/<name>/SKILL.md` with Claude-style frontmatter (`name`,
`description` = when to use it) and step-by-step instructions. Select it with
`--skill <name>`.

## Why the example tool is a clock (not a calculator)
Arithmetic is a *learned skill* (stage 3) — the model should do it itself. A
calculator tool would mask whether it actually learned arithmetic. The current
time is something the model genuinely cannot know, so it's a clean tool-use test.

> Tool-call quality scales with the level/training. Small levels may not emit
> valid `Action` JSON yet; the loop degrades gracefully to a direct answer.
