# Agent usage (tool loop)

Drives the model through a Claude Code / Anthropic SDK-style loop: the model
emits `Action: {"name","input"}`, the runner executes the matching tool, feeds
back an `Observation`, and repeats until the model answers without an Action.
The loop itself lives in [`src/agent.py`](../../src/agent.py) (`run_agent`).

```bash
python uses/agent/run_agent.py --level 1 --stage 8 --query "What time is it?"
python uses/agent/run_agent.py --dummy --query "hi"        # plumbing only
```

## Layout

```
uses/agent/
  run_agent.py            # wires the model + tools + skill into the loop
  tools/
    current_time.py       # example tool (TOOL: get_current_time)
  skills/
    time-helper/SKILL.md  # example skill (Claude-style frontmatter + steps)
```

## Add a tool
Copy `tools/current_time.py`: expose a `TOOL` of type `src.agent.Tool` with a
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
