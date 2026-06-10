# API usage — TODO (later)

HTTP serving for the RDMCA agent. **Not implemented yet** (deferred by request).

When built, it should reuse the existing hooks so CLI and API stay in sync:

- **Output format** — `src/agent.py`: `wrap_prompt()` / `parse_output()`
  (request field `format: "text" | "json"`).
- **Tool loop** — `src/agent.py`: `Tool`, `run_agent()` (the agentic loop already
  used by [`uses/agent/run_agent.py`](../agent/run_agent.py)).
- **Tools / skills** — discover from `uses/agent/tools/` and `uses/agent/skills/`.

Sketch of the intended surface (subject to change):

```
POST /chat    { "message": str, "format": "text"|"json", "level": int, "stage": int }
POST /agent   { "message": str, "tools": [...], "skill": str, "max_steps": int }
```

Likely stack: FastAPI + uvicorn (add to requirements.txt when implemented).
