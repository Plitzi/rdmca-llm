---
name: time-helper
description: Use this skill when the user asks about the current date or time ("what time is it", "what's today's date", "what day is it").
---

# Time Helper

The current date and time are not something you can know on your own. When the
user asks about them:

1. Call the `get_current_time` tool:
   `Action: {"name": "get_current_time", "input": {}}`
2. Read the `Observation` and answer in one short sentence,
   e.g. "It is 2026-06-10, 14:05 UTC (Wednesday)."

Only use this skill for date/time questions. For anything else, answer normally.
