"""
Curriculum stage constants — the SINGLE source of truth for stage gates, names and
the freeze point. Both the trainer (`train_stage.py`) and the dashboard
(`src/training/dashboard.py`) import from here, so a new stage or a changed
threshold can't silently diverge between them.

`STAGE_GATES[stage] = (metric_key, threshold, label)`:
  - metric_key — the benchmark this gate will use once wired (today the gate uses a
    validation-perplexity proxy; see train_stage.evaluate_gate),
  - threshold  — pass mark for that metric,
  - label      — human description shown in logs / the dashboard.
"""
from __future__ import annotations

# BCF_STAGE is the ethics stage and the LATEST possible freeze point. The actual
# freeze happens after the last ACTIVE cognitive stage (train_stage.last_cognitive_stage),
# which is BCF_STAGE when ethics is present and an earlier stage otherwise. Behavioral
# stages (>BCF_STAGE: tool/MCP/skills) train as LoRA sectors on the frozen core.
BCF_STAGE = 6

STAGE_GATES = {
    1: ("blim_accuracy",      0.70, "Language — BLiMP grammaticality"),
    2: ("arc_easy_accuracy",  0.60, "Patterns — ARC easy"),
    3: ("gsm8k_accuracy",     0.15, "Abstraction — GSM8K"),
    4: ("causal_accuracy",    0.65, "Causal and procedural reasoning"),
    5: ("reasoning_accuracy", 0.20, "Reasoning — chain-of-thought (GSM8K)"),
    6: ("bcf_accuracy",       0.90, "Cognitive ethics — BCF probe"),
}

STAGE_NAMES = {
    1: "Language and communication",
    2: "Perception and pattern recognition",
    3: "Abstraction and symbolic composition",
    4: "Causal and procedural reasoning",
    5: "Reasoning",
    6: "Cognitive ethics and BCF",
    7: "Action and tool use",
    8: "Model Context Protocol (MCP)",
    9: "Skills",
}
