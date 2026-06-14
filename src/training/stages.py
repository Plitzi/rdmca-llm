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
#
# Memory management (6) is a COGNITIVE faculty: the frozen core itself learns to
# consume recalled memory (LTSS + episodic), so it sits inside the base, right
# after reasoning (5) and before ethics/BCF (7, the freeze point). Inserting it
# shifted the old ethics/tool/MCP/skills stages up by one (6→7, 7→8, 8→9, 9→10).
BCF_STAGE = 7

STAGE_GATES = {
    1: ("blim_accuracy",      0.70, "Language — BLiMP grammaticality"),
    2: ("arc_easy_accuracy",  0.60, "Patterns — ARC easy"),
    3: ("gsm8k_accuracy",     0.15, "Abstraction — GSM8K"),
    4: ("causal_accuracy",    0.65, "Causal and procedural reasoning"),
    5: ("reasoning_accuracy", 0.20, "Reasoning — chain-of-thought (GSM8K)"),
    6: ("memory_accuracy",    0.50, "Memory — recall and use of injected memory"),
    7: ("bcf_accuracy",       0.90, "Cognitive ethics — BCF probe"),
}

STAGE_NAMES = {
    1: "Language and communication",
    2: "Perception and pattern recognition",
    3: "Abstraction and symbolic composition",
    4: "Causal and procedural reasoning",
    5: "Reasoning",
    6: "Memory management",
    7: "Cognitive ethics and BCF",
    8: "Action and tool use",
    9: "Model Context Protocol (MCP)",
    10: "Skills",
}

# ── Per-stage anti-forgetting profile — a STAGE property, applies at EVERY level ──────
# Levels differ only in data/params/context; a stage's NATURE does not. Stage 3 is
# narrow low-entropy arithmetic at level 1 and at level 5 alike, so its tendency to
# overwrite the shared conversational core (the observed 'hi'→'3' / 'The answer is N'
# collapse) is the same everywhere. These defaults therefore live with the stage, not in
# each level's yaml, and ALL levels inherit them. A level's yaml may still override per
# stage (curriculum.stageN.rehearsal_fraction / .lr_scale) for a genuine exception.
#
# REHEARSAL: fraction of batches drawn from earlier stages (conversation-weighted) so a
# new faculty doesn't erode prior ones. Higher for the narrowest/most-eroding stages.
# Behavioral stages (>BCF) train LoRA sectors on the FROZEN core → no rehearsal needed.
STAGE_REHEARSAL = {2: 0.35, 3: 0.45, 4: 0.35, 5: 0.45, 6: 0.35, 7: 0.35}
DEFAULT_REHEARSAL = 0.15            # cognitive stage with no specific profile

# LR_SCALE: multiplies the stage's peak/min LR. The narrow eroders nudge the shared core
# (they learn their trivial skill fast even at half LR) instead of stamping their format
# over conversation. Stage 1 (conversation) and behavioral stages train at full LR.
STAGE_LR_SCALE = {2: 0.7, 3: 0.5, 4: 0.7, 5: 0.5, 6: 0.7, 7: 0.7}
DEFAULT_LR_SCALE = 1.0

# ── Mood is a CONVERSATIONAL faculty ──────────────────────────────────────────────
# Mood tracks the emotional tone of human/user interaction, so the mood head is only
# meaningful — and only has the (neutral + emotional) data to train on — at stages
# that are conversational. The narrow cognitive stages (patterns/arithmetic/causal/
# reasoning/memory) carry NO conversational signal: training a mood head there had no
# neutral data and produced a near-random classifier (the 'im good'→angry bug). So we
# train it ONLY at stage 1 (where conversation is learned) and again at the BCF stage
# (the core's final frozen state, what the shipped model and behavioral stages use).
# Inference at any other stage falls back to the nearest earlier head (see load_mood_head).
MOOD_TRAIN_STAGES = {1, BCF_STAGE}


def is_mood_stage(stage: int) -> bool:
    """Whether the mood head should be (re)trained at this stage's completion."""
    return stage in MOOD_TRAIN_STAGES
