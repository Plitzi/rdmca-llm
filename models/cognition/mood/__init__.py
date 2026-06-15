"""Cognition mood feature — emotional-state head + taxonomy, owned by this model.

Moods are specific to the conversational cognition model (emotions don't apply to,
say, a hand-pose detector), so all of it lives here, not in the framework core:
  • `lexicon`  — palette, emotion→mood map, SYSTEM-channel annotation, the `moods`
                 switch. Light (no backend); the stage data generators use it.
  • `head`     — the neural MoodHead + tracker/classifier and train/load helpers
                 (uses the backend); chat, agent and the training hook use it.

`post_stage` is the model's stage-completion hook: the trainer calls it after a
cognitive stage finishes (discovered by name on this package — see
src.training.heads), and it trains+saves the mood head when the stage is
conversational and moods are enabled. This is how the agnostic core trains a
cognition-specific head without ever importing this package.
"""

from __future__ import annotations

from pathlib import Path

from models.cognition.mood.head import (
    MOOD_INDEX,
    MoodHead,
    MoodTracker,
    classify_mood,
    load_mood_head,
    mood_loss,
    train_mood_head,
)
from models.cognition.mood.lexicon import (
    MOOD_MARGIN,
    MOODS,
    NEUTRAL,
    emotion_to_mood,
    lexicon_mood,
    mood_system_phrase,
    moods_enabled,
)

__all__ = [
    "MOODS",
    "MOOD_INDEX",
    "MOOD_MARGIN",
    "NEUTRAL",
    "MoodHead",
    "MoodTracker",
    "classify_mood",
    "emotion_to_mood",
    "lexicon_mood",
    "load_mood_head",
    "mood_loss",
    "mood_system_phrase",
    "moods_enabled",
    "post_stage",
    "train_mood_head",
]


def post_stage(model, stage: int, cfg: dict, ckpt_dir: Path, precision: str) -> None:
    """Stage-completion hook for the cognition model: train + save the conversation
    mood head beside this stage's checkpoint. Best-effort and OFF the critical path —
    gated by the `moods` switch and the stage's `trains_mood` flag, and silently
    skipped if the labeled data is unavailable (it must never fail a stage)."""
    from src.plugins import get_stage

    if not moods_enabled(cfg):
        return
    # Mood is a CONVERSATIONAL faculty — only (re)train it on conversational stages
    # (stage 1 + the frozen-core BCF stage). The stage plugin declares this.
    if not get_stage(stage).trains_mood:
        print(
            f"  [mood] stage {stage} is not conversational — keeping the existing "
            "head (chat falls back to the nearest earlier head + lexicon)"
        )
        return
    try:
        from src.modalities.text import TextTokenizer

        train_mood_head(
            model,
            TextTokenizer(),
            ckpt_dir,
            level=cfg.get("level"),
            stage=stage,
            precision=precision,
        )
    except Exception as e:
        print(f"  [mood] skipped ({type(e).__name__}: {e})")
