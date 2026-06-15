"""Validation-batch construction for the graduation gate, including the RETENTION
gate that folds held-out conversation into a later cognitive stage's val set."""

from __future__ import annotations

from src.training.curriculum import is_behavioral_stage


def _val_split_batches(stage: int, cfg: dict, n: int):
    """`n` (tokens, mask) batches from a stage's held-out split (`*.val.jsonl`), or []
    if there is no usable split (missing / empty / sub-one-batch). Completion-masked."""
    from src.data.loader import DataLoader
    from src.modalities.text import TextTokenizer

    try:
        vloader = DataLoader.from_config(stage, cfg, TextTokenizer(), val=True, with_mask=True)
        return [vloader.next_batch() for _ in range(n)]
    except (FileNotFoundError, KeyError, StopIteration):
        return []


def make_val_batches(stage: int, cfg: dict, train_loader, n: int = 8):
    """Fixed validation batches as (tokens, mask) pairs. The mask is the completion-only
    loss mask, so the gate measures perplexity on the SAME (assistant) tokens training
    optimizes.

    RETENTION GATE: for a later cognitive stage (2..BCF) the val set FOLDS IN held-out
    CONVERSATION (stage 1) alongside the stage's own data, so a stage that erodes
    conversation ratchets/fails instead of 'passing' at ppl ~1 by memorizing its narrow
    skill. Prefers held-out splits; falls back to the training stream for the stage's slice."""
    own = _val_split_batches(stage, cfg, n)
    src_own = "held-out split" if own else None
    if not own:
        # The training loader already yields (tokens, mask) pairs — mask matches training.
        own = [train_loader.next_batch() for _ in range(n)]
        src_own = "training stream (no *.val.jsonl — run prepare_data for a disjoint gate)"

    if is_behavioral_stage(stage) or stage <= 1:
        print(f"  [val] {src_own} — {len(own)} batches (completion-masked)")
        return own

    # Retention: half conversation (stage 1), half the stage's own skill. Conversation
    # is the priority skill and the one most eroded, so it anchors the gate.
    half = max(n // 2, 1)
    conv = _val_split_batches(1, cfg, half)
    if not conv:
        print(
            f"  [val] {src_own} — {len(own)} batches; NO stage-1 split for retention "
            f"(run prepare_data on stage 1) — gate measures the new skill only"
        )
        return own
    mixed = conv + own[:half]
    print(
        f"  [val] RETENTION gate — {len(conv)} conversation (stage 1) + {len(own[:half])} "
        f"stage-{stage} batches (completion-masked); a stage that forgets conversation fails"
    )
    return mixed
