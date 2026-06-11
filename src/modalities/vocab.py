"""
Unified multimodal vocabulary layout — single source of truth for token ranges.

The foundational model embeds ONE token space (RDMCA §7.2, Era 3b): text, image
and audio tokens occupy disjoint, contiguous ranges of the same embedding table.

    text  = [0,                       Vt)
    image = [Vt,                      Vt + Vi)
    audio = [Vt + Vi,                 Vt + Vi + Va)   = total

Modality boundary tokens (`<mod:text> <mod:image> <mod:audio> <mod_end>`) live
inside the text range as SentencePiece user-defined symbols, so they get stable
ids and are emitted/embedded like any other text token.

This module has no heavy imports so both the tokenizer scripts (no MLX) and the
runtime (MLX) can use it.
"""
from __future__ import annotations
from typing import Dict, List

IMAGE_VOCAB_SIZE = 8192
AUDIO_VOCAB_SIZE = 4096

# Order matters: these become user-defined symbols appended after the language
# tags at tokenizer-training time.
MODALITY_SPECIALS: List[str] = ["<mod:text>", "<mod:image>", "<mod:audio>", "<mod_end>"]

# ── Control symbols (single source of truth) ─────────────────────────────────
# Structural delimiters the curriculum data uses. They MUST be registered as
# SentencePiece user-defined symbols, or BPE splits e.g. "<think>" into
# ['▁<','th','ink','>'] — a combo the prose corpus never contains, so the stage
# that introduces it (reasoning, tool use, …) can't learn a clean boundary.
#
# SCALABLE CONTRACT: when a new stage/source introduces a structural marker, add
# it to the matching list here — `tokenizer_symbols()` is what train_tokenizer
# feeds to SentencePiece, so every level's tokenizer covers every stage's tokens.
REASONING_SPECIALS: List[str] = ["<think>", "</think>"]
TOOL_SPECIALS:      List[str] = ["<tool_call>", "</tool_call>",
                                 "<tool_response>", "</tool_response>"]
CONTROL_SPECIALS:   List[str] = REASONING_SPECIALS + TOOL_SPECIALS


def tokenizer_symbols(langs: List[str]) -> List[str]:
    """All user-defined symbols a tokenizer must reserve: per-language tags +
    modality boundaries + every stage's control delimiters. One call so the set
    stays consistent across levels and grows automatically as stages add markers."""
    return ([f"<lang:{l}>" for l in langs] + list(MODALITY_SPECIALS)
            + list(CONTROL_SPECIALS))


def build_modality_layout(text_vocab_size: int,
                          image_vocab_size: int = IMAGE_VOCAB_SIZE,
                          audio_vocab_size: int = AUDIO_VOCAB_SIZE) -> Dict:
    """Return offsets/sizes for each modality range and the unified total. The
    image/audio sizes default to the module constants but can be overridden to
    match a VQ-VAE trained with a different codebook size (else the reserved range
    wouldn't match the indices the VQ-VAE actually emits)."""
    img_off = text_vocab_size
    aud_off = text_vocab_size + image_vocab_size
    total   = aud_off + audio_vocab_size
    return {
        "text":  {"offset": 0,       "size": text_vocab_size},
        "image": {"offset": img_off, "size": image_vocab_size},
        "audio": {"offset": aud_off, "size": audio_vocab_size},
        "total": total,
    }
