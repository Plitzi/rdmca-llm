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


def build_modality_layout(text_vocab_size: int) -> Dict:
    """Return offsets/sizes for each modality range and the unified total."""
    img_off = text_vocab_size
    aud_off = text_vocab_size + IMAGE_VOCAB_SIZE
    total   = aud_off + AUDIO_VOCAB_SIZE
    return {
        "text":  {"offset": 0,       "size": text_vocab_size},
        "image": {"offset": img_off, "size": IMAGE_VOCAB_SIZE},
        "audio": {"offset": aud_off, "size": AUDIO_VOCAB_SIZE},
        "total": total,
    }
