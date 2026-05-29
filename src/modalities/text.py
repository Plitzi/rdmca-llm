"""
Text Tokenizer Wrapper — Phase 1
SentencePiece BPE tokenizer, vocab_size=32000.
Language ID tokens prepended to every sequence.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional

import numpy as np


# Active languages for this project: EN + ES
# IDs 4-10 are user-defined symbols in the SentencePiece vocab.
LANG_TOKENS = {
    "en": 4,
    "es": 5,
    "fr": 6,
    "de": 7,
    "zh": 8,
    "ja": 9,
    "ar": 10,
}

DEFAULT_VOCAB_SIZE = 65536   # multilingual (EN+ES)

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


class TextTokenizer:
    """
    Thin wrapper around a trained SentencePiece model.
    Model must be trained with train_tokenizer.py before use.
    """

    def __init__(self, model_path: str = "dist/tokenizer/rdmca_spm.model"):
        self.model_path = Path(model_path)
        self._sp = None
        if self.model_path.exists():
            import sentencepiece as spm
            self._sp = spm.SentencePieceProcessor()
            self._sp.Load(str(self.model_path))

    @property
    def ready(self) -> bool:
        return self._sp is not None

    def encode(self, text: str, lang: str = "en",
               add_bos: bool = True, add_eos: bool = True) -> List[int]:
        if not self._sp:
            raise RuntimeError(
                f"Tokenizer not found at {self.model_path}. "
                "Run: python train_tokenizer.py"
            )
        ids = self._sp.EncodeAsIds(text)
        prefix = [LANG_TOKENS.get(lang, UNK_ID)]
        if add_bos:
            prefix = [BOS_ID] + prefix
        if add_eos:
            ids = ids + [EOS_ID]
        return prefix + ids

    def decode(self, ids: List[int]) -> str:
        if not self._sp:
            raise RuntimeError("Tokenizer not loaded")
        # Filter special tokens before decoding
        filtered = [i for i in ids if i >= len(LANG_TOKENS) + 4]
        return self._sp.DecodeIds(filtered)

    def vocab_size(self) -> int:
        return self._sp.GetPieceSize() if self._sp else DEFAULT_VOCAB_SIZE
