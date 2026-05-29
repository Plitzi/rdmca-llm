"""
Audio Tokenizer — Phase 4 (EnCodec)
Frozen EnCodec codec produces 50 tokens/sec.
Audio token IDs: 40192–44287 in the unified vocabulary.

Spec (Implementation Guide §4.1):
  Model:      EnCodec 24kHz pre-trained
  Bandwidth:  6.0 kbps → ~50 tokens/sec
  Vocab:      4,096 audio tokens (first codebook only)
  Offset:     40192 in unified vocab
"""
from __future__ import annotations
from typing import List, Optional

AUDIO_VOCAB_OFFSET = 40_192
AUDIO_VOCAB_SIZE   = 4_096
ENCODEC_SR         = 24_000
TOKENS_PER_SEC     = 50


class EnCodecTokenizer:
    """
    Wrapper around Meta's EnCodec (frozen, used as feature extractor).
    Install: pip install encodec
    """

    def __init__(self, bandwidth: float = 6.0):
        self.bandwidth = bandwidth
        self._model    = None
        self._load()

    def _load(self) -> None:
        try:
            from encodec import EncodecModel
            self._model = EncodecModel.encodec_model_24khz()
            self._model.set_target_bandwidth(self.bandwidth)
            # Freeze all parameters
            for p in self._model.parameters():
                p.requires_grad = False
        except ImportError:
            pass   # Phase 4 — optional until audio work begins

    @property
    def ready(self) -> bool:
        return self._model is not None

    def encode(self, waveform) -> List[int]:
        """
        waveform: torch.Tensor [1, T] at 24kHz
        Returns token IDs offset into unified audio vocab.
        Phase 4 only.
        """
        if not self.ready:
            raise RuntimeError(
                "EnCodec not available. Run: pip install encodec  (Phase 4)")
        import torch
        encoded = self._model.encode(waveform.unsqueeze(0))
        codes   = encoded[0][0]   # first codebook: [T_compressed]
        return [int(c) + AUDIO_VOCAB_OFFSET for c in codes[0].tolist()]

    def decode(self, token_ids: List[int]):
        """Reconstruct audio from token IDs."""
        if not self.ready:
            raise RuntimeError("EnCodec not available (Phase 4)")
        raise NotImplementedError
