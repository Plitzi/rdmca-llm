"""
Image Tokenizer — Phase 3 (VQVAE)
Patch-based VQVAE producing discrete visual tokens.
Token IDs occupy range 32000–40191 in the unified vocabulary.

Spec (Implementation Guide §3.1):
  Patch size:     16×16 pixels
  Visual vocab:   8,192 tokens (codebook size)
  Input:          224×224 → 14×14 = 196 tokens per image
  Encoder output: 512-dim → projected to d_model
"""
from __future__ import annotations
from typing import List, Optional

IMAGE_VOCAB_OFFSET = 32_000
IMAGE_VOCAB_SIZE   = 8_192
PATCH_SIZE         = 16
INPUT_RESOLUTION   = 224
N_PATCHES          = (INPUT_RESOLUTION // PATCH_SIZE) ** 2   # 196


class VQVAETokenizer:
    """
    VQVAE image tokenizer.
    Phase 3 implementation — not active until Phase 3 kickoff.
    """

    def __init__(self, checkpoint: Optional[str] = None):
        self.checkpoint = checkpoint
        self._model     = None
        # TODO Phase 3: load VQVAE checkpoint (train from scratch or
        # fine-tune from DALL-E dVAE / MobileVIT-based checkpoint)

    @property
    def ready(self) -> bool:
        return self._model is not None

    def encode(self, image) -> List[int]:
        """
        image: PIL.Image or np.ndarray [H, W, 3]
        Returns list of N_PATCHES token IDs offset into unified vocab.
        """
        if not self.ready:
            raise RuntimeError("VQVAETokenizer not initialized (Phase 3)")
        # TODO: preprocess → patch → codebook lookup → offset IDs
        raise NotImplementedError

    def decode(self, token_ids: List[int]):
        """Reconstruct image from visual token IDs."""
        if not self.ready:
            raise RuntimeError("VQVAETokenizer not initialized (Phase 3)")
        raise NotImplementedError
