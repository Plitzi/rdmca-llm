"""
Multimodal Perception Layer (MPL) — RDMCA §7.4 / §12.4

The system's entry point for raw stimuli. Detects modality, tokenizes each
segment with its tokenizer, and assembles a single interleaved token sequence in
the unified vocabulary:

    <mod:image> <img tokens…> <mod_end> <mod:text> <lang:xx> <text tokens…>

Image/audio token indices are shifted by their range offset (from
tokenizer_info.json) so they never collide with text ids. The foundational model
then consumes one flat token stream — the same next-token objective for every
modality (Era 3b).
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.config import load_tokenizer_info
from src.modalities.text import TextTokenizer
from src.modalities.image import ImageVQVAE
from src.modalities.audio import AudioVQVAE

IMAGE_VQVAE_PATH = "dist/tokenizer/image_vqvae.npz"
AUDIO_VQVAE_PATH = "dist/tokenizer/audio_vqvae.npz"


class MultimodalPerception:
    def __init__(self,
                 text_tok: Optional[TextTokenizer] = None,
                 image_tok: Optional[ImageVQVAE] = None,
                 audio_tok: Optional[AudioVQVAE] = None,
                 info: Optional[dict] = None):
        self.text = text_tok or TextTokenizer()
        self.image = image_tok
        self.audio = audio_tok
        self.info = info or load_tokenizer_info() or {}
        layout = self.info.get("modality_layout", {})
        self.img_offset = layout.get("image", {}).get("offset", 0)
        self.aud_offset = layout.get("audio", {}).get("offset", 0)
        mt = self.info.get("modality_tokens", {})
        self.tok_mod_text  = mt.get("mod_text")
        self.tok_mod_image = mt.get("mod_image")
        self.tok_mod_audio = mt.get("mod_audio")
        self.tok_mod_end   = mt.get("mod_end")

    # -- lazy tokenizer loading -----------------------------------------
    def _image(self) -> ImageVQVAE:
        if self.image is None:
            self.image = ImageVQVAE.load(IMAGE_VQVAE_PATH)
        if self.image is None:
            raise RuntimeError(
                "Image tokenizer not trained. Run: "
                "python scripts/train_image_tokenizer.py")
        return self.image

    def _audio(self) -> AudioVQVAE:
        if self.audio is None:
            self.audio = AudioVQVAE.load(AUDIO_VQVAE_PATH)
        if self.audio is None:
            raise RuntimeError(
                "Audio tokenizer not trained. Run: "
                "python scripts/train_audio_tokenizer.py")
        return self.audio

    def _wrap(self, start_tok: Optional[int], body: List[int]) -> List[int]:
        seq = []
        if start_tok is not None:
            seq.append(start_tok)
        seq.extend(body)
        if self.tok_mod_end is not None:
            seq.append(self.tok_mod_end)
        return seq

    # -- per-modality encoders ------------------------------------------
    def encode_text(self, text: str, lang: str = "en",
                    boundary: bool = False) -> List[int]:
        ids = self.text.encode(text, lang=lang, add_bos=not boundary, add_eos=False)
        if boundary:
            return self._wrap(self.tok_mod_text, ids)
        return ids

    def encode_image(self, image) -> List[int]:
        raw = self._image().encode_ids(image)
        return self._wrap(self.tok_mod_image, [self.img_offset + i for i in raw])

    def encode_audio(self, wav) -> List[int]:
        raw = self._audio().encode_ids(wav)
        return self._wrap(self.tok_mod_audio, [self.aud_offset + i for i in raw])

    # -- sequence assembly ----------------------------------------------
    def build_sequence(self, segments: List[Tuple]) -> List[int]:
        """
        segments: list of tuples
          ("text", text, lang)   ("image", image_or_path)   ("audio", wav_or_path)
        Returns one interleaved unified-vocab token list.
        """
        out: List[int] = []
        for seg in segments:
            kind = seg[0]
            if kind == "text":
                text = seg[1]
                lang = seg[2] if len(seg) > 2 else "en"
                out += self.encode_text(text, lang=lang, boundary=True)
            elif kind == "image":
                out += self.encode_image(load_image(seg[1]))
            elif kind == "audio":
                out += self.encode_audio(load_audio(seg[1]))
            else:
                raise ValueError(f"unknown modality segment: {kind}")
        return out


# ---------------------------------------------------------------------------
# File loaders (kept dependency-light)
# ---------------------------------------------------------------------------

def load_image(src):
    """Path/PIL/np → np [H,W,3]. Returns src unchanged if already an array."""
    if isinstance(src, np.ndarray):
        return src
    if isinstance(src, (str, Path)):
        from PIL import Image
        return np.asarray(Image.open(src).convert("RGB"))
    return np.asarray(src)


def load_audio(src, sr: int = 16_000):
    """Path/np → mono waveform np[T]. Returns src unchanged if already an array."""
    if isinstance(src, np.ndarray):
        return src
    if isinstance(src, (str, Path)):
        try:
            import soundfile as sf
            wav, _ = sf.read(str(src))
            return wav.mean(axis=1) if wav.ndim > 1 else wav
        except ImportError:
            import wave
            with wave.open(str(src), "rb") as w:
                frames = w.readframes(w.getnframes())
            return (np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                    / 32768.0)
    return np.asarray(src)
