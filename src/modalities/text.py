"""
Text Tokenizer Wrapper — SentencePiece BPE over a config-driven language set.

The set of languages is NOT hardcoded here: it is decided in the config
(`model.languages`), baked into the SentencePiece model at training time as
`<lang:XX>` user-defined symbols, and persisted to tokenizer_info.json. This
wrapper reads that metadata so the language-id prefix is always consistent with
the trained tokenizer.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

DEFAULT_VOCAB_SIZE = 65536


class TextTokenizer:
    """Thin wrapper around a trained SentencePiece model + tokenizer_info.json."""

    def __init__(self, model_path: str = "dist/tokenizer/rdmca_spm.model"):
        self.model_path = Path(model_path)
        self._sp = None
        self.lang_tokens: Dict[str, int] = {}
        self.text_vocab_size: int = DEFAULT_VOCAB_SIZE
        self._non_text_ids: set = set()

        if self.model_path.exists():
            import sentencepiece as spm
            self._sp = spm.SentencePieceProcessor()
            self._sp.Load(str(self.model_path))
            self.text_vocab_size = self._sp.GetPieceSize()
            self._load_info()

    def _load_info(self) -> None:
        """Pull language ids and modality specials from tokenizer_info.json."""
        info_path = self.model_path.parent / "tokenizer_info.json"
        if not info_path.exists():
            return
        try:
            info = json.loads(info_path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        self.lang_tokens = {k: int(v) for k, v in
                            (info.get("lang_token_ids") or {}).items()}
        # IDs stripped before decoding to readable text: language tags +
        # modality boundary tokens (user-defined symbols that would otherwise
        # decode to their literal "<...>" string).
        self._non_text_ids = set(self.lang_tokens.values())
        self._non_text_ids |= set(
            int(v) for v in (info.get("modality_tokens") or {}).values())

    @property
    def ready(self) -> bool:
        return self._sp is not None

    def encode(self, text: str, lang: str = "en",
               add_bos: bool = True, add_eos: bool = True) -> List[int]:
        if not self._sp:
            raise RuntimeError(
                f"Tokenizer not found at {self.model_path}. "
                "Run: python scripts/train_tokenizer.py")
        ids = self._sp.EncodeAsIds(text)
        prefix: List[int] = []
        if add_bos:
            prefix.append(BOS_ID)
        if lang in self.lang_tokens:
            prefix.append(self.lang_tokens[lang])
        if add_eos:
            ids = ids + [EOS_ID]
        return prefix + ids

    def encode_raw(self, text: str) -> List[int]:
        """Pieces ONLY — no BOS/EOS and no `<lang:XX>` prefix. Use when encoding a
        mid-sequence fragment (e.g. the `<think>` delimiters in the chat loop):
        encode() always injects the language token for a known language, which
        mid-sequence inserts a `<lang:en>` the model only ever saw at the start of
        a sequence — a spurious token that degrades continuation."""
        if not self._sp:
            raise RuntimeError("Tokenizer not loaded")
        return list(self._sp.EncodeAsIds(text))

    def decode(self, ids: List[int]) -> str:
        if not self._sp:
            raise RuntimeError("Tokenizer not loaded")
        # Drop control/specials, language tags, modality boundaries and any
        # non-text (image/audio) token that sits above the text vocab range.
        filtered = [i for i in ids
                    if i > EOS_ID
                    and i < self.text_vocab_size
                    and i not in self._non_text_ids]
        return self._sp.DecodeIds(filtered)

    def vocab_size(self) -> int:
        return self.text_vocab_size
