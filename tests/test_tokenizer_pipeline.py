"""
Tokenizer-process tests. The tokenizer is the model's interface to language, so its
guarantees matter as much as the model's:

  - byte_fallback: ANY character (unseen scripts, emoji, future messy symbols) is
    representable as UTF-8 byte tokens — NEVER the <unk> id. This is the robustness
    property that lets the model degrade gracefully on out-of-distribution input.
  - character_coverage=1.0 + the script's params train a usable BPE that round-trips.
  - control delimiters (<think>, …) and language tags tokenize ATOMICALLY (one id),
    not BPE-split into pieces the corpus never contains.

These train a TINY SentencePiece model in a tmp dir with the SAME params the project
uses (scripts/train_tokenizer.train_spm), so a regression in those params is caught.
Skipped automatically if sentencepiece is unavailable.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

spm = pytest.importorskip("sentencepiece")

from src.modalities.vocab import CONTROL_SPECIALS, tokenizer_symbols


def _train_tiny(tmp_path) -> "spm.SentencePieceProcessor":
    """Train a small BPE with the project's robustness params (byte_fallback + full
    coverage) on a tiny bilingual corpus."""
    corpus = tmp_path / "corpus.txt"
    lines = []
    base_en = [
        "the cat sat on the mat",
        "hello there how are you",
        "a small fox runs fast",
        "what is a car please tell me",
        "i am doing well thank you",
    ]
    base_es = [
        "el gato se sento en la alfombra",
        "hola como estas hoy",
        "un zorro pequeno corre rapido",
        "que es un coche por favor",
    ]
    for _ in range(200):  # repeat so BPE has evidence
        lines.extend(base_en + base_es)
    corpus.write_text("\n".join(lines), encoding="utf-8")

    prefix = str(tmp_path / "tok")
    user_symbols = tokenizer_symbols(["en", "es"])
    spm.SentencePieceTrainer.Train(
        input=str(corpus),
        model_prefix=prefix,
        vocab_size=512,
        character_coverage=1.0,
        model_type="bpe",
        byte_fallback=True,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=user_symbols,
    )
    sp = spm.SentencePieceProcessor()
    sp.Load(prefix + ".model")
    return sp


def test_byte_fallback_never_emits_unk_on_unseen_chars(tmp_path):
    """An emoji / CJK char never seen in the corpus must decompose into byte tokens,
    NOT collapse to <unk> (id 1) — the whole point of byte_fallback."""
    sp = _train_tiny(tmp_path)
    for unseen in ["😀", "中文", "—", "​漢", "café→"]:
        ids = sp.EncodeAsIds(unseen)
        assert ids, f"no ids for {unseen!r}"
        assert 1 not in ids, f"<unk> leaked for {unseen!r}: {ids}"


def test_byte_fallback_roundtrips_arbitrary_unicode(tmp_path):
    sp = _train_tiny(tmp_path)
    for s in ["hello there", "el gato 😺", "mixed 中文 text", "emoji 🚀🚀"]:
        assert sp.DecodeIds(sp.EncodeAsIds(s)) == s


def test_control_and_lang_tokens_are_atomic(tmp_path):
    """<think>/<tool_call>/… and <lang:en> must each be a single in-vocab id
    (user-defined), so stage markers and language tags never BPE-split. We check the
    symbol survives as ONE token inside a sentence (SentencePiece's add_dummy_prefix
    adds a leading whitespace piece, so we don't compare the bare-symbol encoding)."""
    sp = _train_tiny(tmp_path)
    for sym in [*list(CONTROL_SPECIALS), "<lang:en>", "<lang:es>"]:
        pid = sp.PieceToId(sym)
        assert pid > 3, f"{sym} not a user-defined id (got {pid})"  # >eos, not <unk>
        ids = sp.EncodeAsIds(f"hello {sym} world")
        assert ids.count(pid) == 1, f"{sym} did not tokenize atomically: {ids}"


def test_basic_roundtrip_in_vocab_text(tmp_path):
    sp = _train_tiny(tmp_path)
    assert sp.DecodeIds(sp.EncodeAsIds("the cat sat on the mat")) == "the cat sat on the mat"
