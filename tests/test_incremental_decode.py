"""
Tests for IncrementalDecoder (uses/chat/run_chat.py): the O(n) streaming decoder
must return EXACTLY the same text as a full re-decode at every prefix, across the
tricky cases (multi-char pieces, SentencePiece leading-space, byte-fallback splits),
and must actually re-decode less than the naive whole-sequence approach.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from uses.chat.run_chat import IncrementalDecoder


def _assert_matches(decode_fn, ids, keep=4):
    """Feeding ids one-by-one must equal decode_fn(ids[:k]) at every k."""
    inc = IncrementalDecoder(decode_fn, keep=keep)
    for k, t in enumerate(ids, 1):
        got = inc.append(t)
        want = decode_fn(ids[:k])
        assert got == want, f"at k={k}: {got!r} != {want!r}"


# ── 1. one char per token ────────────────────────────────────────────────────
def test_single_char_tokens():
    decode = lambda xs: "".join(chr(ord('a') + (i % 26)) for i in xs)
    _assert_matches(decode, list(range(50)))


# ── 2. variable-length multi-char pieces ─────────────────────────────────────
def test_multichar_pieces():
    vocab = {0: "hello", 1: " world", 2: "!", 3: "ABC", 4: "x", 5: "..."}
    decode = lambda xs: "".join(vocab[i] for i in xs)
    ids = [0, 1, 2, 3, 4, 5, 0, 4, 4, 1, 2, 3, 5, 0, 1] * 3
    _assert_matches(decode, ids)


# ── 3. SentencePiece-like leading-space + strip ──────────────────────────────
def test_sentencepiece_leading_space():
    # pieces use ▁ for a leading space; decode joins, maps ▁→space, strips the
    # single leading space (exactly what SentencePiece does).
    vocab = {0: "▁the", 1: "▁cat", 2: "s", 3: "▁sat", 4: ".", 5: "▁a", 6: "▁dog"}
    def decode(xs):
        return "".join(vocab[i] for i in xs).replace("▁", " ").lstrip()
    ids = [0, 1, 2, 3, 4, 5, 6, 3, 4, 0, 1, 3, 4, 5, 6] * 3
    _assert_matches(decode, ids, keep=3)


# ── 4. byte-fallback: one char split across two tokens ───────────────────────
def test_byte_fallback_split():
    # tokens 10/11 are two byte-halves of "é"; decoding them SEPARATELY yields a
    # replacement char, so the frozen-prefix cut must NOT happen across them. The
    # decoder must still return the correct full text (it re-decodes, never freezes
    # a bad seam).
    def decode(xs):
        out = bytearray()
        for i in xs:
            if i == 10: out += b"\xc3"
            elif i == 11: out += b"\xa9"          # 0xC3 0xA9 = 'é'
            else: out.append(ord('a') + (i % 26))
        return out.decode("utf-8", errors="replace")
    ids = [0, 1, 10, 11, 2, 3, 10, 11, 4, 5, 6, 10, 11, 7, 8] * 2
    _assert_matches(decode, ids, keep=2)


# ── 5. real tokenizer if available ───────────────────────────────────────────
def test_real_tokenizer_roundtrip():
    try:
        from src.modalities.text import TextTokenizer
        tok = TextTokenizer()
    except Exception:
        pytest.skip("tokenizer not available")
    if not getattr(tok, "ready", False):
        pytest.skip("tokenizer not trained")
    ids = tok.encode("The quick brown fox jumps over the lazy dog. "
                     "Hello, world! ¿Cómo estás?", lang="en", add_bos=False, add_eos=False)
    _assert_matches(tok.decode, ids, keep=4)


# ── 6. fewer decode calls than naive (the O(n) win) ──────────────────────────
def test_decodes_less_than_naive():
    calls = {"n": 0, "chars": 0}
    base = {i: f"tok{i}_" for i in range(30)}
    def decode(xs):
        calls["n"] += 1
        calls["chars"] += sum(len(base[i]) for i in xs)
        return "".join(base[i] for i in xs)
    ids = list(range(30))
    inc = IncrementalDecoder(decode, keep=4)
    for t in ids:
        inc.append(t)
    inc_chars = calls["chars"]
    # naive: re-decode the whole prefix each step → sum of prefix lengths
    naive_chars = sum(sum(len(base[i]) for i in ids[:k]) for k in range(1, len(ids) + 1))
    assert inc_chars < naive_chars, f"incremental {inc_chars} !< naive {naive_chars}"
