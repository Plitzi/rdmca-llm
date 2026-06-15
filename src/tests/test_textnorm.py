"""
Tests for the ingestion normalization + garbage gate (src/data/textnorm.py):

  - format noise is normalized (NFKC, smart quotes/dashes → ASCII, control chars,
    inline-whitespace collapse) while CONTENT (incl. accents) is preserved;
  - conversational STRUCTURE survives — role markers + newlines stay intact, so the
    loader's turn splitter still works after normalization;
  - normalization is idempotent;
  - the garbage filter drops clearly-broken text (mojibake, single-char/symbol spam)
    but keeps legitimate diverse/multilingual/code/short text.
"""

from src.data.loader import _split_turns
from src.data.textnorm import (
    clean_record_text,
    conversational_quality_ok,
    is_garbage,
    normalize_text,
)


# ── normalization ─────────────────────────────────────────────────────────────
def test_smart_punctuation_mapped_to_ascii():
    assert normalize_text("“Hi” — it’s fine…") == '"Hi" - it\'s fine...'


def test_inline_whitespace_collapsed_and_trimmed():
    assert normalize_text("a\t  b ​c   ") == "a b c"


def test_accents_and_non_ascii_letters_preserved():
    s = "El zorro marrón rápido — café"
    assert "marrón" in normalize_text(s) and "café" in normalize_text(s)


def test_control_chars_stripped_but_newline_tab_semantics():
    out = normalize_text("a\x00\x07b\nc")
    assert out == "ab\nc"


def test_idempotent():
    s = "“Weird”   spacing—here\n\n\n\nand   tabs\t\t"
    once = normalize_text(s)
    assert normalize_text(once) == once


def test_transcript_structure_preserved_for_loader():
    """Role markers + newlines must survive so _split_turns still segments the turns
    (completion-only masking depends on it)."""
    raw = "User:   what is   2+2?\n\n\n\nAssistant:  it’s   4"
    norm = normalize_text(raw)
    turns = _split_turns(norm)
    assert turns is not None
    roles = [r for r, _ in turns]
    assert roles == ["User", "Assistant"]
    assert "it's 4" in turns[1][1]


# ── garbage filter ─────────────────────────────────────────────────────────────
def test_garbage_detects_mojibake_and_spam():
    assert is_garbage("real words here " + "�" * 30)  # mojibake dominates
    assert is_garbage("a" * 40)  # single-char spam
    assert is_garbage("=" * 30 + " >>>>>>>>>>>>>>>")  # symbol soup


def test_garbage_keeps_legitimate_text():
    assert not is_garbage("The quick brown fox jumps over the lazy dog today.")
    assert not is_garbage("El zorro marrón rápido salta sobre el perro perezoso.")
    assert not is_garbage("def add(a, b): return a + b  # sum two numbers")
    assert not is_garbage("ok")  # too short to judge → kept


def test_clean_record_text_normalizes_then_gates():
    assert clean_record_text("“Hello”   world") == '"Hello" world'
    assert clean_record_text("�" * 50) == ""  # garbage → dropped
    assert clean_record_text("") == ""


# ── conversational quality gate ────────────────────────────────────────────────
def test_conversational_quality_keeps_clean_exchange():
    assert conversational_quality_ok("User: Hi there!\nAssistant: Hello! How can I help?")
    assert conversational_quality_ok("System: Be kind.\nUser: hola\nAssistant: ¡Hola!")


def test_conversational_quality_requires_real_exchange():
    assert not conversational_quality_ok("User: hi\nUser: anyone?")  # no response role
    assert not conversational_quality_ok("Assistant: hello\nAssistant: hi")  # no context role


def test_conversational_quality_rejects_unlearnable():
    assert not conversational_quality_ok("User: code?\nAssistant: ```py\nprint(1)\n```")
    assert not conversational_quality_ok("User: hi\nAssistant: " + "word " * 200)  # too long
    assert not conversational_quality_ok("User: \nAssistant: hi")  # empty turn
    assert not conversational_quality_ok(
        "User: links\nAssistant: http://a.com http://b.com http://c.com"
    )  # link dump


def test_conversational_quality_passes_prose_through():
    # Non-structured text isn't this gate's job (only wired to conversational sources).
    assert conversational_quality_ok("Once upon a time there was a small blue fox.")
