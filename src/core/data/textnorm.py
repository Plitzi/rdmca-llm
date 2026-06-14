"""
Ingestion-time text normalization + garbage filtering.

Every source (current and FUTURE, however messy) passes through here at write time
(scripts/prepare_data.write_jsonl), so the model trains on a CONSISTENT surface form
regardless of where the text came from. We normalize FORMAT noise (encoding, weird
whitespace, smart quotes, control chars) — NOT content — and drop only clearly broken
lines (mojibake, single-char/line spam, symbol soup). Diversity of CONTENT is kept;
the byte-fallback tokenizer handles any character this leaves behind.

Crucially, conversational STRUCTURE is preserved: newlines and the `Role:` line
markers (User:/Assistant:/…) that the loader splits on are kept intact — only
intra-line whitespace is collapsed.
"""
from __future__ import annotations

import re
import unicodedata

# Smart punctuation → ASCII. We map only PUNCTUATION, never letters, so accented
# multilingual text (é, ñ, ü, …) is untouched.
_PUNCT_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",   # ‘ ’ ‚ ‛
    "′": "'", "´": "'", "`": "'",                   # ′ ´ `
    "“": '"', "”": '"', "„": '"', "‟": '"',   # “ ” „ ‟
    "″": '"',                                                 # ″
    "–": "-", "—": "-", "―": "-", "−": "-",   # – — ― −
    "…": "...",                                              # …
    " ": " ",                                               # non-breaking space
}
_TRANS = str.maketrans(_PUNCT_MAP)

# Inline whitespace: regular space/tab + assorted Unicode spaces and zero-width
# marks — collapsed to a single space PER LINE. Newlines are handled separately so
# transcript structure survives.
_WS_INLINE = re.compile(
    "[ \t   -​  　﻿]+")
_MULTINEWLINE = re.compile(r"\n{3,}")
_ALNUM = re.compile(r"[^\W_]", re.UNICODE)        # Unicode letters + digits
_REPLACEMENT = "�"                           # the � mojibake marker


def normalize_text(text: str) -> str:
    """Return `text` with format noise normalized (NFKC, smart punctuation → ASCII,
    control chars stripped, inline whitespace collapsed) while PRESERVING newlines and
    role markers. Idempotent: normalizing twice equals normalizing once."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = t.translate(_TRANS)
    # Drop control / zero-width / format chars, but keep newline and tab.
    t = "".join(ch for ch in t
                if ch in ("\n", "\t") or unicodedata.category(ch)[0] != "C")
    # Collapse inline whitespace per line and trim; keep the newline skeleton.
    lines = [_WS_INLINE.sub(" ", ln).strip() for ln in t.split("\n")]
    t = "\n".join(lines)
    t = _MULTINEWLINE.sub("\n\n", t)              # ≤2 consecutive newlines
    return t.strip()


def is_garbage(text: str) -> bool:
    """True for clearly-broken text that should be dropped (not just down-weighted).
    Conservative on purpose — only the strongest signals fire, so legitimate diverse
    or multilingual content is never discarded:

      - mojibake: many U+FFFD replacement chars;
      - single-character spam (e.g. 'aaaaaaaaaa…' / '======');
      - symbol soup: almost no letters/digits among the non-space characters.
    """
    if not text:
        return True
    compact = text.replace("\n", "").replace("\t", "").replace(" ", "")
    n = len(compact)
    if n < 20:                                    # too short to judge — let it through
        return False
    if compact.count(_REPLACEMENT) / n > 0.02:    # mojibake
        return True
    # Most-common non-space character dominates → repeated-char spam.
    most = max(compact.count(c) for c in set(compact))
    if most / n > 0.40:
        return True
    # Hardly any letters/digits → symbol soup / separators / art.
    if len(_ALNUM.findall(compact)) / n < 0.20:
        return True
    return False


def clean_record_text(text: str) -> str:
    """Normalize, then return "" if the result is garbage (caller skips empties).
    The single call site is the JSONL write choke point (prepare_data.write_jsonl)."""
    t = normalize_text(text)
    return "" if is_garbage(t) else t


# ── conversational quality (content gate for dialogue / instruct sources) ──────
# Format/garbage filtering above is universal. This is a CONTENT gate applied only to
# turn-structured conversational sources: the #1-priority register for L1 is the model
# UNDERSTANDING and ANSWERING, so we keep clean, appropriately-short exchanges and drop
# the long technical/monologue/markdown-dump records (the ones that spiked instruct ppl
# to ~72) which a tiny base cannot learn to produce.
_ROLE_RE = re.compile(r"^(User|Assistant|System|Tools|Action|Observation):", re.M)
_RESPONSE_ROLES = {"Assistant", "Action"}
_CONTEXT_ROLES = {"User", "System"}
_URL_RE = re.compile(r"https?://|www\.")
_CODE_FENCE = "```"


def _role_turns(text: str):
    """Split a `Role: …` transcript into [(role, body), …]; [] if not structured."""
    marks = list(_ROLE_RE.finditer(text))
    if not marks or marks[0].start() != 0:
        return []
    turns = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        role = m.group(1)
        body = text[m.end():end].strip()
        turns.append((role, body))
    return turns


def conversational_quality_ok(text: str, max_turn_chars: int = 700,
                              max_total_chars: int = 1400, max_turns: int = 12) -> bool:
    """True if a conversational record is clean enough for a tiny L1 base to learn to
    ANSWER from. Requires a real exchange (≥1 context + ≥1 response turn), bounds turn
    and total length, and rejects code dumps / link-heavy / empty-turn records. Lenient
    by design — it trims the unlearnable tail, not the diversity. Non-structured text
    (prose) returns True (this gate is only wired to conversational sources)."""
    turns = _role_turns(text)
    if not turns:
        return True
    if len(turns) > max_turns or len(text) > max_total_chars:
        return False
    roles = {r for r, _ in turns}
    if not (roles & _RESPONSE_ROLES) or not (roles & _CONTEXT_ROLES):
        return False                              # not a User↔Assistant exchange
    if _CODE_FENCE in text or len(_URL_RE.findall(text)) > 2:
        return False                              # code/link dumps aren't L1 conversation
    for _role, body in turns:
        if not body or len(body) > max_turn_chars:
            return False                          # empty or unlearnably long turn
    return True
