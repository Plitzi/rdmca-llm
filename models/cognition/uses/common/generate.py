"""
Token generation core — backend-agnostic sampling + the KV-cached decode loop +
the two-phase <think>/answer generation. Extracted from models/cognition/uses/chat/run_chat.py so the
chat and agent runtimes (and tests) share one implementation.
"""

from __future__ import annotations

import sys
import time

import numpy as np

import src.backend as backend
from models.cognition.uses.common import agent

# Anti-logic-bomb generation guards. Generation is already bounded by
# `max_new_tokens`, but an adversarial prompt can still drive a tiny model into a
# degenerate loop that burns the whole budget (and, with thinking on, stalls the
# turn). These detect that and stop early:
_MAX_TOKEN_REPEAT = 32  # same token N× in a row → degenerate
_CYCLE_MAX_LEN = 8  # look for a repeating cycle up to this length …
_CYCLE_MIN_REPS = 5  # … repeated at least this many times → looping
GEN_DEADLINE_S = 90.0  # default per-generation wall-clock cap (0 = unlimited)


def _looping(generated: list) -> bool:
    """True if the tail of `generated` is a short cycle repeated many times — the
    signature of a stuck 'thinking' loop. O(_CYCLE_MAX_LEN) per call."""
    n = len(generated)
    for c in range(1, _CYCLE_MAX_LEN + 1):
        span = c * _CYCLE_MIN_REPS
        if n < span:
            break
        tail = generated[-span:]
        if all(tail[i] == tail[i % c] for i in range(span)):
            return True
    return False


def sample_top_p(
    logits,
    temperature: float,
    top_p: float,
    top_k: int = 0,
    recent_ids=None,
    rep_penalty: float = 1.0,
) -> int:
    logits_np = np.asarray(backend.current().ops.to_numpy(logits), dtype=np.float32).copy()
    # Repetition penalty (HF-style): push down logits of recently emitted tokens
    # so the model stops looping ("I'm sorry. I'm sorry. …"). Window-limited so it
    # never blocks tokens that legitimately recur over a longer span.
    if rep_penalty and rep_penalty != 1.0 and recent_ids:
        idx = np.fromiter({int(i) for i in recent_ids}, dtype=np.int64)
        if idx.size:
            vals = logits_np[idx]
            logits_np[idx] = np.where(vals > 0, vals / rep_penalty, vals * rep_penalty)
    if temperature == 0.0:
        return int(np.argmax(logits_np))
    logits_np = logits_np / temperature
    logits_np -= logits_np.max()
    probs = np.exp(logits_np)
    probs /= probs.sum()
    sorted_idx = np.argsort(probs)[::-1]
    if top_k and top_k > 0:  # restrict to the top_k most likely
        sorted_idx = sorted_idx[:top_k]
    sorted_prob = probs[sorted_idx]
    cumulative = np.cumsum(sorted_prob)
    cutoff = int(np.searchsorted(cumulative, top_p)) + 1
    top_idx = sorted_idx[:cutoff]
    top_prob = probs[top_idx]
    top_prob /= top_prob.sum()
    return int(np.random.choice(top_idx, p=top_prob))


class IncrementalDecoder:
    """O(n) streaming decode. A naive streamer re-decodes the WHOLE token list every
    step (`decode_fn(generated)`), which is O(n²) total work over a generation. This
    keeps a frozen text PREFIX and only re-decodes a short live TAIL each step, so
    total work is O(n).

    It returns EXACTLY the text of a full decode (asserted token-by-token in
    tests) — never a naive `+= decode([tok])`, which corrupts subword/byte-fallback
    merges and SentencePiece's leading-space handling. The prefix is frozen only at a
    boundary VERIFIED to reconstruct (`full.endswith(tail_redecode)`); since tokens
    are only appended at the end, a once-clean left boundary stays clean, and the
    whole live tail is re-decoded every step so any local merge is always inside the
    tail, never across the frozen seam. If a split is never clean it simply keeps
    re-decoding (still correct, just less amortized)."""

    def __init__(self, decode_fn, keep: int = 16):
        self._decode = decode_fn
        self._keep = keep  # tokens kept in the live tail
        self._toks: list[int] = []
        self._anchor = 0  # tail re-decode starts here
        self._prefix = ""  # verified prefix of decode(all toks)

    def append(self, tok_id: int) -> str:
        """Add one token; return the full decoded text so far (== decode(all))."""
        self._toks.append(tok_id)
        full = self._prefix + self._decode(self._toks[self._anchor :])
        # Freeze more of the prefix once the live tail is long, but only if the cut
        # reconstructs exactly — so the frozen text is always a true prefix.
        if len(self._toks) - self._anchor > 2 * self._keep:
            new_anchor = len(self._toks) - self._keep
            new_tail = self._decode(self._toks[new_anchor:])
            if full.endswith(new_tail):
                self._prefix = full[: len(full) - len(new_tail)]
                self._anchor = new_anchor
        return full


def generate(
    model,
    input_ids: list,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    vocab_size: int,
    context_len: int = 2048,
    stream: bool = True,
    decode_fn=None,
    max_seconds: float | None = None,
    stop_strings: tuple[str, ...] | None = None,
    top_k: int = 0,
    rep_penalty: float = 1.0,
    rep_window: int = 128,
    suppress_think: bool = False,
    should_stop=None,
) -> tuple[list[int], float]:
    """
    Returns (generated_ids, tokens_per_second).

    If stream=True, prints tokens as they are generated. When `decode_fn` is
    given (e.g. tokenizer.decode), it decodes the running output and prints the
    new text delta each step — real token streaming; otherwise it falls back to
    a per-token ▌ marker (used on the no-tokenizer plumbing path).

    Anti-logic-bomb guards (besides the `max_new_tokens` cap): generation also
    stops on a degenerate token loop (`_looping`) and on an optional wall-clock
    deadline (`max_seconds`), so a crafted prompt can't wedge the turn.
    """
    ops = backend.current().ops
    engine = backend.current().engine

    # Prefill the prompt ONCE into a per-layer KV cache, then decode one token at a
    # time feeding only the new token. This is the O(n³)→O(n²) win: each step does
    # O(1) new-token work instead of reprocessing the whole sequence. `eval` on the
    # cache tensors each step keeps MLX's lazy graph from growing unbounded.
    ids = list(input_ids)
    if len(ids) > context_len:  # keep the prompt within the positional limit
        ids = ids[-context_len:]
    prefill = ops.array(np.asarray([ids], dtype=np.int64))
    logits, caches = model.logits_cached(prefill, caches=None, pos_offset=0)
    next_logits = logits[0, -1, :]  # [vocab] — logits for the first new token
    engine.eval(next_logits, *[t for kv in caches for t in kv])
    cur_len = len(ids)

    generated: list[int] = []
    printed = ""
    repeat_run = 0
    boundary_hit = False  # broke at a turn-boundary leak → don't flush past it
    # O(n) streaming decode: one short tail re-decode per token instead of
    # re-decoding the whole sequence each step. Identical text, no O(n²) cost.
    inc = IncrementalDecoder(decode_fn) if decode_fn is not None else None
    t0 = time.perf_counter()

    EOS_ID = 3

    for _ in range(max_new_tokens):
        # User abort (Ctrl-C via InterruptGuard): stop NOW, return what we have so
        # far so the partial answer is kept and the session continues.
        if should_stop is not None and should_stop():
            break

        next_id = sample_top_p(
            next_logits,
            temperature,
            top_p,
            top_k=top_k,
            recent_ids=generated[-rep_window:] if rep_penalty != 1.0 else None,
            rep_penalty=rep_penalty,
        )

        if next_id == EOS_ID:
            break

        # Anti-loop: a single token repeated many times, or a short repeating
        # cycle, is a stuck generation — stop rather than burn the budget.
        repeat_run = repeat_run + 1 if (generated and next_id == generated[-1]) else 1
        if repeat_run >= _MAX_TOKEN_REPEAT:
            break

        generated.append(next_id)
        # Decode incrementally ONCE per token (O(n) total); reused by both the
        # stop-string check and the streamer below.
        text = inc.append(next_id) if inc is not None else None
        # What to DISPLAY/scan: with thinking hidden (think off), show only the
        # answer after `</think>` — never the scratchpad the model emits on its own.
        disp = agent.visible_stream_text(text) if (suppress_think and text is not None) else text

        if _looping(generated):
            break
        if max_seconds is not None and (time.perf_counter() - t0) > max_seconds:
            break

        # Stop-string check (e.g. role-tag turn-boundary leakage). Needs the
        # decoded text, so it runs on the tokenizer path; we print only up to the
        # boundary (in stream mode) and stop before emitting the leaked new turn.
        if stop_strings and disp is not None:
            cut = agent.first_stop_index(disp, stop_strings)
            if cut is not None:
                if stream:
                    sys.stdout.write(disp[len(printed) : cut])
                    sys.stdout.flush()
                    printed = disp[:cut]
                boundary_hit = True
                break

        if stream:
            if disp is not None:
                # Emit only up to the safe boundary — hold back a trailing fragment
                # that could still grow into a role tag (e.g. 'User' → 'User:'), so a
                # forming turn boundary is never half-printed.
                safe = agent.safe_stream_len(disp)
                if safe > len(printed):
                    sys.stdout.write(disp[len(printed) : safe])
                    sys.stdout.flush()
                    printed = disp[:safe]
            else:
                print("▌", end="", flush=True)

        # Advance the cache with the chosen token → logits for the next step. Stop
        # if the positional window is full (the chat trims history, so this is rare).
        if cur_len >= context_len:
            break
        new_tok = ops.array(np.asarray([[next_id]], dtype=np.int64))
        logits, caches = model.logits_cached(new_tok, caches=caches, pos_offset=cur_len)
        next_logits = logits[0, -1, :]
        engine.eval(next_logits, *[t for kv in caches for t in kv])
        cur_len += 1

    # Flush any held-back tail (real text that never became a role tag). Skipped
    # when we stopped at a boundary — everything past it is the leaked next turn.
    if stream and decode_fn is not None and not boundary_hit:
        text = decode_fn(generated)
        disp = agent.visible_stream_text(text) if suppress_think else text
        if len(disp) > len(printed):
            sys.stdout.write(disp[len(printed) :])
            sys.stdout.flush()

    elapsed = time.perf_counter() - t0
    tps = len(generated) / elapsed if elapsed > 0 else 0.0
    return generated, tps


def generate_thinking(
    model,
    prompt_ids: list,
    *,
    tokenizer,
    lang: str,
    max_new_tokens: int,
    think_budget: int,
    temperature: float,
    top_p: float,
    vocab_size: int,
    context_len: int,
    stream: bool = False,
    think_prefix: str = "",
    answer_prefix: str = "",
    max_seconds: float | None = None,
    answer_stop: tuple[str, ...] | None = agent.ANSWER_STOP_STRINGS,
    top_k: int = 0,
    rep_penalty: float = 1.0,
    should_stop=None,
) -> tuple[str, list[int], float]:
    """Two-phase generation: a budget-capped <think> scratchpad, then the answer.

    Returns (think_text, answer_ids, tok_per_s). With think_budget <= 0 it is a
    single plain generation and think_text is "". The scratchpad is force-closed
    when the budget is hit (or trimmed at </think> if the model closes early), so
    the answer is always generated from a well-formed `… </think>` prefix —
    mirroring how Claude bounds extended thinking with a token budget.

    When `stream=True` the scratchpad and answer are printed live (decoded
    incrementally), each preceded by its prefix label; callers then skip their
    own printing of the same content.
    """
    decode_fn = tokenizer.decode
    sgen = {
        "temperature": temperature,
        "top_p": top_p,
        "vocab_size": vocab_size,
        "context_len": context_len,
        "stream": stream,
        "decode_fn": (decode_fn if stream else None),
        "max_seconds": max_seconds,
        "top_k": top_k,
        "rep_penalty": rep_penalty,
        "should_stop": should_stop,
    }

    def _label(prefix):
        if stream and prefix:
            sys.stdout.write(prefix)
            sys.stdout.flush()

    if think_budget <= 0:
        # think OFF: a reasoning-trained model still emits <think> on its own, so
        # hide the scratchpad in the stream and show only the answer (suppress_think).
        _label(answer_prefix)
        ids, tps = generate(
            model,
            list(prompt_ids),
            max_new_tokens=max_new_tokens,
            stop_strings=answer_stop,
            suppress_think=True,
            **sgen,
        )
        # Fallback: if the model OPENED a <think> it never CLOSED, the visible answer
        # is empty (a blank reply — the symptom on an under-trained reasoning stage).
        # Force-close the scratchpad and continue from it (Phase-B style) so 'think
        # off' NEVER returns blank — without ever showing the raw scratchpad.
        raw = decode_fn(ids) if ids else ""
        if ids and agent.THINK_OPEN in raw and not agent.visible_stream_text(raw).strip():
            scratch = raw.split(agent.THINK_OPEN, 1)[1].split(agent.THINK_CLOSE, 1)[0].strip()
            closed = f"{agent.THINK_OPEN} {scratch} {agent.THINK_CLOSE}\n"
            _label(answer_prefix)
            ids, tps2 = generate(
                model,
                list(prompt_ids) + tokenizer.encode_raw(closed),
                max_new_tokens=max_new_tokens,
                stop_strings=answer_stop,
                **sgen,
            )
            tps = tps2 or tps
        return "", ids, tps

    # Raw pieces only — NO `<lang:XX>` prefix. encode() would inject the language
    # token mid-sequence (the model only saw it at the start), degrading the
    # scratchpad/answer continuation. See TextTokenizer.encode_raw.
    def enc(s):
        return tokenizer.encode_raw(s)

    # Phase A — scratchpad. Prime with the opening tag so generation starts inside it.
    # Stop if the model runs into a new turn (a 'User:'/… leak) — reasoning should
    # never cross a turn boundary, same as the answer phase.
    _label(think_prefix)
    think_ids, tps_a = generate(
        model,
        list(prompt_ids) + enc(agent.THINK_OPEN),
        max_new_tokens=think_budget,
        stop_strings=answer_stop,
        **sgen,
    )
    think_text = tokenizer.decode(think_ids) if think_ids else ""
    if agent.THINK_CLOSE in think_text:  # model closed early
        think_text = think_text.split(agent.THINK_CLOSE)[0]
    think_text = think_text.strip()

    # Phase B — answer, from a force-closed scratchpad prefix.
    closed = f"{agent.THINK_OPEN} {think_text} {agent.THINK_CLOSE}\n"
    _label(answer_prefix)
    answer_ids, tps_b = generate(
        model,
        list(prompt_ids) + enc(closed),
        max_new_tokens=max_new_tokens,
        stop_strings=answer_stop,
        **sgen,
    )
    tps = next((t for t in (tps_b, tps_a) if t), 0.0)  # report a meaningful rate
    return think_text, answer_ids, tps
