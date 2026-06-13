"""
Interactive-session ergonomics shared by the use cases (chat, agent), modelled on
Claude Code:

  - InterruptGuard — press Ctrl-C DURING a model response to ABORT just that
    response (keep the session alive), instead of killing the program. Wrap a
    generation in the guard and pass `guard.stopped` as the loop's stop check.

  - SessionInput — a background stdin reader so the user can TYPE WHILE the model
    is generating: those lines QUEUE and are processed on the next turn (so you can
    steer/correct an agent that is heading the wrong way). One thread owns stdin;
    the REPL pulls turns via `next_message()` instead of bare `input()`. Piped
    (non-interactive) stdin still works: lines drain in order, then EOF.

Both are dependency-free and POSIX/macOS friendly (no raw-mode / curses).
"""
from __future__ import annotations

import queue
import signal
import sys
import threading
from typing import List, Optional


class InterruptGuard:
    """Context manager: Ctrl-C inside the `with` aborts the RESPONSE, not the app.

        with InterruptGuard() as g:
            generate(..., should_stop=g.stopped)
        if g.was_interrupted:
            print("[interrupted]")

    Restores the previous SIGINT handler on exit, so Ctrl-C at the prompt keeps its
    normal meaning (raise KeyboardInterrupt → leave the session). Must be entered on
    the main thread (signal handlers can only be installed there)."""

    def __init__(self) -> None:
        self._stop = False
        self._prev = None

    def stopped(self) -> bool:
        return self._stop

    @property
    def was_interrupted(self) -> bool:
        return self._stop

    def _on_sigint(self, *_a) -> None:
        self._stop = True

    def __enter__(self) -> "InterruptGuard":
        self._stop = False
        try:
            self._prev = signal.signal(signal.SIGINT, self._on_sigint)
        except (ValueError, OSError):
            self._prev = None          # not on the main thread → no abort, still safe
        return self

    def __exit__(self, *_a) -> bool:
        if self._prev is not None:
            try:
                signal.signal(signal.SIGINT, self._prev)
            except (ValueError, OSError):
                pass
        return False                   # never swallow exceptions


class SessionInput:
    """Background line-reader: lines typed any time (incl. WHILE generating) are
    queued; the REPL consumes them with `next_message()`. Daemon thread, so it never
    blocks process exit."""

    def __init__(self) -> None:
        self._q: "queue.Queue[str]" = queue.Queue()
        self._eof = threading.Event()  # set once stdin closes (Ctrl-D / pipe end)
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        # readline() (not `for line in stdin`) so a typed line is delivered as soon
        # as Enter is pressed, not buffered behind a block read.
        while True:
            line = sys.stdin.readline()
            if line == "":             # EOF (Ctrl-D / piped input exhausted)
                self._eof.set()
                return
            self._q.put(line.rstrip("\n"))

    def pending(self) -> int:
        """How many messages were typed ahead (queued during generation)."""
        return self._q.qsize()

    def next_message(self, prompt: str = "") -> Optional[str]:
        """Block for the next line and return it (already-queued lines come back
        immediately, in order). Returns None at EOF. Polls so Ctrl-C at the prompt
        still raises KeyboardInterrupt (leave the session)."""
        if prompt:
            sys.stdout.write(prompt)
            sys.stdout.flush()
        while True:
            try:
                return self._q.get(timeout=0.2)
            except queue.Empty:
                if self._eof.is_set() and self._q.empty():
                    return None        # drained and stdin closed
                continue

    def drain_pending(self) -> List[str]:
        """Return (and remove) all lines already queued — e.g. typed while the model
        was generating — in order."""
        out: List[str] = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                break
        return out
