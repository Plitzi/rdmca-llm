"""
Training Dashboard — rich terminal UI for RDMCA stage training.
Displays a live-updating panel with loss, speed, ETA, memory and gate status.
"""
from __future__ import annotations
import sys
import threading
import time
from collections import deque
from typing import Optional

import mlx.core as mx

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress,
    SpinnerColumn, TaskProgressColumn, TextColumn, TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text
from rich import box

# Unicode sparkline chars (low → high)
_SPARKS = " ▁▂▃▄▅▆▇█"

STAGE_GATES = {
    1: ("BLiMP grammaticality", 0.70),
    2: ("ARC Easy accuracy",    0.60),
    3: ("GSM8K accuracy",       0.15),
    4: ("Causal reasoning",     0.65),
    5: ("BCF probe set",        0.90),
}

STAGE_NAMES = {
    1: "Language and communication",
    2: "Perception and pattern recognition",
    3: "Abstraction and symbolic composition",
    4: "Causal and procedural reasoning",
    5: "Cognitive ethics and BCF",
}


def _sparkline(values: list[float], width: int = 12) -> str:
    if not values:
        return "─" * width
    lo, hi = min(values), max(values)
    rng = hi - lo or 1.0
    chars = [_SPARKS[int((v - lo) / rng * (len(_SPARKS) - 1))] for v in values[-width:]]
    return "".join(chars)


def _mem_str() -> str:
    """Active GPU memory via MLX Metal API (returns '─' if unavailable)."""
    try:
        active = mx.get_active_memory() / 1e9
        peak   = mx.get_peak_memory()   / 1e9
        return f"{active:.1f} GB active  /  {peak:.1f} GB peak"
    except Exception:
        return "─"


def _arrow(current: float, previous: float) -> str:
    if previous is None or abs(current - previous) < 0.001:
        return "─"
    return "[green]↓[/green]" if current < previous else "[red]↑[/red]"


_SPIN_CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class CompileSpinner:
    """
    Simple stdout spinner that runs in its own thread.
    Use before the Live dashboard so it works even when mx.eval()
    holds the GIL during Metal JIT compilation.

    Usage:
        with CompileSpinner("Compiling…"):
            mx.eval(...)   # blocks for 1-3 min on first step
    """

    def __init__(self, msg: str = "Compiling computation graph…"):
        self._msg   = msg
        self._stop  = threading.Event()
        self._thread: threading.Thread | None = None

    def _spin(self) -> None:
        t0 = time.time()
        i  = 0
        while not self._stop.is_set():
            elapsed = _fmt_time(time.time() - t0)
            c = _SPIN_CHARS[i % len(_SPIN_CHARS)]
            sys.stdout.write(f"\r  {c}  {self._msg}  {elapsed}  ")
            sys.stdout.flush()
            i += 1
            self._stop.wait(timeout=0.1)
        sys.stdout.write("\r" + " " * 70 + "\r")
        sys.stdout.flush()

    def __enter__(self) -> "CompileSpinner":
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()


class TrainingDashboard:
    """
    Live terminal dashboard for one training stage.

    Usage:
        dash = TrainingDashboard(stage=1, n_tokens_target=1_500_000_000)
        with dash:
            for step in range(total_steps):
                ...
                dash.update(step, tokens_seen, loss, lr, tps)
    """

    def __init__(self,
                 stage: int,
                 n_tokens_target: int,
                 resume_step: int = 0,
                 resume_tokens: int = 0):
        self.stage           = stage
        self.n_tokens_target = n_tokens_target
        self.stage_name      = STAGE_NAMES.get(stage, f"Stage {stage}")

        self._console      = Console()
        self._live: Optional[Live] = None

        # Stats history
        self._loss_hist: deque[float] = deque(maxlen=50)
        self._tps_hist:  deque[float] = deque(maxlen=20)
        self._prev_loss: Optional[float] = None

        # Gate evaluation result (updated externally)
        self.gate_score:  Optional[float] = None
        self.gate_passed: bool = False

        # Last checkpoint info
        self.last_ckpt_step: Optional[int] = None

        # Progress bar
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self._console,
            refresh_per_second=4,
        )
        self._task = self._progress.add_task(
            f"Stage {stage}",
            total=n_tokens_target,
            completed=resume_tokens,
        )

        self._step   = resume_step
        self._tokens = resume_tokens
        self._loss   = 0.0
        self._lr     = 0.0
        self._tps    = 0.0
        self._t_start = time.time()

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, step: int, tokens_seen: int, loss: float,
               lr: float, tps: float) -> None:
        self._step   = step
        self._tokens = tokens_seen
        self._loss   = loss
        self._lr     = lr
        self._tps    = tps
        self._loss_hist.append(loss)
        self._tps_hist.append(tps)
        self._progress.update(self._task, completed=tokens_seen)
        if self._live:
            self._live.update(self._build_layout())

    def set_gate_result(self, score: float, passed: bool) -> None:
        self.gate_score  = score
        self.gate_passed = passed
        if self._live:
            self._live.update(self._build_layout())

    def set_checkpoint(self, step: int) -> None:
        self.last_ckpt_step = step

    def print(self, msg: str) -> None:
        """Print a message above the live panel without triggering a re-render."""
        if self._live:
            self._live.console.print(msg)

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "TrainingDashboard":
        self._live = Live(
            self._build_layout(),
            console=self._console,
            refresh_per_second=10,
            screen=False,
            vertical_overflow="visible",
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args) -> None:
        if self._live:
            self._live.__exit__(*args)
            self._live = None

    # ── Layout builder ────────────────────────────────────────────────────────

    def _build_layout(self) -> Panel:
        elapsed = time.time() - self._t_start

        # ── Stats table ───────────────────────────────────────────────────
        stats = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        stats.add_column("key",   style="bold cyan",  no_wrap=True, width=18)
        stats.add_column("value", style="white",      no_wrap=False)

        loss_vals = list(self._loss_hist)
        tps_vals  = list(self._tps_hist)

        avg_loss = sum(loss_vals[-20:]) / max(len(loss_vals[-20:]), 1)
        avg_tps  = sum(tps_vals)        / max(len(tps_vals), 1)
        arr      = _arrow(self._loss, self._prev_loss)
        self._prev_loss = self._loss

        def _fmt_tok(n: int) -> str:
            if n >= 1_000_000_000: return f"{n/1e9:.2f}B"
            if n >= 1_000_000:     return f"{n/1e6:.1f}M"
            return f"{n/1e3:.0f}K"

        # ETA from current throughput
        remaining = max(self.n_tokens_target - self._tokens, 0)
        if self._tps > 0:
            eta_str = _fmt_time(remaining / self._tps)
        else:
            eta_str = "—"

        stats.add_row("Step",   f"{self._step:,}")
        stats.add_row("Tokens", f"{_fmt_tok(self._tokens)}  /  {_fmt_tok(self.n_tokens_target)}")
        stats.add_row("Loss",       f"{self._loss:.4f}  {arr}  "
                                    f"[dim]{_sparkline(loss_vals)}[/dim]")
        stats.add_row("Loss (avg)", f"{avg_loss:.4f}  [dim](last 20 steps)[/dim]")
        stats.add_row("LR",         f"{self._lr:.2e}")
        stats.add_row("Speed",      f"{self._tps/1000:.1f} K tok/s  "
                                    f"[dim](avg {avg_tps/1000:.1f} K)[/dim]")
        stats.add_row("Elapsed",    _fmt_time(elapsed))
        stats.add_row("ETA",        f"[bold]{eta_str}[/bold]  [dim]remaining[/dim]")
        stats.add_row("GPU memory", _mem_str())

        if self.last_ckpt_step is not None:
            steps_ago = self._step - self.last_ckpt_step
            stats.add_row("Last ckpt", f"step {self.last_ckpt_step:,}  "
                                       f"[dim]({steps_ago:,} steps ago)[/dim]")

        # ── Gate row ──────────────────────────────────────────────────────
        gate_name, gate_thresh = STAGE_GATES.get(self.stage, ("─", 0))
        if self.gate_score is None:
            gate_val = Text("not evaluated yet", style="dim")
        elif self.gate_passed:
            gate_val = Text(f"{self.gate_score:.3f}  ✓  PASSED", style="bold green")
        else:
            gate_val = Text(f"{self.gate_score:.3f}  ✗  need ≥ {gate_thresh:.2f}",
                            style="yellow")
        stats.add_row(f"Gate ({gate_name})", gate_val)

        # ── Progress bar ──────────────────────────────────────────────────
        pct  = min(self._tokens / max(self.n_tokens_target, 1) * 100, 100)
        bars = int(pct / 2.5)
        bar  = ("█" * bars + "░" * (40 - bars))
        progress_line = Text()
        progress_line.append(bar, style="green" if pct >= 100 else "cyan")
        progress_line.append(f"  {pct:.1f}%")

        # ── Assemble panel ────────────────────────────────────────────────
        layout = Table.grid(padding=(0, 0))
        layout.add_row(progress_line)
        layout.add_row("")
        layout.add_row(stats)

        title = (f"[bold]Stage {self.stage}[/bold]  "
                 f"[dim]{self.stage_name}[/dim]")
        return Panel(layout, title=title, border_style="bright_blue",
                     padding=(0, 1))


def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"
