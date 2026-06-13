"""
Training Dashboard — rich terminal UI for RDMCA stage training.
Displays a live-updating panel with loss, speed, ETA, memory and gate status.
"""
from __future__ import annotations
import math
import time
from collections import deque
from pathlib import Path
from typing import Optional

import src.backend as backend

from rich.console import Console, Group
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

# Stage gates/names come from the shared source of truth (src/training/stages.py).
from src.training.stages import STAGE_NAMES


def _sparkline(values: list[float], width: int = 12) -> str:
    import math
    # Drop non-finite values (e.g. NaN from an unstable fp16 run) so the
    # dashboard never crashes; show a marker if nothing finite remains.
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return "⚠ NaN" if values else "─" * width
    lo, hi = min(finite), max(finite)
    rng = hi - lo or 1.0
    chars = [_SPARKS[int((v - lo) / rng * (len(_SPARKS) - 1))] for v in finite[-width:]]
    return "".join(chars)


def _mem_str() -> str:
    """Active accelerator memory via the active backend (returns '─' if
    unavailable, e.g. CPU-only torch)."""
    try:
        stats = backend.current().engine.memory_stats()
        active = stats.get("active", 0) / 1e9
        peak   = stats.get("peak", 0) / 1e9
        if peak <= 0:
            return "─"
        return f"{active:.1f} GB active  /  {peak:.1f} GB peak"
    except Exception:
        return "─"


def _arrow(current: float, previous: float) -> str:
    if previous is None or abs(current - previous) < 0.001:
        return "─"
    return "[green]↓[/green]" if current < previous else "[red]↑[/red]"


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
                 resume_tokens: int = 0,
                 params: int = 0,
                 n_layers: int = 0,
                 d_model: int = 0,
                 plain: bool = False,
                 log_path=None,
                 loss_ce_weight: float = 1.0):
        self.stage           = stage
        self.n_tokens_target = n_tokens_target
        self.stage_name      = STAGE_NAMES.get(stage, f"Stage {stage}")
        # The training `loss` is a COMPOSITE: the MRL head mean (1 CE-unit) plus
        # `mtp_loss_weight` per MTP head. exp(composite) wildly OVERSTATES perplexity
        # (e.g. exp(12)=184K at init, when the true per-token PP ≈ exp(9)=vocab size).
        # Dividing by the CE-unit weight before exp() gives a FAITHFUL per-token PP.
        # (The authoritative PP is still the gate's eval_ce; this is the live trend.)
        self._loss_ce_weight = max(float(loss_ce_weight), 1e-6)
        # plain=True (or RDMCA_PLAIN_LOGS): no live/animated panel — emit ordinary
        # scrolling terminal lines instead. The live dashboard repaints its region
        # several times a second, which (a) drops older lines out of the fixed log
        # panel and (b) clears any text selection mid-copy (the "flickering"). Plain
        # mode trades the pretty panel for a full, selectable, persistent scrollback.
        import os
        self._plain = bool(plain) or os.environ.get("RDMCA_PLAIN_LOGS", "").lower() \
            in ("1", "true", "yes", "on")
        self._plain_last = -1                  # last step printed in plain mode

        # Persistent plain-text log of the WHOLE run (loss evolution, gates,
        # checkpoints), independent of the live panel — so nothing scrolls out of
        # reach and the history is greppable/copyable after the fact. Opened in
        # append mode so --resume keeps adding to the same file.
        self._flog = None
        # Structured metrics sink (metrics.csv next to the log) for plotting the run —
        # loss/ppl/lr/tps per step + val-ppl/best per gate eval. Machine-readable so
        # scripts/plot_metrics.py (or any tool) can chart the curves after the fact.
        self._metrics = None
        if log_path is not None:
            try:
                Path(log_path).parent.mkdir(parents=True, exist_ok=True)
                self._flog = open(log_path, "a", encoding="utf-8")
                mpath = Path(log_path).with_name("metrics.csv")
                new = not mpath.exists() or mpath.stat().st_size == 0
                self._metrics = open(mpath, "a", encoding="utf-8")
                if new:
                    self._metrics.write("kind,step,tokens_m,loss,ppl,lr,tps,grad_norm,"
                                        "val_ppl,best_val_ppl,passed\n")
                    self._metrics.flush()
            except OSError as e:
                self._console.print(f"[yellow][log] could not open {log_path}: {e}[/yellow]")

    def _metric_row(self, kind, *, step="", tokens_m="", loss="", ppl="", lr="", tps="",
                    grad_norm="", val_ppl="", best_val_ppl="", passed="") -> None:
        if self._metrics is None:
            return
        try:
            self._metrics.write(f"{kind},{step},{tokens_m},{loss},{ppl},{lr},{tps},"
                                f"{grad_norm},{val_ppl},{best_val_ppl},{passed}\n")
            self._metrics.flush()
        except (OSError, ValueError):
            self._metrics = None
        # Model geometry — shown next to Speed so the dev sees WHY the tok/s is what it
        # is: throughput is compute-bound, and per-token compute ≈ 6·N (fwd+bwd, the
        # Kaplan/Chinchilla rule) scales with depth. Doubling layers ⇒ ~proportionally
        # more FLOP/tok ⇒ fewer tok/s on the same hardware (not a regression).
        self.params          = params
        self.n_layers        = n_layers
        self.d_model         = d_model
        self._flops_per_tok  = 6 * params          # train fwd+bwd ≈ 6N FLOP/token

        self._console      = Console()
        self._live: Optional[Live] = None
        self._is_tty       = self._console.is_terminal      # animate only on a TTY
        # Recent log lines shown UNDER the pinned dashboard (one redraw region, so
        # switching terminals/tmux panes never leaves the panel half-drawn). On a
        # non-TTY (piped to a file) we print lines normally so the file keeps them.
        self._log: deque[str] = deque(maxlen=10)

        # Stats history
        self._loss_hist: deque[float] = deque(maxlen=50)
        self._tps_hist:  deque[float] = deque(maxlen=20)
        self._prev_loss: Optional[float] = None

        # Gate evaluation result (updated externally). The gate is a PERPLEXITY proxy
        # (LOWER is better), ratcheting toward `gate_best` and floored at `gate_floor`.
        self.gate_score:  Optional[float] = None
        self.gate_passed: bool = False
        self.gate_floor:  Optional[float] = None   # per-stage perplexity floor (≤)
        self.gate_best:   Optional[float] = None   # running best (the ratchet bar)

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
        self._grad_norm: Optional[float] = None
        self._passes: Optional[int] = None
        self._best_loss = float("inf")
        self._t_start = time.time()

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, step: int, tokens_seen: int, loss: float,
               lr: float, tps: float, grad_norm: float | None = None,
               passes: int | None = None) -> None:
        self._step   = step
        self._tokens = tokens_seen
        self._loss   = loss
        self._lr     = lr
        self._tps    = tps
        if grad_norm is not None:
            self._grad_norm = grad_norm
        if passes is not None:
            self._passes = passes
        if loss < self._best_loss:
            self._best_loss = loss
        self._loss_hist.append(loss)
        self._tps_hist.append(tps)
        self._progress.update(self._task, completed=tokens_seen)
        if self._live:
            self._live.update(self._build_layout())
        # A concise progress line for the plain console AND/OR the persistent log
        # file (built once). The live panel already shows this, so skip the console
        # echo there, but still record it to the file so the file has the full curve.
        if (self._plain or self._flog) and step != self._plain_last:
            self._plain_last = step
            ppl = math.exp(min(loss / self._loss_ce_weight, 20)) if loss > 0 else float("nan")
            gn  = f" | gnorm={self._grad_norm:.2f}" if self._grad_norm is not None else ""
            line = (f"[step {step:>7,}] {tokens_seen/1e6:8.1f}M tok | loss={loss:.4f} "
                    f"| ppl~{ppl:6.2f} | lr={lr:.2e} | {tps:6.0f} tok/s "
                    f"| best={self._best_loss:.4f}{gn}")
            if self._plain:
                # markup=False so the literal "[step N]" brackets aren't parsed as
                # rich markup tags (which would silently drop them).
                self._console.print(line, highlight=False, markup=False)
            self._record(line)
            self._metric_row("train", step=step, tokens_m=f"{tokens_seen/1e6:.3f}",
                             loss=f"{loss:.4f}", ppl=f"{ppl:.4f}", lr=f"{lr:.3e}",
                             tps=f"{tps:.0f}",
                             grad_norm=("" if self._grad_norm is None
                                        else f"{self._grad_norm:.3f}"))

    def set_target(self, n_tokens_target: int) -> None:
        """Adjust the token target (and progress-bar total) mid-run — e.g. when the
        trainer caps re-cycling of a small corpus to a lower effective budget, so
        the bar still completes at 100% against the real target."""
        self.n_tokens_target = n_tokens_target
        self._progress.update(self._task, total=n_tokens_target)
        if self._live:
            self._live.update(self._build_layout())

    def set_gate_result(self, score: float, passed: bool,
                        threshold: float = None, best: float = None) -> None:
        self.gate_score  = score
        self.gate_passed = passed
        if threshold is not None:
            self.gate_floor = threshold
        if best is not None and math.isfinite(best):
            self.gate_best = best
        self._record(f"[gate] score={score:.4f} -> {'NEW BEST' if passed else 'not a new best'}")
        self._metric_row("gate", step=self._step, tokens_m=f"{self._tokens/1e6:.3f}",
                         val_ppl=f"{score:.4f}",
                         best_val_ppl=("" if self.gate_best is None else f"{self.gate_best:.4f}"),
                         passed=int(bool(passed)))
        if self._live:
            self._live.update(self._build_layout())

    def set_checkpoint(self, step: int) -> None:
        self.last_ckpt_step = step

    def _record(self, msg: str) -> None:
        """Append a timestamped line to the persistent training log file (if any),
        flushing each line so a crash still leaves a complete log."""
        if self._flog is None:
            return
        try:
            self._flog.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")
            self._flog.flush()
        except (OSError, ValueError):
            self._flog = None                  # don't let a logging error kill training

    def print(self, msg: str) -> None:
        """Record a log line. On the live panel it appears in the panel's log area
        (under the dashboard); in plain mode (or piped to a file) it is printed
        plainly. Either way it is also appended to the persistent log file."""
        self._record(msg)
        if self._is_tty and self._live:
            self._log.append(msg)
            self._live.update(self._build_layout())
        else:
            self._console.print(msg, markup=False, highlight=False)

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "TrainingDashboard":
        self._record(f"=== Stage {self.stage} ({self.stage_name}) | "
                     f"{self.params/1e6:.1f}M params | target {self.n_tokens_target/1e6:.0f}M tokens "
                     f"| target_loss curve below ===")
        if self._is_tty and not self._plain:    # animate only on a real terminal (not plain)
            self._live = Live(
                self._build_layout(),
                console=self._console,
                refresh_per_second=10,
                screen=False,
                # "crop" (not "visible"): on a terminal repaint/resize — e.g. switching
                # tabs/tmux panes — "visible" lets the frame overflow and redraw below the
                # stale one, duplicating the panel (you'd see "Stage 1" twice). "crop"
                # keeps Live within the known viewport so it clears + repaints cleanly.
                vertical_overflow="crop",
            )
            self._live.__enter__()
        return self

    def __exit__(self, *args) -> None:
        if self._live:
            self._live.__exit__(*args)
            self._live = None
        if self._flog is not None:
            try:
                self._flog.close()
            finally:
                self._flog = None
        if self._metrics is not None:
            try:
                self._metrics.close()
            finally:
                self._metrics = None

    # ── Layout builder ────────────────────────────────────────────────────────

    def _build_layout(self) -> Panel:
        elapsed = time.time() - self._t_start

        # ── Stats table ───────────────────────────────────────────────────
        stats = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        stats.add_column("key",   style="bold cyan",  no_wrap=True, width=18)
        stats.add_column("value", style="white",      no_wrap=False)

        loss_vals = list(self._loss_hist)
        tps_vals  = list(self._tps_hist)

        # Both averages use the last 20 samples (recent window). avg_tps drives the
        # ETA below, so a recent window keeps it responsive when throughput shifts
        # (quantization, load) instead of dragging an all-time average. Kept explicit
        # so it stays correct even if _tps_hist's maxlen changes.
        avg_loss = sum(loss_vals[-20:]) / max(len(loss_vals[-20:]), 1)
        avg_tps  = sum(tps_vals[-20:])  / max(len(tps_vals[-20:]), 1)
        arr      = _arrow(self._loss, self._prev_loss)
        self._prev_loss = self._loss

        def _fmt_tok(n: int) -> str:
            if n >= 1_000_000_000: return f"{n/1e9:.2f}B"
            if n >= 1_000_000:     return f"{n/1e6:.1f}M"
            return f"{n/1e3:.0f}K"

        # Train perplexity proxy and best-loss-so-far. Divide the COMPOSITE loss by
        # its CE-unit weight first (see _loss_ce_weight) so this is a faithful
        # per-token PP, not exp(MRL+MTP+aux) which overstates it badly.
        ppl_loss = self._loss / self._loss_ce_weight
        try:
            ppl = math.exp(ppl_loss) if ppl_loss < 30 else float("inf")
        except (OverflowError, ValueError):
            ppl = float("inf")
        best = "" if self._best_loss == float("inf") else f" · best {self._best_loss:.4f}"

        # ETA from the average throughput over the remaining tokens.
        remaining = max(self.n_tokens_target - self._tokens, 0)
        eta = _fmt_time(remaining / avg_tps) if avg_tps > 0 and remaining else "─"

        stats.add_row("Step",       f"{self._step:,}")
        stats.add_row("Tokens",     f"{_fmt_tok(self._tokens)}  /  {_fmt_tok(self.n_tokens_target)}")
        stats.add_row("Loss",       f"{self._loss:.4f}  {arr}  "
                                    f"[dim]{_sparkline(loss_vals)}[/dim]")
        stats.add_row("Loss (avg)", f"{avg_loss:.4f}  [dim](last 20{best})[/dim]")
        stats.add_row("Perplexity", f"{ppl:.1f}  [dim]per-token est.[/dim]")
        stats.add_row("LR",         f"{self._lr:.2e}")
        stats.add_row("Speed",      f"{self._tps/1000:.1f} K tok/s  "
                                    f"[dim](avg {avg_tps/1000:.1f} K)[/dim]")
        # Why this speed: it's compute-bound. Per-token cost ≈ 6·N FLOP scales with
        # depth, so tok/s ≈ achieved_FLOP/s ÷ (6N). Showing N's geometry + MFLOP/tok +
        # the achieved TFLOP/s makes clear that e.g. 8 layers vs 6 (+33% FLOP/tok) is
        # WHY tok/s dropped — the hardware is saturated, not a regression.
        if self._flops_per_tok:
            mflop = self._flops_per_tok / 1e6
            tflops = self._tps * self._flops_per_tok / 1e12      # tok/s × FLOP/tok
            stats.add_row("Compute",
                          f"{self.n_layers}L×{self.d_model}d · {self.params/1e6:.1f}M params · "
                          f"[dim]≈{mflop:.0f} MFLOP/tok · {tflops:.2f} TFLOP/s[/dim]")
        if self._grad_norm is not None:
            stats.add_row("Grad norm", f"{self._grad_norm:.3f}")
        stats.add_row("Elapsed",    _fmt_time(elapsed))
        stats.add_row("ETA",        eta)
        if self._passes is not None:
            stats.add_row("Corpus passes", f"{self._passes}")
        stats.add_row("GPU memory", _mem_str())

        if self.last_ckpt_step is not None:
            steps_ago = self._step - self.last_ckpt_step
            stats.add_row("Last ckpt", f"step {self.last_ckpt_step:,}  "
                                       f"[dim]({steps_ago:,} steps ago)[/dim]")

        # ── Gate row ──────────────────────────────────────────────────────
        # The live gate is a PERPLEXITY proxy (LOWER is better), ratcheting toward the
        # running best and gated by the per-stage floor — NOT the (future) accuracy
        # benchmark in STAGE_GATES, so it must NOT render as "need ≥ <accuracy>".
        gate_name = STAGE_NAMES.get(self.stage, "─")
        if self.gate_score is None:
            gate_val = Text("not evaluated yet", style="dim")
        else:
            best_s  = f"  ·  best {self.gate_best:.2f}" if self.gate_best is not None else ""
            floor_s = f"  ·  floor ≤ {self.gate_floor:.1f}" if self.gate_floor is not None else ""
            if self.gate_passed:
                gate_val = Text(f"ppl {self.gate_score:.2f}  ✓  new best{best_s}{floor_s}",
                                style="bold green")
            else:
                gate_val = Text(f"ppl {self.gate_score:.2f}  ·  not a new best{best_s}{floor_s}",
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
        panel = Panel(layout, title=title, border_style="bright_blue",
                      padding=(0, 1))

        # Dashboard pinned ON TOP; recent log lines BELOW it — both in one Group so
        # the whole block redraws together (robust to terminal/pane switches).
        if not self._log:
            return panel
        # Render logs as plain Text (no markup parsing) so literal tags like
        # "[ckpt]"/"[gate]" are shown verbatim instead of being eaten as styles.
        log_panel = Panel(Text("\n".join(self._log)), title="[dim]log[/dim]",
                          border_style="dim", padding=(0, 1))
        return Group(panel, log_panel)


def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"
