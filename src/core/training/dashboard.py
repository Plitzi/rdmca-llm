"""
Training Dashboard — rich terminal UI for RDMCA stage training.
Displays a live-updating panel with loss, speed, ETA, memory and gate status.
"""

from __future__ import annotations

import math
import time
from collections import deque
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

# The live panel rendering (the "view") lives in dashboard_view; this file owns the
# dashboard STATE + I/O. STAGE_NAMES is sourced there from the registry.
from src.core.training.dashboard_view import STAGE_NAMES, render_panel


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

    def __init__(
        self,
        stage: int,
        n_tokens_target: int,
        resume_step: int = 0,
        resume_tokens: int = 0,
        params: int = 0,
        n_layers: int = 0,
        d_model: int = 0,
        plain: bool = False,
        log_path=None,
        loss_ce_weight: float = 1.0,
        append: bool = True,
        gate_baseline: float | None = None,
    ):
        self.stage = stage
        # Per-stage ENTRY perplexity (the inherited checkpoint's val PP before this
        # stage trains). The gate's absolute PP carries an offset from the previous
        # stage + the rehearsal mix, so the meaningful signal is the DELTA from entry:
        # how much THIS stage moved its own starting point. Shown next to the gate.
        self.gate_baseline = gate_baseline
        self.n_tokens_target = n_tokens_target
        self.stage_name = STAGE_NAMES.get(stage, f"Stage {stage}")
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

        self._plain = bool(plain) or os.environ.get("RDMCA_PLAIN_LOGS", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._plain_last = -1  # last step printed in plain mode

        # Persistent plain-text log of the WHOLE run (loss evolution, gates,
        # checkpoints), independent of the live panel — so nothing scrolls out of
        # reach and the history is greppable/copyable after the fact. Opened in
        # append mode so --resume keeps adding to the same file.
        self._flog = None
        # Structured metrics sink (metrics.csv next to the log) for plotting the run —
        # loss/ppl/lr/tps per step + val-ppl/best per gate eval. Machine-readable so
        # scripts/plot_metrics.py (or any tool) can chart the curves after the fact.
        self._metrics = None
        # append=True (a --resume) keeps the existing log/metrics; a FRESH run truncates
        # them so a re-train doesn't append to (and mix old-schema rows into) the old file.
        _mode = "a" if append else "w"
        if log_path is not None:
            try:
                Path(log_path).parent.mkdir(parents=True, exist_ok=True)
                self._flog = open(log_path, _mode, encoding="utf-8")  # noqa: SIM115 (instance handle, closed in close())
                mpath = Path(log_path).with_name("metrics.csv")
                new = (_mode == "w") or not mpath.exists() or mpath.stat().st_size == 0
                self._metrics = open(mpath, _mode, encoding="utf-8")  # noqa: SIM115 (instance handle, closed in close())
                if new:
                    self._metrics.write(
                        "kind,step,tokens_m,loss,ppl,lr,tps,grad_norm,"
                        "val_ppl,best_val_ppl,passed,replay,entry_ppl\n"
                    )
                    self._metrics.flush()
            except OSError as e:
                self._console.print(f"[yellow][log] could not open {log_path}: {e}[/yellow]")

        # Model geometry — shown next to Speed so the dev sees WHY the tok/s is what it
        # is: throughput is compute-bound, and per-token compute ≈ 6·N (fwd+bwd, the
        # Kaplan/Chinchilla rule) scales with depth. Doubling layers ⇒ ~proportionally
        # more FLOP/tok ⇒ fewer tok/s on the same hardware (not a regression).
        self.params = params
        self.n_layers = n_layers
        self.d_model = d_model
        self._flops_per_tok = 6 * params  # train fwd+bwd ≈ 6N FLOP/token

        self._console = Console()
        self._live: Live | None = None
        self._is_tty = self._console.is_terminal  # animate only on a TTY
        # Recent log lines shown UNDER the pinned dashboard (one redraw region, so
        # switching terminals/tmux panes never leaves the panel half-drawn). On a
        # non-TTY (piped to a file) we print lines normally so the file keeps them.
        self._log: deque[str] = deque(maxlen=10)

        # Stats history
        self._loss_hist: deque[float] = deque(maxlen=50)
        self._tps_hist: deque[float] = deque(maxlen=20)
        self._prev_loss: float | None = None

        # Gate evaluation result (updated externally). The gate is a PERPLEXITY proxy
        # (LOWER is better), ratcheting toward `gate_best` and floored at `gate_floor`.
        self.gate_score: float | None = None
        self.gate_passed: bool = False
        self.gate_floor: float | None = None  # per-stage perplexity floor (≤)
        self.gate_best: float | None = None  # running best (the ratchet bar)

        # Last checkpoint info
        self.last_ckpt_step: int | None = None

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

        self._step = resume_step
        self._tokens = resume_tokens
        self._loss = 0.0
        self._lr = 0.0
        self._tps = 0.0
        self._grad_norm: float | None = None
        self._passes: int | None = None
        self._best_loss = float("inf")
        self._t_start = time.time()
        # With rehearsal the per-step loss is BIMODAL — narrow-skill batches sit near 0
        # while interleaved conversation-replay batches are much higher. A single raw loss
        # then looks like wild "spikes" (it's just alternating populations). Track a
        # smoothed EMA per batch type so the trend is readable and forgetting is visible
        # (the rehearsal EMA climbing = conversation being eroded).
        self._ema_primary: float | None = None
        self._ema_replay: float | None = None
        self._last_replay = False

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        step: int,
        tokens_seen: int,
        loss: float,
        lr: float,
        tps: float,
        grad_norm: float | None = None,
        passes: int | None = None,
        replay: bool | None = None,
    ) -> None:
        self._step = step
        self._tokens = tokens_seen
        self._loss = loss
        self._lr = lr
        self._tps = tps
        if grad_norm is not None:
            self._grad_norm = grad_norm
        if passes is not None:
            self._passes = passes
        if loss < self._best_loss:
            self._best_loss = loss
        if replay is not None:
            self._last_replay = bool(replay)
            a = 0.1  # EMA smoothing factor
            if replay:
                self._ema_replay = (
                    loss if self._ema_replay is None else (1 - a) * self._ema_replay + a * loss
                )
            else:
                self._ema_primary = (
                    loss if self._ema_primary is None else (1 - a) * self._ema_primary + a * loss
                )
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
            gn = f" | gnorm={self._grad_norm:.2f}" if self._grad_norm is not None else ""
            # Smoothed per-type EMA so the bimodal rehearsal sawtooth reads as two stable
            # trends, not "spikes". Only shown once both populations have been seen.
            ema = ""
            if self._ema_primary is not None and self._ema_replay is not None:
                ema = f" | ema[skill {self._ema_primary:.2f} · rehearsal {self._ema_replay:.2f}]"
            line = (
                f"[step {step:>7,}] {tokens_seen / 1e6:8.1f}M tok | loss={loss:.4f} "
                f"| ppl~{ppl:6.2f} | lr={lr:.2e} | {tps:6.0f} tok/s "
                f"| best={self._best_loss:.4f}{gn}{ema}"
            )
            if self._plain:
                # markup=False so the literal "[step N]" brackets aren't parsed as
                # rich markup tags (which would silently drop them).
                self._console.print(line, highlight=False, markup=False)
            self._record(line)
            self._metric_row(
                "train",
                step=step,
                tokens_m=f"{tokens_seen / 1e6:.3f}",
                loss=f"{loss:.4f}",
                ppl=f"{ppl:.4f}",
                lr=f"{lr:.3e}",
                tps=f"{tps:.0f}",
                grad_norm=("" if self._grad_norm is None else f"{self._grad_norm:.3f}"),
                replay=int(self._last_replay),
            )

    def set_target(self, n_tokens_target: int) -> None:
        """Adjust the token target (and progress-bar total) mid-run — e.g. when the
        trainer caps re-cycling of a small corpus to a lower effective budget, so
        the bar still completes at 100% against the real target."""
        self.n_tokens_target = n_tokens_target
        self._progress.update(self._task, total=n_tokens_target)
        if self._live:
            self._live.update(self._build_layout())

    def set_gate_result(
        self, score: float, passed: bool, threshold: float | None = None, best: float | None = None
    ) -> None:
        self.gate_score = score
        self.gate_passed = passed
        if threshold is not None:
            self.gate_floor = threshold
        if best is not None and math.isfinite(best):
            self.gate_best = best
        self._record(f"[gate] score={score:.4f} -> {'NEW BEST' if passed else 'not a new best'}")
        self._metric_row(
            "gate",
            step=self._step,
            tokens_m=f"{self._tokens / 1e6:.3f}",
            val_ppl=f"{score:.4f}",
            best_val_ppl=("" if self.gate_best is None else f"{self.gate_best:.4f}"),
            passed=int(bool(passed)),
            entry_ppl=("" if self.gate_baseline is None else f"{self.gate_baseline:.4f}"),
        )
        if self._live:
            self._live.update(self._build_layout())

    def set_checkpoint(self, step: int) -> None:
        self.last_ckpt_step = step

    def _metric_row(
        self,
        kind,
        *,
        step="",
        tokens_m="",
        loss="",
        ppl="",
        lr="",
        tps="",
        grad_norm="",
        val_ppl="",
        best_val_ppl="",
        passed="",
        replay="",
        entry_ppl="",
    ) -> None:
        """Append one machine-readable row to metrics.csv (if any) — see plot_metrics.py."""
        if self._metrics is None:
            return
        try:
            self._metrics.write(
                f"{kind},{step},{tokens_m},{loss},{ppl},{lr},{tps},"
                f"{grad_norm},{val_ppl},{best_val_ppl},{passed},{replay},"
                f"{entry_ppl}\n"
            )
            self._metrics.flush()
        except (OSError, ValueError):
            self._metrics = None

    def _record(self, msg: str) -> None:
        """Append a timestamped line to the persistent training log file (if any),
        flushing each line so a crash still leaves a complete log."""
        if self._flog is None:
            return
        try:
            self._flog.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")
            self._flog.flush()
        except (OSError, ValueError):
            self._flog = None  # don't let a logging error kill training

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

    def __enter__(self) -> TrainingDashboard:
        self._record(
            f"=== Stage {self.stage} ({self.stage_name}) | "
            f"{self.params / 1e6:.1f}M params | target {self.n_tokens_target / 1e6:.0f}M tokens "
            f"| target_loss curve below ==="
        )
        if self._is_tty and not self._plain:  # animate only on a real terminal (not plain)
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

    def _build_layout(self):
        """Render the live panel from current state (see dashboard_view.render_panel)."""
        return render_panel(self)
