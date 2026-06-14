"""Rendering for the training dashboard — the live panel + its formatting helpers.

Split out of dashboard.py so that file holds the dashboard STATE + I/O (updates,
log/metrics files, context manager) while this holds the VIEW: a pure function from
a TrainingDashboard's current state to a rich renderable. Nothing here mutates the
dashboard.
"""

from __future__ import annotations

import math
import time

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import src.core.backend as backend
from src.models import all_stages

# Stage names come from the registry (each plugin's declared name).
STAGE_NAMES = {p.number: p.name for p in all_stages()}

# Unicode sparkline chars (low → high)
_SPARKS = " ▁▂▃▄▅▆▇█"


def _sparkline(values: list[float], width: int = 12) -> str:
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
        peak = stats.get("peak", 0) / 1e9
        if peak <= 0:
            return "─"
        return f"{active:.1f} GB active  /  {peak:.1f} GB peak"
    except Exception:
        return "─"


def _arrow(current: float, previous: float) -> str:
    if previous is None or abs(current - previous) < 0.001:
        return "─"
    return "[green]↓[/green]" if current < previous else "[red]↑[/red]"


def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def render_panel(dash) -> Panel | Group:
    """Build the dashboard's renderable from the current state of `dash` (a
    TrainingDashboard). Pure: reads `dash`, returns a rich Panel/Group."""
    elapsed = time.time() - dash._t_start

    # ── Stats table ───────────────────────────────────────────────────
    stats = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    stats.add_column("key", style="bold cyan", no_wrap=True, width=18)
    stats.add_column("value", style="white", no_wrap=False)

    loss_vals = list(dash._loss_hist)
    tps_vals = list(dash._tps_hist)

    # Both averages use the last 20 samples (recent window). avg_tps drives the
    # ETA below, so a recent window keeps it responsive when throughput shifts
    # (quantization, load) instead of dragging an all-time average. Kept explicit
    # so it stays correct even if _tps_hist's maxlen changes.
    avg_loss = sum(loss_vals[-20:]) / max(len(loss_vals[-20:]), 1)
    avg_tps = sum(tps_vals[-20:]) / max(len(tps_vals[-20:]), 1)
    arr = _arrow(dash._loss, dash._prev_loss)
    dash._prev_loss = dash._loss

    def _fmt_tok(n: int) -> str:
        if n >= 1_000_000_000:
            return f"{n / 1e9:.2f}B"
        if n >= 1_000_000:
            return f"{n / 1e6:.1f}M"
        return f"{n / 1e3:.0f}K"

    # Train perplexity proxy and best-loss-so-far. Divide the COMPOSITE loss by
    # its CE-unit weight first (see _loss_ce_weight) so this is a faithful
    # per-token PP, not exp(MRL+MTP+aux) which overstates it badly.
    ppl_loss = dash._loss / dash._loss_ce_weight
    try:
        ppl = math.exp(ppl_loss) if ppl_loss < 30 else float("inf")
    except (OverflowError, ValueError):
        ppl = float("inf")
    best = "" if dash._best_loss == float("inf") else f" · best {dash._best_loss:.4f}"

    # ETA from the average throughput over the remaining tokens.
    remaining = max(dash.n_tokens_target - dash._tokens, 0)
    eta = _fmt_time(remaining / avg_tps) if avg_tps > 0 and remaining else "─"

    stats.add_row("Step", f"{dash._step:,}")
    stats.add_row("Tokens", f"{_fmt_tok(dash._tokens)}  /  {_fmt_tok(dash.n_tokens_target)}")
    stats.add_row("Loss", f"{dash._loss:.4f}  {arr}  [dim]{_sparkline(loss_vals)}[/dim]")
    stats.add_row("Loss (avg)", f"{avg_loss:.4f}  [dim](last 20{best})[/dim]")
    stats.add_row("Perplexity", f"{ppl:.1f}  [dim]per-token est.[/dim]")
    stats.add_row("LR", f"{dash._lr:.2e}")
    stats.add_row(
        "Speed", f"{dash._tps / 1000:.1f} K tok/s  [dim](avg {avg_tps / 1000:.1f} K)[/dim]"
    )
    # Why this speed: it's compute-bound. Per-token cost ≈ 6·N FLOP scales with
    # depth, so tok/s ≈ achieved_FLOP/s ÷ (6N). Showing N's geometry + MFLOP/tok +
    # the achieved TFLOP/s makes clear that e.g. 8 layers vs 6 (+33% FLOP/tok) is
    # WHY tok/s dropped — the hardware is saturated, not a regression.
    if dash._flops_per_tok:
        mflop = dash._flops_per_tok / 1e6
        tflops = dash._tps * dash._flops_per_tok / 1e12  # tok/s × FLOP/tok
        stats.add_row(
            "Compute",
            f"{dash.n_layers}L×{dash.d_model}d · {dash.params / 1e6:.1f}M params · "
            f"[dim]≈{mflop:.0f} MFLOP/tok · {tflops:.2f} TFLOP/s[/dim]",
        )
    if dash._grad_norm is not None:
        stats.add_row("Grad norm", f"{dash._grad_norm:.3f}")
    stats.add_row("Elapsed", _fmt_time(elapsed))
    stats.add_row("ETA", eta)
    if dash._passes is not None:
        stats.add_row("Corpus passes", f"{dash._passes}")
    stats.add_row("GPU memory", _mem_str())

    if dash.last_ckpt_step is not None:
        steps_ago = dash._step - dash.last_ckpt_step
        stats.add_row(
            "Last ckpt", f"step {dash.last_ckpt_step:,}  [dim]({steps_ago:,} steps ago)[/dim]"
        )

    # ── Gate row ──────────────────────────────────────────────────────
    # The live gate is a PERPLEXITY proxy (LOWER is better), ratcheting toward the
    # running best and gated by the per-stage floor — NOT the (future) accuracy
    # benchmark in STAGE_GATES, so it must NOT render as "need ≥ <accuracy>".
    gate_name = STAGE_NAMES.get(dash.stage, "─")
    if dash.gate_score is None:
        gate_val = Text("not evaluated yet", style="dim")
    else:
        best_s = f"  ·  best {dash.gate_best:.2f}" if dash.gate_best is not None else ""
        floor_s = f"  ·  floor ≤ {dash.gate_floor:.1f}" if dash.gate_floor is not None else ""
        # Compared to the stage's STARTING ppl (the inherited baseline): show the
        # start value + an arrow so direction is unambiguous — ↓ = improved (ppl
        # dropped), ↑ = got worse. Corrects for the inherited offset + rehearsal mix.
        entry_s = ""
        if dash.gate_baseline and math.isfinite(dash.gate_baseline) and dash.gate_baseline > 0:
            d = (dash.gate_score - dash.gate_baseline) / dash.gate_baseline * 100
            arrow = "↓" if d < 0 else ("↑" if d > 0 else "=")
            entry_s = f"  ·  start {dash.gate_baseline:.1f} {arrow}{abs(d):.0f}%"
        if dash.gate_passed:
            gate_val = Text(
                f"ppl {dash.gate_score:.2f}  ✓  new best{best_s}{entry_s}{floor_s}",
                style="bold green",
            )
        else:
            gate_val = Text(
                f"ppl {dash.gate_score:.2f}  ·  not a new best{best_s}{entry_s}{floor_s}",
                style="yellow",
            )
    stats.add_row(f"Gate ({gate_name})", gate_val)

    # ── Progress bar ──────────────────────────────────────────────────
    pct = min(dash._tokens / max(dash.n_tokens_target, 1) * 100, 100)
    bars = int(pct / 2.5)
    bar = "█" * bars + "░" * (40 - bars)
    progress_line = Text()
    progress_line.append(bar, style="green" if pct >= 100 else "cyan")
    progress_line.append(f"  {pct:.1f}%")

    # ── Assemble panel ────────────────────────────────────────────────
    layout = Table.grid(padding=(0, 0))
    layout.add_row(progress_line)
    layout.add_row("")
    layout.add_row(stats)

    title = f"[bold]Stage {dash.stage}[/bold]  [dim]{dash.stage_name}[/dim]"
    panel = Panel(layout, title=title, border_style="bright_blue", padding=(0, 1))

    # Dashboard pinned ON TOP; recent log lines BELOW it — both in one Group so
    # the whole block redraws together (robust to terminal/pane switches).
    if not dash._log:
        return panel
    # Render logs as plain Text (no markup parsing) so literal tags like
    # "[ckpt]"/"[gate]" are shown verbatim instead of being eaten as styles.
    log_panel = Panel(
        Text("\n".join(dash._log)), title="[dim]log[/dim]", border_style="dim", padding=(0, 1)
    )
    return Group(panel, log_panel)
