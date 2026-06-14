"""
Context & token accounting — one structured report per generation, shared by every
use case (chat, agent, future serving). It makes the context window legible the way
a good assistant manages its own budget: how many tokens each part costs (system
prompt, tool specs, skills, memory, history), what went in / came out / was spent
reasoning, how full the window is, the active mood, and the run stats.

`to_dict()` is the billing/telemetry payload; `render()` / `render_compact()` are
the human views. No heavy imports (no backend) so any surface can build one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


def count_tokens(tokenizer, text: str) -> int:
    """Token count of a text fragment (no BOS/EOS), 0 on empty/None/failure."""
    if not text:
        return 0
    try:
        return len(tokenizer.encode(text, add_bos=False, add_eos=False))
    except Exception:
        try:
            return len(tokenizer.encode_raw(text))
        except Exception:
            return 0


@dataclass
class ContextReport:
    """One generation's context budget + token accounting. All counts are plain
    ints so this stays backend-free and JSON-serializable for billing."""

    surface: str = "chat"  # which use case produced it (chat | agent | …)
    context_len: int = 0
    # context-window composition (tokens each part occupies)
    system_tokens: int = 0  # system prompt / persona (+ mood line)
    tools_tokens: int = 0  # tool specifications
    skills_tokens: int = 0  # injected SKILL.md context
    memory_tokens: int = 0  # memory/grounding injected into the prompt
    history_tokens: int = 0  # prior conversation turns
    # per-generation token flow
    tokens_in: int = 0  # full context fed to the model this turn
    tokens_out: int = 0  # answer tokens generated
    tokens_reasoning: int = 0  # <think> scratchpad tokens (subset of effort)
    # mood
    mood: str = "neutral"
    mood_dist: dict[str, float] | None = None
    # available resources (counts, not context tokens)
    memory_files: int = 0  # stored experiences available
    skills_available: int = 0
    tools_available: int = 0
    # run stats
    tps: float = 0.0
    params: dict = field(default_factory=dict)  # temp, top_p, think, format, lang…

    @property
    def used(self) -> int:
        return self.tokens_in + self.tokens_out

    @property
    def free(self) -> int:
        return max(0, self.context_len - self.used)

    @property
    def fill_pct(self) -> float:
        return (100.0 * self.used / self.context_len) if self.context_len else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(used=self.used, free=self.free, fill_pct=round(self.fill_pct, 1))
        return d

    def _mood_str(self) -> str:
        if not self.mood_dist:
            return self.mood
        top = sorted(self.mood_dist.items(), key=lambda kv: -kv[1])[:3]
        return f"{self.mood}  [" + " · ".join(f"{m} {p:.2f}" for m, p in top) + "]"

    def render(self) -> str:
        """Full multi-line context panel."""

        def row(label, tok, note=""):
            return f"  {label:<17}{tok:>6} tok" + (f"   {note}" if note else "")

        lines = [f"context ── {self.surface} " + "─" * 32]
        lines.append(row("system prompt", self.system_tokens))
        if self.surface == "agent" or self.tools_tokens:
            lines.append(
                row("system tools", self.tools_tokens, f"({self.tools_available} available)")
            )
            lines.append(row("skills", self.skills_tokens, f"({self.skills_available} available)"))
        lines.append(row("memory", self.memory_tokens, f"({self.memory_files} experiences)"))
        lines.append(row("history", self.history_tokens))
        lines.append("  " + "·" * 38)
        lines.append(row("tokens in", self.tokens_in))
        lines.append(row("tokens out", self.tokens_out))
        lines.append(row("tokens reasoning", self.tokens_reasoning))
        lines.append(
            f"  {'window':<17}{self.used:>6}/{self.context_len} "
            f"({self.fill_pct:.0f}%) · {self.free} free"
        )
        lines.append(f"  {'mood':<17}{self._mood_str()}")
        p = self.params
        stat = f"{self.tps:.1f} tok/s"
        for k in ("temp", "top_p", "think", "format", "lang"):
            if k in p:
                stat += f" · {k}={p[k]}"
        lines.append(f"  {'stats':<17}{stat}")
        return "\n".join(lines)

    def render_compact(self) -> str:
        """One-line summary for per-turn display."""
        r = (
            f"in {self.tokens_in} · out {self.tokens_out} · think {self.tokens_reasoning}"
            f" · {self.used}/{self.context_len} ({self.fill_pct:.0f}%)"
            f" · mood {self.mood} · {self.tps:.0f} tok/s"
        )
        return f"  [{r}]"
