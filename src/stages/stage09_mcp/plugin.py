"""Stage 9 — Model Context Protocol (MCP). Behavioral: LoRA sector for tool use over
MCP / JSON-RPC."""

from __future__ import annotations

from src.stages.base import StageKind, StagePlugin
from src.stages.stage09_mcp.sources import SOURCES

PLUGIN = StagePlugin(
    number=9,
    slug="mcp",
    name="Model Context Protocol (MCP)",
    entry_level=0,
    kind=StageKind.BEHAVIORAL,
    sources=SOURCES,
)
