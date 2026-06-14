"""Stage 9 — Model Context Protocol (MCP). Behavioral: LoRA sector for tool use over
MCP / JSON-RPC."""

from __future__ import annotations

from src.models.cognition.stage09_mcp.sources import SOURCES
from src.models.sdk import StagePlugin

PLUGIN = StagePlugin(
    number=9,
    slug="mcp",
    name="Model Context Protocol (MCP)",
    entry_level=0,
    frozen_base=False,
    sources=SOURCES,
)
