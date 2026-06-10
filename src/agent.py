"""Agent output formatting — the model can return plain text or JSON, selectable
by the consumer. Today the chat CLI exposes it via `--format` (and the `/format`
command); a serving API would expose the same as a request field. Centralized
here so every consumer stays in sync.

The base model learned both registers: natural language (stages 1-5) and JSON
tool-use transcripts (stage 6, "Action and tool use"). `text` mode leaves
generation untouched; `json` mode primes the model toward a JSON object and
parses the result into a structured payload.
"""
from __future__ import annotations
import json
import re
from typing import Optional

OUTPUT_FORMATS = ("text", "json")

# Short priming so the model emits JSON (mirrors the agentic stage). Kept brief
# for small models and small context windows.
_JSON_PRIMER = "\nRespond with a single JSON object.\n"
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def normalize_format(fmt: Optional[str]) -> str:
    """Validate/normalize an output-format name."""
    fmt = (fmt or "text").lower()
    if fmt not in OUTPUT_FORMATS:
        raise ValueError(f"Unknown output format '{fmt}' — choose from {OUTPUT_FORMATS}.")
    return fmt


def wrap_prompt(prompt: str, fmt: str) -> str:
    """Prepare the prompt text for the chosen output format."""
    if normalize_format(fmt) == "json":
        return prompt.rstrip() + _JSON_PRIMER
    return prompt


def parse_output(text: str, fmt: str) -> dict:
    """Turn a raw generation into the structured result the consumer expects.

    text → {"format": "text", "text": ...}
    json → {"format": "json", "json": <obj|None>, "valid": bool, "raw": ...}
    """
    if normalize_format(fmt) == "text":
        return {"format": "text", "text": text}
    obj, valid = None, False
    m = _JSON_OBJ_RE.search(text)               # first {...} span in the output
    if m:
        try:
            obj = json.loads(m.group(0))
            valid = True
        except (ValueError, TypeError):
            obj = None
    return {"format": "json", "json": obj, "valid": valid, "raw": text}
