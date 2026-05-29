"""
Inference Entry Point — RDMCA
Autoregressive generation with sector-adaptive context.
Uses the frozen foundational core + active LoRA sector adapters.
"""
from __future__ import annotations
from typing import List, Optional

import mlx.core as mx
import numpy as np


class RDMCAGenerator:
    """
    Autoregressive text generator with sector-weighted LoRA injection.
    After foundational training is complete (Phase 1), sectors are
    attached and each token pass runs through the relevant adapters.
    """

    def __init__(self, model, sectors: dict,
                 tokenizer, sector_router):
        self.model          = model
        self.sectors        = sectors
        self.tokenizer      = tokenizer
        self.sector_router  = sector_router

    def generate(self, prompt: str,
                 max_new_tokens: int = 256,
                 temperature: float = 0.8,
                 top_p: float = 0.9,
                 lang: str = "en") -> str:
        """
        Generate text from a prompt.
        Returns the decoded completion (prompt not included).
        """
        input_ids = self.tokenizer.encode(prompt, lang=lang, add_eos=False)
        tokens    = mx.array(input_ids)[None]   # [1, S]
        generated: List[int] = []

        for _ in range(max_new_tokens):
            logits = self.model.logits(tokens)          # [1, S, vocab]
            next_logits = logits[0, -1, :]              # [vocab]

            if temperature > 0:
                next_logits = next_logits / temperature
                next_token = _sample_top_p(next_logits, top_p)
            else:
                next_token = int(mx.argmax(next_logits).item())

            if next_token == 3:   # EOS
                break

            generated.append(next_token)
            new_tok = mx.array([[next_token]])
            tokens  = mx.concatenate([tokens, new_tok], axis=1)

        return self.tokenizer.decode(generated)


def _sample_top_p(logits: mx.array, p: float) -> int:
    """Nucleus sampling."""
    probs = mx.softmax(logits, axis=-1)
    probs_np = np.array(probs.tolist())
    sorted_idx  = np.argsort(probs_np)[::-1]
    sorted_prob = probs_np[sorted_idx]
    cumulative  = np.cumsum(sorted_prob)
    cutoff      = np.searchsorted(cumulative, p) + 1
    top_idx     = sorted_idx[:cutoff]
    top_prob    = sorted_prob[:cutoff]
    top_prob    = top_prob / top_prob.sum()
    return int(np.random.choice(top_idx, p=top_prob))
