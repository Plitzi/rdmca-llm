"""
Stage 5 (reasoning / chain-of-thought) data-source guards.

Synthetic CoT must open AND close its <think> block and put the answer AFTER it, so
that with thinking turned OFF the model still shows exactly the answer line (never the
scratchpad, never empty). Lives with the stage so deleting the stage takes its test too.
"""

import itertools

from models.cognition.stage05_reasoning.sources import gen_cot
from models.cognition.uses.common.agent import visible_stream_text
from src.plugins.sdk import REASONING_SPECIALS

THINK_OPEN, THINK_CLOSE = REASONING_SPECIALS


def test_gen_cot_closes_think_and_answers():
    for rec in itertools.islice(gen_cot(200, seed=7), 200):
        t = rec["text"]
        assert THINK_OPEN in t and THINK_CLOSE in t  # block opened AND closed
        assert t.index(THINK_OPEN) < t.index(THINK_CLOSE)  # in the right order
        assert "The answer is" in t.split(THINK_CLOSE, 1)[1]  # answer AFTER the block
        # think OFF would show exactly the answer line — never empty, never the scratchpad
        vis = visible_stream_text(t)
        assert vis.strip().startswith("The answer is")
        assert THINK_OPEN not in vis
