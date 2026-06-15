"""cognition's checkpoint-quality banner formatter (loading.describe_checkpoint_meta).
The generic best→final→latest resolution it pairs with is the framework's and is covered by
src/tests/test_checkpoint_resolution.py; here we only check cognition's one-line summary."""

from models.cognition.uses.common.loading import describe_checkpoint_meta


def test_describe_checkpoint_meta_formats_quality():
    desc = describe_checkpoint_meta(
        {"score": 14.2, "step": 7920, "tokens": 64_000_000, "met_bar": True}
    )
    assert "val ppl 14.20" in desc and "step 7,920" in desc
    assert "64.0M tok" in desc and "met_bar=True" in desc


def test_describe_checkpoint_meta_empty():
    assert describe_checkpoint_meta(None) == ""
    assert describe_checkpoint_meta({}) == ""
