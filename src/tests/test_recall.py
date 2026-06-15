"""Cross-surface memory recall (src/memory/recall.py): the <mem> block formatting, the
model-embedding, and fused/threshold/dedup retrieval from the experience log."""

import json

import numpy as np

from src.memory.recall import Memory, MemoryRecall, memory_block


def _tiny_model():
    from src.model.config import ModelConfig
    from src.model.transformer import RDMCAFoundational

    cfg = ModelConfig(
        d_model=32, n_layers=1, n_heads=2, n_kv_heads=1, ffn_dim=64, context_len=32,
        vocab_size=64, mrl_dims=[16, 32], dropout=0.0,
    )  # fmt: skip
    return RDMCAFoundational(cfg)


class _FakeTok:
    ready = True

    def encode(self, text, add_bos=False, add_eos=False):
        return [(ord(c) % 60) + 1 for c in text][:20] or [1]


class _EmptyLTSS:
    def __len__(self):
        return 0


def test_memory_block_formats_and_empty():
    assert memory_block("") == ""
    out = memory_block("- a fact")
    assert out.startswith("<mem>") and "- a fact" in out


def test_embed_returns_vector_or_none():
    rec = MemoryRecall(_tiny_model(), _FakeTok(), ltss=_EmptyLTSS())
    assert rec.embed("hello world") is not None
    assert rec.embed("") is None  # empty text → no embedding


def test_cos_bounds():
    a = np.array([1.0, 0.0], dtype=np.float32)
    assert abs(MemoryRecall._cos(a, a) - 1.0) < 1e-5
    assert abs(MemoryRecall._cos(a, np.array([0.0, 1.0], dtype=np.float32))) < 1e-5


def test_recall_from_experiences_and_as_context(tmp_path):
    exp_path = tmp_path / "experiences.jsonl"
    exp_path.write_text(json.dumps({"text": "the capital of France is Paris"}) + "\n")
    rec = MemoryRecall(_tiny_model(), _FakeTok(), ltss=_EmptyLTSS(), experiences_path=str(exp_path))
    # Same text as the stored experience → cosine 1.0, clears the min_score threshold.
    mems = rec.recall("the capital of France is Paris", k=3, min_score=0.2)
    assert mems and isinstance(mems[0], Memory)
    ctx = rec.as_context(mems)
    assert "<mem>" in ctx and "Paris" in ctx
    assert rec.as_context([]) == ""


def test_recall_empty_query_returns_nothing(tmp_path):
    rec = MemoryRecall(_tiny_model(), _FakeTok(), ltss=_EmptyLTSS())
    assert rec.recall("") == []


class _FakeLTSS:
    """An LTSS with one stored node, to exercise recall's LTSS-search branch."""

    def __len__(self):
        return 1

    def search(self, q, k=5):
        return [("n1", 0.99)]

    def get_content(self, node_id):
        return "a remembered fact"


def test_recall_includes_ltss_hits():
    rec = MemoryRecall(_tiny_model(), _FakeTok(), ltss=_FakeLTSS())
    mems = rec.recall("anything", k=3, min_score=0.2)
    assert any(m.source == "ltss" and "remembered fact" in m.text for m in mems)
