"""Dynamic context manager (src/routing/context_manager.py): per-sector slots, routing
(via a supplied route_fn or the single-slot fallback), overflow eviction to the episodic
buffer, recency-ordered assembly, and wiring to a model via build_context_manager."""

import numpy as np

from src.memory.episodic_buffer import EpisodicBuffer
from src.routing.context_manager import ContextManager, build_context_manager


def test_single_slot_fallback_assembles_and_caps():
    cm = ContextManager(d_model=8, context_len=32, slot_len=16)
    cm.add(list(range(10)))
    asm = cm.assemble()
    assert asm  # tokens are retained
    assert cm.assemble(max_len=4) == asm[-4:]  # cap keeps the most recent
    assert 1 in cm.active_sectors()  # fallback routes to slot 1


def test_route_fn_directs_to_chosen_sector():
    routed_to = 3
    cm = ContextManager(
        d_model=8, context_len=32, slot_len=16, route_fn=lambda toks: [(routed_to, 0.9)]
    )
    cm.add(list(range(8)))
    assert routed_to in cm.active_sectors()


def test_overflow_evicts_to_episodic_buffer():
    buf = EpisodicBuffer()
    decoded = []
    cm = ContextManager(
        d_model=8,
        context_len=64,
        slot_len=4,  # tiny slot → forces eviction
        buffer=buf,
        embed_fn=lambda toks: np.ones(8, dtype=np.float32),
        decode_fn=lambda toks: decoded.append(toks) or "evicted-text",
        route_fn=lambda toks: [(1, 1.0)],
    )
    cm.add(list(range(40)))  # well over slot_len → overflow
    assert len(buf) >= 1  # evicted chunks landed in the buffer, not discarded
    assert decoded  # the decoder was used so the experience carries real text


def test_clear_resets_state():
    cm = ContextManager(d_model=8, context_len=32, slot_len=16)
    cm.add(list(range(10)))
    cm.clear()
    assert cm.assemble() == [] and cm.active_sectors() == []


def _tiny_model():
    from src.model.config import ModelConfig
    from src.model.transformer import RDMCAFoundational

    cfg = ModelConfig(
        d_model=32, n_layers=1, n_heads=2, n_kv_heads=1, ffn_dim=64, context_len=64,
        vocab_size=128, mrl_dims=[16, 32], dropout=0.0,
    )  # fmt: skip
    return RDMCAFoundational(cfg)


def test_build_context_manager_wires_embed_and_routes_gracefully():
    model = _tiny_model()
    cm = build_context_manager(model, tokenizer=None)
    assert isinstance(cm, ContextManager)
    # embed_fn produces the last-token hidden state vector
    vec = cm.embed_fn([1, 2, 3, 4])
    assert vec is not None and vec.shape == (model.cfg.d_model,)
    assert cm.embed_fn([]) is None
    # No attached gate → route_fn returns None (manager falls back), no crash
    assert cm.route_fn([1, 2, 3]) is None
    cm.add([1, 2, 3, 4, 5])
    assert cm.assemble()
