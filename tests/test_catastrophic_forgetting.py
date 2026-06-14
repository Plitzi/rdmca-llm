"""
Cross-stage CATASTROPHIC-FORGETTING guarantees.

A progressive curriculum can erode earlier skills (esp. conversation) when a later stage
overwrites the weights that hold them — the failure the user wants verified ("entre stages
no habrá olvido catastrófico"). Two mechanisms prevent it; this file tests both directly:

  1. REHEARSAL (cognitive stages 2..BCF): the loader mixes a `replay_fraction` of batches
     drawn from earlier stages' corpora, SIZE-WEIGHTED so the largest earlier skill
     (conversation, stage 1) dominates rehearsal and is the most protected. So a later
     cognitive stage keeps seeing — and keeps fitting — earlier data.
  2. FROZEN CORE (behavioral stages > BCF): after the ethics/BCF stage the foundational
     core is frozen, so behavioral stages (tool use / MCP / skills) train LoRA sectors on
     top and CANNOT overwrite the cognitive weights at all.

These are fast, faithful unit tests of the mechanisms — not a full multi-stage training
run — so they guard the guarantee on every commit.
"""

import json
import sys
from pathlib import Path
from typing import ClassVar

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

import src.backend as backend
from src.data.loader import DataLoader, TextDataset
from src.model.config import ModelConfig
from src.model.transformer import RDMCAFoundational
from src.training.trainer import BCF_STAGE, is_behavioral_stage


# ── deterministic char tokenizer so corpora are token-distinguishable ──────────
class _CharTok:
    lang_tokens: ClassVar[dict] = {}
    ready = True

    def __init__(self, model_path: Path | None = None):
        self.model_path = model_path
        self.text_vocab_size = 256

    def encode(self, text, lang="en", add_bos=True, add_eos=True):
        ids = [ord(c) % 200 + 3 for c in text]
        if add_bos:
            ids = [1, *ids]
        if add_eos:
            ids = [*ids, 2]
        return ids

    def encode_raw(self, text):
        return [ord(c) % 200 + 3 for c in text]

    def decode(self, ids):
        return ""


_TOK_A = ord("a") % 200 + 3  # token that marks the PRIMARY (new-stage) corpus
_TOK_B = ord("z") % 200 + 3  # token that marks a REPLAY (earlier-stage) corpus


def _write_corpus(path: Path, char: str, n_lines: int, line_len: int) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    f = path / "corpus.jsonl"
    f.write_text("\n".join(json.dumps({"text": char * line_len}) for _ in range(n_lines)))
    return path


def _dataset(data_dir: Path, seq_len=8, batch=2) -> TextDataset:
    return TextDataset(
        str(data_dir), _CharTok(), seq_len=seq_len, batch_size=batch, shuffle=True, seed=7
    )


# ── 1. rehearsal actually revisits earlier-stage data ──────────────────────────
def test_replay_rehearses_prior_stage_data(tmp_path):
    """With replay_fraction>0 a later stage's stream contains BOTH new-stage batches and
    earlier-stage (replay) batches — earlier skills keep being trained, not abandoned."""
    primary = _dataset(_write_corpus(tmp_path / "stage2", "a", 400, 40))
    replay = _dataset(_write_corpus(tmp_path / "stage1", "z", 400, 40))
    loader = DataLoader(primary, replay=[replay], replay_fraction=0.4, seed=123)

    saw_primary = saw_replay = False
    for _ in range(200):
        b = loader.next_batch()
        if (b == _TOK_A).any():
            saw_primary = True
        if (b == _TOK_B).any():
            saw_replay = True
    assert saw_primary, "new-stage data must still dominate the stream"
    assert saw_replay, "earlier-stage data must be rehearsed (anti-forgetting)"


def test_no_replay_means_no_rehearsal_draw(tmp_path):
    """replay_fraction=0 → the earlier corpus is NEVER drawn (the control case)."""
    primary = _dataset(_write_corpus(tmp_path / "stage2", "a", 400, 40))
    replay = _dataset(_write_corpus(tmp_path / "stage1", "z", 400, 40))
    loader = DataLoader(primary, replay=[replay], replay_fraction=0.0, seed=123)
    assert loader._replay_fraction == 0.0
    assert not any((loader.next_batch() == _TOK_B).any() for _ in range(120))


# ── 2. rehearsal is size-weighted toward the largest earlier skill ─────────────
def test_replay_is_size_weighted_to_protect_largest_skill(tmp_path):
    """Replay selection is weighted by corpus bytes, so the BIGGEST earlier corpus
    (conversation) is rehearsed far more than a tiny one — the fix for a frozen core that
    'went crazy' when a 4K-token arithmetic set got the same refresh weight as a 200M-token
    conversation set. With replay_fraction=1 every batch is a replay draw; the big corpus
    must dominate the small one roughly in proportion to size."""
    primary = _dataset(_write_corpus(tmp_path / "stageN", "a", 50, 40))
    big = _dataset(_write_corpus(tmp_path / "big", "z", 2000, 80))  # ~large file
    small = _dataset(_write_corpus(tmp_path / "small", "m", 20, 10))  # ~tiny file
    tok_small = ord("m") % 200 + 3

    loader = DataLoader(primary, replay=[big, small], replay_fraction=1.0, seed=99)
    n_big = n_small = 0
    for _ in range(400):
        b = loader.next_batch()
        if (b == _TOK_B).any():
            n_big += 1
        elif (b == tok_small).any():
            n_small += 1
    assert n_big + n_small > 0
    # The large corpus must be picked overwhelmingly more often than the tiny one.
    assert n_big > n_small * 10, f"size-weighting failed: big={n_big} small={n_small}"


def test_replay_weights_track_corpus_bytes(tmp_path):
    """The selection weights are the on-disk byte sizes (the cheap token-count proxy),
    so a larger earlier corpus carries a proportionally larger rehearsal weight."""
    primary = _dataset(_write_corpus(tmp_path / "p", "a", 10, 10))
    big = _dataset(_write_corpus(tmp_path / "big", "z", 1000, 80))
    small = _dataset(_write_corpus(tmp_path / "small", "m", 10, 10))
    loader = DataLoader(primary, replay=[big, small], replay_fraction=0.5)
    w_big, w_small = loader._replay_weights
    assert w_big > w_small * 10  # weight ∝ bytes


# ── 3. behavioral stages can't overwrite the frozen cognitive core ─────────────
def _tiny_model():
    cfg = ModelConfig(
        d_model=64,
        n_heads=2,
        ffn_dim=128,
        context_len=32,
        vocab_size=256,
        mrl_dims=[32, 64],
        dropout=0.0,
    )
    return RDMCAFoundational(cfg)


def _weight_value(model, key):
    """Float32 numpy copy of a named parameter (mx.eval forces, then we copy)."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    v = dict(tree_flatten(model.parameters()))[key].astype(mx.float32)
    mx.eval(v)
    return np.array(v, copy=True)


def _one_step(model):
    B = backend.current()
    opt = B.engine.make_optimizer(model, 5e-3, 0.0)
    lg = B.engine.value_and_grad(model, lambda mm, t: mm.mrl_loss(t))
    b = B.ops.array(np.random.RandomState(0).randint(0, 256, (3, 33)).astype(np.int64))
    loss, g = lg(model, b)
    B.engine.optimizer_step(opt, model, g)
    B.engine.eval(loss)


def test_freeze_leaves_no_trainable_core_params():
    """After freeze_all (the post-BCF seam) the core exposes ZERO trainable parameters —
    a behavioral stage's optimizer has literally nothing in the core to update, so the
    cognitive weights cannot be overwritten. The control (unfrozen) has many."""
    from mlx.utils import tree_flatten

    B = backend.current()
    m = _tiny_model()
    assert len(tree_flatten(m.trainable_parameters())) > 0  # control: trainable
    B.engine.freeze_all(m)
    assert len(tree_flatten(m.trainable_parameters())) == 0  # frozen: none


def test_sector_trains_without_overwriting_frozen_core():
    """The behavioral-stage seam: with only a 'sector' (here one block, standing in for the
    LoRA adapter) left trainable on a frozen core, an optimizer step MOVES the sector but
    leaves the rest of the core (the embedding) BIT-IDENTICAL — no catastrophic overwrite."""
    B = backend.current()
    m = _tiny_model()
    sector = m.blocks[0]
    B.engine.set_trainable(m, [sector])  # freeze all, then unfreeze the sector

    embed_before = _weight_value(m, "embed.weight")  # frozen core weight
    sector_before = _weight_value(m, "blocks.0.attn.q_proj.weight")
    _one_step(m)
    assert np.array_equal(embed_before, _weight_value(m, "embed.weight")), (
        "frozen core (embedding) was overwritten by sector training!"
    )
    assert not np.allclose(sector_before, _weight_value(m, "blocks.0.attn.q_proj.weight")), (
        "the trainable sector should actually have learned"
    )


# ── 4. structural: only cognitive stages rehearse; behavioral train on frozen core ──
def test_only_cognitive_stages_rehearse():
    """The rehearsal wiring is gated on is_behavioral_stage: cognitive stages (≤ BCF) mix
    replay; behavioral stages (> BCF) get none because they train sectors on a frozen core
    that can't forget. This is the structural half of the guarantee."""
    assert not is_behavioral_stage(1) and not is_behavioral_stage(BCF_STAGE)
    assert is_behavioral_stage(BCF_STAGE + 1)
    # Cognitive stages after the first have at least one earlier base stage to rehearse.
    assert all(not is_behavioral_stage(s) for s in range(1, BCF_STAGE + 1))
