"""
Phase 4 Acceptance Tests — Audio modality & Parametric Growth (PGQ)
"""
import numpy as np

from src.modalities.audio import AudioVQVAE, logmel, N_MELS
from src.modalities.vocab import AUDIO_VOCAB_SIZE


def _sine(secs=1.0, sr=16000, f=220.0):
    t = np.linspace(0, secs, int(sr * secs), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_logmel_shape():
    mel = logmel(_sine())
    assert mel.ndim == 2 and mel.shape[1] == N_MELS


def test_audio_vqvae_roundtrip():
    a = AudioVQVAE()
    ids = a.encode_ids(_sine())
    assert len(ids) > 0
    assert all(0 <= i < AUDIO_VOCAB_SIZE for i in ids)
    mel = a.decode_mel(ids)
    assert mel.shape[1] == N_MELS


def test_pgq_expansion_grows_rank():
    """Saturation → PGQ expands the busiest sector's LoRA rank in place."""
    from src.model.transformer import RDMCAFoundational, ModelConfig
    from src.model.lora import build_all_sectors
    from src.consolidation.pgq import PGQ
    import mlx.core as mx

    m = RDMCAFoundational(ModelConfig(d_model=64, n_layers=2, n_heads=2,
                                      ffn_dim=128, context_len=32, vocab_size=200,
                                      mrl_dims=[32, 64]))
    mx.eval(m.parameters())
    m.attach_sectors(build_all_sectors(64, 2))
    r0 = m.sectors[3].rank
    pgq = PGQ()
    res = pgq.evaluate("c", saturation=0.6, exc_rate=0.6, pred_error=0.6,
                       cluster_novel=0.6, busiest_sector_id=3,
                       sectors=m.sectors, model=m)
    assert res.decision == "expand"
    assert m.sectors[3].rank > r0
    # forward still works (added components are zero-output at first)
    out = m.logits(mx.array([[1, 2, 3]])); mx.eval(out)
    assert out.shape == (1, 3, 200)


def test_pgq_new_sector_creation():
    """High GNS instantiates a brand-new sector on the model."""
    from src.model.transformer import RDMCAFoundational, ModelConfig
    from src.model.lora import build_all_sectors
    from src.consolidation.pgq import PGQ
    import mlx.core as mx

    m = RDMCAFoundational(ModelConfig(d_model=64, n_layers=2, n_heads=2,
                                      ffn_dim=128, context_len=32, vocab_size=200,
                                      mrl_dims=[32, 64]))
    mx.eval(m.parameters())
    m.attach_sectors(build_all_sectors(64, 2))
    n0 = len(m.sectors)
    res = PGQ().evaluate("c", saturation=1.0, exc_rate=1.0, pred_error=1.0,
                         cluster_novel=1.0, busiest_sector_id=1,
                         sectors=m.sectors, model=m)
    assert res.decision == "new_sector"
    assert len(m.sectors) == n0 + 1
