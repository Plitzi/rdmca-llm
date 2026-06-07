#!/usr/bin/env python3
import sys, os
try:
    import mlx.core  # noqa: F401
except ModuleNotFoundError:
    venv_py = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           ".venv", "bin", "python")
    if os.path.exists(venv_py) and os.path.abspath(sys.executable) != os.path.abspath(venv_py):
        os.execv(venv_py, [venv_py] + sys.argv)
    print("ERROR: mlx not found. Run: source .venv/bin/activate")
    sys.exit(1)

"""
Consolidation Daemon — Phase 2+
Runs during system idle time (CPU < 20% for 5+ minutes).
Executes the full consolidation pipeline on accumulated experiences.

Usage:
  python consolidation_daemon.py --profile m2max
  python consolidation_daemon.py --profile m2max --once
"""
import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/daemon.log"),
    ],
)

IDLE_CPU_THRESHOLD = 20.0    # %
IDLE_DURATION_SECS = 300     # 5 minutes of continuous idle
POLL_INTERVAL_SECS = 60


def cpu_percent() -> float:
    """Current system CPU usage percentage."""
    try:
        import psutil
        return psutil.cpu_percent(interval=1)
    except ImportError:
        return 0.0   # if psutil not available, assume idle


def wait_for_idle() -> None:
    """Block until system has been idle for IDLE_DURATION_SECS."""
    idle_since = None
    while True:
        cpu = cpu_percent()
        if cpu < IDLE_CPU_THRESHOLD:
            if idle_since is None:
                idle_since = time.time()
                logging.info(f"System idle (CPU {cpu:.1f}%) — waiting {IDLE_DURATION_SECS}s")
            elif time.time() - idle_since >= IDLE_DURATION_SECS:
                logging.info("Idle threshold reached — starting consolidation")
                return
        else:
            if idle_since is not None:
                logging.info(f"CPU spike {cpu:.1f}% — resetting idle timer")
            idle_since = None
        time.sleep(POLL_INTERVAL_SECS)


def _ckpt_root(cfg: dict) -> Path:
    profile = cfg.get("profile")
    return Path("dist/checkpoints") / profile if profile else Path("dist/checkpoints")


def _build_model(cfg: dict):
    """Load the frozen foundational core + attach (and reload) LoRA sectors."""
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    from src.model.transformer import RDMCAFoundational, ModelConfig
    from src.model.lora import build_all_sectors
    from src.config import load_tokenizer_info, unified_vocab_size

    mcfg = dict(cfg["model"])
    info = load_tokenizer_info()
    mcfg["vocab_size"] = unified_vocab_size(info, mcfg.get("vocab_size", 65536))
    model_cfg = ModelConfig(**{k: v for k, v in mcfg.items()
                               if k in ModelConfig.__dataclass_fields__})
    model = RDMCAFoundational(model_cfg)

    root = _ckpt_root(cfg)
    frozen = root / "foundational" / "theta_f_frozen.npz"
    if not frozen.exists():
        logging.error(f"No frozen core at {frozen}. Train through Stage 5 first.")
        return None, None
    model.load_weights(list(mx.load(str(frozen)).items()))

    sectors = build_all_sectors(model_cfg.d_model, model_cfg.n_layers)
    model.attach_sectors(sectors)
    sec_path = root / "sectors.npz"
    if sec_path.exists():
        saved = dict(mx.load(str(sec_path)))
        for sid, adapter in sectors.items():
            flat = {k.split("/", 1)[1]: v for k, v in saved.items()
                    if k.startswith(f"S{sid}/")}
            if flat:
                adapter.update(tree_unflatten(list(flat.items())))
    mx.eval(model.parameters())
    return model, root


def _save_sectors(model, root: Path) -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten
    flat = {}
    for sid, adapter in (model.sectors or {}).items():
        for k, v in tree_flatten(adapter.parameters()):
            flat[f"S{sid}/{k}"] = v
    if flat:
        mx.savez(str(root / "sectors.npz"), **flat)


def run_consolidation(cfg: dict) -> None:
    """Build the full pipeline from the experience queue and run one cycle."""
    import numpy as np
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    from src.memory.episodic_buffer import EpisodicBuffer, Experience
    from src.memory.experience_log import load_experiences, clear_experiences
    from src.memory.ltss import LTSS
    from src.relevance.engine import RelevanceEngine
    from src.model.bcf import BCFHead, _hidden_states
    from src.modalities.text import TextTokenizer
    from src.consolidation.snapshot import SectorSnapshotManager
    from src.consolidation.ambiguity import AmbiguityHandler
    from src.consolidation.pgq import PGQ
    from src.consolidation.pipeline import ConsolidationPipeline
    from src.routing.semantic_router import SemanticTokenRouter
    from src.routing.sector_router import SectorRouter

    logging.info("Consolidation cycle starting …")
    records = load_experiences()
    if not records:
        logging.info("No experiences queued — nothing to consolidate.")
        return

    model, root = _build_model(cfg)
    if model is None:
        return
    tokenizer = TextTokenizer()
    d_model   = model.cfg.d_model

    # Embed each experience with the frozen core's final hidden state.
    texts = [r.get("text", "") for r in records]
    embs  = np.array(_hidden_states(model, tokenizer, texts))   # [N, d_model]

    buffer = EpisodicBuffer(max_size=max(len(records), 1000))
    for r, e in zip(records, embs):
        buffer.add(Experience(text=r.get("text", ""),
                              embedding=e.astype(np.float32),
                              modality=r.get("modality", "text")))

    ltss = LTSS(emb_dim=d_model)
    re = RelevanceEngine(ltss=ltss)
    re.update_state(embs.mean(axis=0))

    bcf = BCFHead(d_model)
    bcf_path = root / "stage5" / "bcf_head.npz"
    if bcf_path.exists():
        bcf.update(tree_unflatten(list(mx.load(str(bcf_path)).items())))

    pipeline = ConsolidationPipeline(
        buffer=buffer, ltss=ltss, re=re, bcf=bcf, sectors=model.sectors,
        snapshot_mgr=SectorSnapshotManager(), ambiguity=AmbiguityHandler(),
        pgq=PGQ(), model=model, tokenizer=tokenizer,
        semantic_router=SemanticTokenRouter(d_model), sector_router=SectorRouter(),
    )
    entry = pipeline.run()
    _save_sectors(model, root)
    clear_experiences()
    logging.info(f"Consolidation done | sectors_updated={entry.sectors_updated} | "
                 f"promoted_to_ltss={len(ltss)} | health={entry.health_score:.2f}")


def main():
    parser = argparse.ArgumentParser(description="RDMCA Consolidation Daemon")
    parser.add_argument("--config", default=None)
    parser.add_argument("--profile", default=None,
                        help="Hardware profile: nano | m2max | test | …")
    parser.add_argument("--once", action="store_true",
                        help="Run one cycle immediately and exit (skip idle detection)")
    args = parser.parse_args()

    from src.config import resolve_config_path, load_config
    config_path = resolve_config_path(args.config, args.profile)
    cfg = load_config(config_path)

    Path("logs").mkdir(exist_ok=True)
    logging.info(f"Daemon started | config={config_path} | once={args.once}")

    if args.once:
        run_consolidation(cfg)
        return

    while True:
        wait_for_idle()
        try:
            run_consolidation(cfg)
        except Exception as e:
            logging.exception(f"Consolidation error: {e}")
        time.sleep(POLL_INTERVAL_SECS)


if __name__ == "__main__":
    main()
