#!/usr/bin/env python3
import os
import sys

try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    venv_py = os.path.join(_repo, ".venv", "bin", "python")
    if os.path.exists(venv_py) and os.path.abspath(sys.executable) != os.path.abspath(venv_py):
        os.execv(venv_py, [venv_py, *sys.argv])
    print("ERROR: dependencies not found. Run: source .venv/bin/activate")
    sys.exit(1)

"""
Consolidation Daemon — Phase 2+
Runs during system idle time (CPU < 20% for 5+ minutes).
Executes the full consolidation pipeline on accumulated experiences.

An internal runtime component (not a developer build/inspect utility), so it lives in
the consolidation subsystem and is launched as a module:

Usage:
  python -m src.core.consolidation.daemon --level 5
  python -m src.core.consolidation.daemon --level 5 --once
"""
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/daemon.log"),
    ],
)

IDLE_CPU_THRESHOLD = 20.0  # %
IDLE_DURATION_SECS = 300  # 5 minutes of continuous idle
POLL_INTERVAL_SECS = 60


def cpu_percent() -> float:
    """Current system CPU usage percentage."""
    try:
        import psutil

        return psutil.cpu_percent(interval=1)
    except ImportError:
        return 0.0  # if psutil not available, assume idle


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


def _build_model(cfg: dict):
    """Load the frozen foundational core + attach (and reload) LoRA sectors."""
    import numpy as np

    import src.core.backend as backend
    from src.core.config import get_precision, load_tokenizer_info, unified_vocab_size
    from src.core.model.config import ModelConfig
    from src.core.model.lora import build_all_sectors
    from src.core.model.transformer import RDMCAFoundational, set_model_precision

    B = backend.current()
    mcfg = dict(cfg["model"])
    info = load_tokenizer_info()
    mcfg["vocab_size"] = unified_vocab_size(info, mcfg.get("vocab_size", 65536))
    model_cfg = ModelConfig(
        **{k: v for k, v in mcfg.items() if k in ModelConfig.__dataclass_fields__}
    )
    model = RDMCAFoundational(model_cfg)

    from src.core.training.curriculum import ckpt_root

    root = ckpt_root(cfg)
    frozen = root / "foundational" / "theta_f_frozen.npz"
    if not frozen.exists():
        logging.error(f"No frozen core at {frozen}. Train through Stage 5 first.")
        return None, None
    B.engine.load_weights(model, str(frozen))

    sectors = build_all_sectors(model_cfg.d_model, model_cfg.n_layers)
    moe_cfg = cfg.get("moe") or {}
    model.attach_sectors(
        sectors,
        moe=bool(moe_cfg.get("enabled", True)),
        top_k=int(moe_cfg.get("top_k", 2)),
        capacity_factor=float(moe_cfg.get("capacity_factor", 1.25)),
    )
    set_model_precision(model, get_precision(cfg))  # move sectors + gate to device/dtype
    sec_path = root / "sectors.npz"
    if sec_path.exists():
        saved = dict(np.load(str(sec_path)))
        for sid, adapter in sectors.items():
            flat = {k.split("/", 1)[1]: v for k, v in saved.items() if k.startswith(f"S{sid}/")}
            if flat:
                B.engine.load_state_dict(adapter, flat)
        gate_flat = {k.split("/", 1)[1]: v for k, v in saved.items() if k.startswith("GATE/")}
        if gate_flat and model.gate is not None:
            B.engine.load_state_dict(model.gate, gate_flat)
    set_model_precision(model, get_precision(cfg))
    return model, root


def _save_sectors(model, root: Path) -> None:
    import numpy as np

    import src.core.backend as backend

    B = backend.current()
    flat = {}
    for sid, adapter in (model.sectors or {}).items():
        for k, v in B.engine.state_dict(adapter).items():
            flat[f"S{sid}/{k}"] = v
    if getattr(model, "gate", None) is not None:  # persist the MoE router too
        for k, v in B.engine.state_dict(model.gate).items():
            flat[f"GATE/{k}"] = v
    if flat:
        np.savez(str(root / "sectors.npz"), **flat)


def run_consolidation(cfg: dict) -> None:
    """Build the full pipeline from the experience queue and run one cycle."""
    import numpy as np

    import src.core.backend as backend
    from src.core.config import get_precision
    from src.core.consolidation.ambiguity import AmbiguityHandler
    from src.core.consolidation.pgq import PGQ
    from src.core.consolidation.pipeline import ConsolidationPipeline
    from src.core.consolidation.snapshot import SectorSnapshotManager
    from src.core.memory.episodic_buffer import EpisodicBuffer, Experience
    from src.core.memory.experience_log import clear_experiences, load_experiences
    from src.core.memory.ltss import LTSS
    from src.core.modalities.text import TextTokenizer
    from src.core.model.bcf import BCFHead, _hidden_states
    from src.core.relevance.engine import RelevanceEngine
    from src.core.routing.sector_router import SectorRouter
    from src.core.routing.semantic_router import SemanticTokenRouter

    logging.info("Consolidation cycle starting …")
    records = load_experiences()
    if not records:
        logging.info("No experiences queued — nothing to consolidate.")
        return

    model, root = _build_model(cfg)
    if model is None:
        return
    B = backend.current()
    tokenizer = TextTokenizer()
    d_model = model.cfg.d_model

    # Embed each experience with the frozen core's final hidden state.
    texts = [r.get("text", "") for r in records]
    embs = B.ops.to_numpy(_hidden_states(model, tokenizer, texts))  # [N, d_model]

    buffer = EpisodicBuffer(max_size=max(len(records), 1000))
    for r, e in zip(records, embs, strict=False):
        buffer.add(
            Experience(
                text=r.get("text", ""),
                embedding=e.astype(np.float32),
                modality=r.get("modality", "text"),
                feedback=r.get("feedback", "neutral"),
            )
        )

    ltss = LTSS(emb_dim=d_model)
    re = RelevanceEngine(ltss=ltss)
    re.update_state(embs.mean(axis=0))

    bcf = BCFHead(d_model)
    B.engine.set_precision(bcf, get_precision(cfg))  # match model device/dtype
    # The BCF head is saved at the dir of the stage that froze the core — the LAST
    # ACTIVE cognitive stage (`last_cognitive_stage`, = BCF_STAGE/ethics when present,
    # else an earlier stage). The old hardcoded "stage5" loaded nothing on most
    # configs, leaving a RANDOM head that filtered experiences as noise.
    from src.core.training.curriculum import last_cognitive_stage

    freeze_stage = last_cognitive_stage(cfg)
    bcf_path = (root / f"stage{freeze_stage}" / "bcf_head.npz") if freeze_stage else None
    if bcf_path and bcf_path.exists():
        B.engine.load_weights(bcf, str(bcf_path))
        B.engine.set_precision(bcf, get_precision(cfg))
    else:
        logging.warning(
            f"BCF head not found ({bcf_path}) — using an untrained head; "
            "the BCF experience filter will be unreliable."
        )

    from src.core.consolidation.validation import default_validator

    ambiguity = AmbiguityHandler()
    pipeline = ConsolidationPipeline(
        buffer=buffer,
        ltss=ltss,
        re=re,
        bcf=bcf,
        sectors=model.sectors,
        snapshot_mgr=SectorSnapshotManager(),
        ambiguity=ambiguity,
        pgq=PGQ(),
        model=model,
        tokenizer=tokenizer,
        semantic_router=SemanticTokenRouter(d_model),
        sector_router=SectorRouter(),
        # Confidence-gated validation: self-approve when consistent with prior
        # knowledge, else escalate to the human queue. The peer-model / web-research
        # channels stay inert until a client/tool is configured (default_validator).
        validator=default_validator(ambiguity_handler=ambiguity),
    )
    entry = pipeline.run()
    _save_sectors(model, root)
    clear_experiences()
    logging.info(
        f"Consolidation done | sectors_updated={entry.sectors_updated} | "
        f"promoted_to_ltss={len(ltss)} | health={entry.health_score:.2f}"
    )


def main():
    parser = argparse.ArgumentParser(description="RDMCA Consolidation Daemon")
    parser.add_argument("--config", default=None, help="Explicit config path (overrides --level)")
    parser.add_argument(
        "--level",
        type=int,
        default=None,
        help="Educational level 1-5 (which frozen base to consolidate on)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one cycle immediately and exit (skip idle detection)",
    )
    args = parser.parse_args()

    from src.core.config import load_config, require_backend, resolve_config_path

    config_path = resolve_config_path(args.config, args.level)
    cfg = load_config(config_path)
    # Select the active model so checkpoint paths resolve under the right model.
    from src.plugins import set_active_model

    set_active_model(cfg.get("model_name"))
    require_backend(cfg)  # selects the configured backend (mlx | torch)

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
