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
  python consolidation_daemon.py --config configs/rdmca_t2.yaml
  python consolidation_daemon.py --config configs/rdmca_t2.yaml --once
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

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


def run_consolidation(cfg: dict) -> None:
    """
    Build the consolidation pipeline and run one cycle.
    TODO Phase 2: wire up all components from src/consolidation/pipeline.py
    """
    logging.info("Consolidation cycle starting …")
    # Placeholder — will be replaced when Phase 2 is implemented
    logging.warning("ConsolidationPipeline not yet wired up (Phase 2 TODO)")
    logging.info("Consolidation cycle finished (no-op)")


def main():
    parser = argparse.ArgumentParser(description="RDMCA Consolidation Daemon")
    parser.add_argument("--config", default="configs/rdmca_t2.yaml")
    parser.add_argument("--once", action="store_true",
                        help="Run one cycle immediately and exit (skip idle detection)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    Path("logs").mkdir(exist_ok=True)
    logging.info(f"Daemon started | config={args.config} | once={args.once}")

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
