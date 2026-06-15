#!/usr/bin/env python3
"""
RDMCA Data Preparation Script — config-driven, per level + stage
================================================================
Writes the training corpus for each curriculum stage of a LEVEL, using the
sources, complexity filter and token budget declared in that level's config
(`configs/levels/levelN.yaml`). Output:
  models/<model>/stage{N}_<slug>/data/level{L}/{source}.jsonl   {"text": "...", "lang": "<code>"}
plus a {source}.meta.json sidecar recording token count + whether the source
was exhausted (used to decide if a re-run can skip it).

Where the data comes from is per-source:
  - Lower levels use each stage plugin's OWN simple/graded sources (tinystories,
    dialogue, arithmetic, analogies, agentic/MCP/skills, reasoning) — small,
    conversational/structured (see models/<model>/stageNN_*/sources.py).
  - Higher levels add the FULL external corpora (src/core/data/corpora.py: Wikipedia per
    language, ARC, GSM8K, MATH, ethics), with Wikipedia routed to a stage by category
    keywords (STAGE_KEYWORDS) and prose readability-graded (Flesch-Kincaid).

Token budgets come from each stage's `n_tokens` in the config (NOT hardcoded
here), split across the stage's sources.

Usage:
  python scripts/prepare_data.py --level 1 --stage all
  python scripts/prepare_data.py --level 1 --stage 1
  python scripts/prepare_data.py --level 1 --stage 1 --limit 100  # 100MB test
  python scripts/prepare_data.py --level 1 --stage 1 --lang en    # English only
"""

# Re-exec into the project venv BEFORE importing third-party deps, so the script
# works when launched with a bare `python` outside the venv. (Kept after the
# docstring so `__doc__` / argparse epilog / pydoc see the help text.)
from __future__ import annotations

import os
import sys

_venv = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv", "bin", "python"
)
if os.path.exists(_venv) and os.path.abspath(sys.executable) != os.path.abspath(_venv):
    os.execv(_venv, [_venv, *sys.argv])

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.data.corpora import full_corpus_streamers, setup_hf_token
from src.core.data.jsonl_writer import validate_jsonl, write_jsonl
from src.core.data.textnorm import conversational_quality_ok  # ingestion content gate

# Sources that are free prose and SHOULD be readability-graded. Everything else
# (dialogue, tool/skill/MCP JSON, arithmetic, analogies, causal) is conversational
# or structured and must NOT be grade-filtered — see the gate below.
_READABILITY_FILTERED = {"tinystories", "simple_wikipedia", "wikipedia"}
# Turn-structured conversational sources get the CONTENT quality gate
# (conversational_quality_ok): keep clean, short exchanges; drop long technical /
# monologue / code-dump records a tiny L1 base can't learn to answer. This is the
# lever for the #1 goal — a model that understands and replies — and works for any
# future conversational provider, not just today's.
_CONVERSATIONAL_FILTERED = {"dialogue", "instruct", "basic_chat", "smalltalk"}


def prepare_stage_for_level(
    level: int, stage: int, cfg: dict, langs: list[str], limit_mb: int | None = None
) -> None:
    """Prepare graded data for one (level, stage), reading the level config's
    curriculum entry: which sources, the complexity filter, the token budget and
    the output dir. Skips stages whose entry_level is above this level."""
    from src.core.training.curriculum import stage_enabled
    from src.plugins import stage_data_dir, stream_source
    from src.plugins.sdk import passes_filter

    curriculum = cfg.get("curriculum", {}) or {}
    stage_key = f"stage{stage}"
    if stage_key not in curriculum:
        print(f"  Stage {stage}: not active at level {level} — skipping.")
        return
    if not stage_enabled(stage, cfg):
        print(f"  Stage {stage}: disabled (plugin/config) — skipping.")
        return
    stage_cfg = curriculum[stage_key]
    entry = int(stage_cfg.get("entry_level", 1))
    if entry > level:
        print(f"  Stage {stage}: enters at level {entry} (> {level}) — skipping.")
        return

    data = stage_cfg.get("data", {}) or {}
    sources = data.get("sources", []) or []
    flt = data.get("filter")  # None at level 5
    arith = (flt or {}).get("arithmetic_level", level) if isinstance(flt, dict) else level

    out_dir = Path(stage_data_dir(stage, cfg))
    out_dir.mkdir(parents=True, exist_ok=True)

    budget_m = max(int(stage_cfg.get("n_tokens", 100_000_000) // 1_000_000), 1)
    default_per = max(budget_m // max(len(sources), 1), 1)
    # Optional PER-SOURCE token budgets (M tokens), so a small CLEAN curated source
    # (basic_chat, definitions) gets a controlled, meaningful share and a large noisy
    # one (tinystories) can be capped — instead of the naive uniform split that
    # drowns curated sources. Falls back to the uniform default for unlisted sources.
    src_budgets = data.get("source_budgets_m") or {}
    extra = full_corpus_streamers(stage, langs, limit_mb)

    print(f"\n{'=' * 60}")
    print(f"Level {level} · Stage {stage}: {stage_cfg.get('name', '')}")
    print(f"  sources={sources}  filter={flt}  budget~{budget_m}M tokens → {out_dir}/")
    print(f"{'=' * 60}")

    for source in sources:
        per_source_m = max(int(src_budgets.get(source, default_per)), 1)
        out_path = out_dir / f"{source}.jsonl"
        ok, reason = validate_jsonl(out_path, per_source_m)
        if ok:
            print(f"  OK (valid): {out_path.name}  —  {reason}")
            continue
        it = stream_source(
            source,
            langs=langs,
            n_tokens=per_source_m * 1_000_000,
            arithmetic_level=arith,
            limit_mb=limit_mb,
            extra_streamers=extra,
        )
        if it is None:
            print(f"  [skip] unknown source '{source}'")
            continue
        # Readability (Flesch-Kincaid grade) gating only makes sense for free
        # PROSE. Applying grade≤2 to real human dialogue decimated it (it nearly
        # always scores above a preschool grade), starving the model of the
        # conversational `User:/Assistant:` format — so only prose sources are
        # graded; conversational/structured ones pass through.
        if flt and source in _READABILITY_FILTERED:  # None / non-prose ⇒ keep all
            it = (rec for rec in it if passes_filter(rec.get("text", ""), flt))
        elif source in _CONVERSATIONAL_FILTERED:  # content quality for conversation
            it = (rec for rec in it if conversational_quality_ok(rec.get("text", "")))
        tokens, exhausted = write_jsonl(it, out_path, per_source_m)
        # Sidecar: records completeness so a re-run can tell "source exhausted"
        # (complete, smaller than budget) from "download interrupted" (partial).
        out_path.with_suffix(".meta.json").write_text(
            json.dumps({"tokens": tokens, "budget_m": per_source_m, "exhausted": exhausted})
        )
        note = "exhausted" if exhausted else "budget reached"
        print(f"  {source}: {tokens / 1e6:.1f}M tokens ({note}) → {out_path.name}")

    print(f"  Stage {stage} ready in {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="RDMCA curriculum data preparation")
    parser.add_argument(
        "--level",
        type=int,
        default=None,
        help="Educational level 1-5 (preescolar..universidad). "
        "Determines the graded data sources + complexity.",
    )
    parser.add_argument("--stage", default="all", help="Stage number (1-5) or 'all'")
    parser.add_argument("--config", default=None, help="Explicit config path (overrides --level)")
    parser.add_argument(
        "--model",
        default=None,
        help="Model whose stages to prepare (package under models/, e.g. cognition). "
        "Overrides the config's model_name; defaults to cognition.",
    )
    parser.add_argument("--lang", default=None, help="Comma-separated override of config languages")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit each Wikipedia stream to N MB (testing)"
    )
    args = parser.parse_args()

    setup_hf_token()

    from src.core.config import (
        MAX_LEVEL,
        get_languages,
        get_level,
        load_config,
        resolve_config_path,
    )

    cfg_path = resolve_config_path(args.config, args.level)
    cfg = load_config(cfg_path)
    # Select the active model (CLI --model wins; registry default = cognition) before
    # touching the stage registry, so it discovers THIS model's stages and data dirs.
    from src.core.config import select_model

    select_model(cfg, args.model)
    level = get_level(cfg)  # NB: level 0 is valid → use `is None`
    if level is None:  # custom config w/o a level → least filtering
        level = args.level if args.level is not None else MAX_LEVEL
    # Languages: --lang override > config(model.languages) > ['en']
    langs = [l.strip() for l in args.lang.split(",")] if args.lang else get_languages(cfg)
    # "all" → every stage declared in this level's curriculum (data-driven, so
    # new stages like agentic/MCP are picked up automatically).
    if args.stage == "all":
        stages = sorted(int(k.replace("stage", "")) for k in cfg.get("curriculum", {}))
    else:
        stages = [int(args.stage)]

    print(f"Level {level} ({cfg.get('name', 'custom')}) | languages: {langs} | config: {cfg_path}")
    _NETWORK_ERRORS = (
        "RemoteProtocolError",
        "ConnectError",
        "ReadTimeout",
        "ConnectionError",
        "ServerDisconnected",
        "TimeoutError",
    )

    try:
        for s in stages:
            prepare_stage_for_level(level, s, cfg, langs=langs, limit_mb=args.limit)
    except KeyboardInterrupt:
        print("\n\nInterrupted. Run the same command again to resume.")
        sys.exit(0)
    except Exception as e:
        if type(e).__name__ in _NETWORK_ERRORS or "disconnected" in str(e).lower():
            print(f"\nNetwork error: {e}")
            print("Run the same command again to resume — files already written are kept.")
            sys.exit(1)
        raise  # anything else: show full traceback

    print(f"\nDone. Next: python scripts/train_tokenizer.py --level {level}")
    sys.stdout.flush()
    sys.stderr.flush()
    # The HuggingFace datasets streaming iterators leave multiprocessing
    # SemLock objects dangling when a stream is closed early (e.g. on the MB
    # limit). Force a GC pass so their finalizers run and unregister from the
    # resource_tracker — otherwise the forced os._exit() below skips that
    # cleanup and the tracker prints a spurious "leaked semaphore" warning.
    import gc

    gc.collect()
    # os._exit() below skips normal cleanup, so any SemLock still registered
    # with the resource_tracker triggers a spurious "leaked semaphore" warning
    # when the tracker process detects our pipe closing. The tracker's resource
    # registry lives in *its* subprocess (not reachable from here), so the only
    # way to stop the warning is to kill that subprocess before it runs its
    # end-of-life check. The OS reclaims the leftover semaphores on exit anyway.
    try:
        import signal
        from multiprocessing import resource_tracker

        pid = getattr(resource_tracker._resource_tracker, "_pid", None)
        if pid is not None:
            os.kill(pid, signal.SIGKILL)
    except Exception:
        pass
    # Force exit — the HuggingFace datasets library leaves background
    # threads running after streaming ends, which blocks normal exit.
    os._exit(0)


if __name__ == "__main__":
    main()
