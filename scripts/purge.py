#!/usr/bin/env python3
"""
Purge generated artifacts for a fresh-from-zero training run.

Removes ONLY things the pipeline generates — checkpoints, the trained tokenizer,
prepared training corpora, runtime memory and logs. It never touches your
inputs: configs, `.env`, source code, `data/benchmarks/` (BCF probes you
provide), or the shared HuggingFace download cache.

Data pipeline (two distinct artifacts — don't confuse them):
  HF cache (raw downloads)  →  prepare_data  →  models/<model>/*/data/level* (prepared
  corpora)  →  train
  • --data / --keep-data act on the PREPARED corpora (prepare_data's output).
  • --hf-cache acts on the RAW DOWNLOADS (prepare_data's input). Dropping it forces
    a re-download from the network; keeping the prepared data only skips re-preparing.

Targets (pick any combination, or --all):
  --checkpoints   dist/checkpoints/<model>/  + dist/snapshots/  (trained weights, frozen core, sectors)
  --tokenizer     dist/tokenizer/    + dist/tokenizer*.bak (SentencePiece + image/audio VQ-VAE)
  --data          models/<model>/*/data/level*/  (PREPARED corpora — output of prepare_data)
  --runtime       data/runtime/       (experiences.jsonl, ltss.db — consolidation memory)
  --logs          logs/               (daemon.log, cycle_*.json, human_queue.jsonl)
  --hf-cache      ~/.cache/huggingface/{datasets,hub}  (RAW HF downloads — prepare_data's
                  input; honors HF_HOME / HF_DATASETS_CACHE / HF_HUB_CACHE). Opt-in only —
                  NOT in --all, shared across projects, slow to refill (re-downloads).
  --keep-data     with --all, KEEP the prepared corpora (skip re-preparing). Independent of --hf-cache.

Scope with --level N and/or --model NAME to limit --checkpoints and --data (else all
levels/models). tokenizer/runtime/logs/hf-cache are global and always purged in full.

Tracked `.gitkeep` markers are preserved: a purged folder that has one stays as
an empty, tracked folder (the repo keeps its directory skeleton in git).

Safety: prints exactly what will be deleted (with sizes) and asks for
confirmation. Use --dry-run to only preview, or --yes to skip the prompt.

Examples:
  python scripts/purge.py --all --dry-run            # preview a full wipe
  python scripts/purge.py --all --yes                # full fresh start
  python scripts/purge.py --all --keep-data --yes    # fresh weights/tokenizer, keep prepared data
  python scripts/purge.py --checkpoints --data --level 1   # redo level 1 only
  python scripts/purge.py --tokenizer --checkpoints  # keep prepared data, retrain
  python scripts/purge.py --all --hf-cache --yes     # wipe everything incl. HF cache
  python scripts/purge.py --hf-cache --dry-run       # preview just the HF cache
"""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Repo-internal targets, purged by --all.
TARGET_NAMES = ("checkpoints", "tokenizer", "data", "runtime", "logs")
# Opt-in only (NOT in --all): the HuggingFace download cache is shared across
# projects and slow to refill, so it's never wiped implicitly.
EXTRA_NAMES = ("hf_cache",)


def _hf_cache_paths() -> list[Path]:
    """HuggingFace dataset + hub cache dirs, honoring HF env vars."""
    hf_home = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache/huggingface"))
    datasets = Path(os.environ.get("HF_DATASETS_CACHE") or (hf_home / "datasets"))
    hub = Path(
        os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or (hf_home / "hub")
    )
    return [datasets, hub]


def _paths_for(target: str, level: int | None, model: str | None) -> list[Path]:
    """Resolve a target name to the concrete paths it would remove. Checkpoints and
    prepared corpora are namespaced by MODEL (a package under models/); `model=None`
    spans every model, `level=None` spans every level."""
    lvl = f"level{level}" if level is not None else None
    mdl = model or "*"  # glob across models when unscoped
    if target == "checkpoints":
        if model and lvl:
            return [REPO / "dist/checkpoints" / model / lvl]
        if model:
            return [REPO / "dist/checkpoints" / model]
        if lvl:
            return sorted((REPO / "dist/checkpoints").glob(f"*/{lvl}"))
        return [REPO / "dist/checkpoints", REPO / "dist/snapshots"]
    if target == "tokenizer":  # global (trained per level into one dir)
        return [REPO / "dist/tokenizer", *sorted((REPO / "dist").glob("tokenizer*.bak"))]
    if target == "data":  # prepared corpora, inside each stage package (gitignored)
        glob = f"models/{mdl}/*/data/{lvl}" if lvl else f"models/{mdl}/*/data"
        return sorted(REPO.glob(glob))
    if target == "runtime":
        return [REPO / "data/runtime"]
    if target == "logs":
        return [REPO / "logs"]
    if target == "hf_cache":  # shared, outside the repo
        return _hf_cache_paths()
    return []


def _remove(path: Path) -> None:
    """Delete `path`, but preserve any `.gitkeep` marker (and the directory that
    holds it) so the repo's tracked directory skeleton survives a purge.

    Files are unlinked; a `.gitkeep` is never removed. A directory is emptied
    recursively and then removed only if nothing was kept inside it — so a folder
    containing a `.gitkeep` stays as an empty, tracked folder."""
    if path.is_symlink() or path.is_file():
        if path.name != ".gitkeep":
            try:
                path.unlink()
            except OSError as e:  # read-only file etc. — skip, keep going
                print(f"  [skip] could not remove {_display(path)}: {e}")
        return
    for child in path.iterdir():
        _remove(child)
    with contextlib.suppress(OSError):
        path.rmdir()  # succeeds only if now empty (no .gitkeep kept)
        # OSError → a .gitkeep (or kept subdir) remains → keep the folder


def _display(p: Path) -> str:
    """Repo-relative path when inside the repo, else absolute (e.g. HF cache)."""
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def _size_bytes(path: Path) -> int:
    if path.is_file() or path.is_symlink():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file() and not p.is_symlink():
            with contextlib.suppress(OSError):
                total += p.stat().st_size
    return total


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Purge generated artifacts for a fresh training run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--all", action="store_true", help="Purge every repo target (not the HF cache)")
    for t in TARGET_NAMES:
        ap.add_argument(f"--{t}", action="store_true", help=f"Purge {t}")
    ap.add_argument(
        "--hf-cache",
        dest="hf_cache",
        action="store_true",
        help="Also delete the shared HuggingFace download cache "
        "(datasets + hub). Opt-in only — NOT included in --all; slow to refill.",
    )
    ap.add_argument(
        "--keep-data",
        action="store_true",
        help="With --all, do NOT purge prepared corpora (data/level*) — "
        "re-use the already-prepared data and skip re-preparing.",
    )
    ap.add_argument(
        "--level",
        type=int,
        default=None,
        help="Limit --checkpoints/--data to one level (else all levels)",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Limit --checkpoints/--data to one model (a package under models/, "
        "e.g. cognition); else every model.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Preview only; delete nothing")
    ap.add_argument("--yes", "-y", action="store_true", help="Skip the confirmation prompt")
    args = ap.parse_args()

    selected = [t for t in TARGET_NAMES if args.all or getattr(args, t)]
    if args.keep_data:  # explicit opt-out of data purge
        selected = [t for t in selected if t != "data"]
    if args.hf_cache:  # opt-in, never via --all
        selected.append("hf_cache")
    if not selected:
        ap.error(
            "nothing selected — pass --all, --hf-cache, or one or more of: "
            + ", ".join(f"--{t}" for t in TARGET_NAMES)
        )

    # Resolve and de-duplicate existing paths, remembering which targets are empty.
    plan: list[tuple[str, Path, int]] = []
    seen: set[Path] = set()
    absent: list[str] = []
    for t in selected:
        hits = [p for p in _paths_for(t, args.level, args.model) if p.exists() and p not in seen]
        if not hits:
            absent.append(t)
        for p in hits:
            seen.add(p)
            plan.append((t, p, _size_bytes(p)))

    scope_bits = []
    if args.model is not None:
        scope_bits.append(f"model {args.model}")
    if args.level is not None:
        scope_bits.append(f"level {args.level}")
    scope = f" ({', '.join(scope_bits)})" if scope_bits else ""
    print(f"\nPurge plan{scope}:")
    if not plan:
        print("  Nothing to delete — all selected targets are already absent.")
        return
    total = 0
    for t, p, sz in plan:
        total += sz
        print(f"  [{t:<11}] {_display(p)}  ({_human(sz)})")
    if absent:
        print(f"  (already empty: {', '.join(absent)})")
    print(f"  Total: {_human(total)} across {len(plan)} path(s)")
    kept = "configs/, .env, src/, data/benchmarks/"
    if "hf_cache" not in selected:
        kept += ", HF cache"
    print(f"  Kept (inputs): {kept}.")
    if "hf_cache" in selected:
        print("  ⚠️  HF cache is shared across projects — re-downloads on next prepare_data.")

    if args.dry_run:
        print("\n[dry-run] Nothing deleted.")
        return

    if not args.yes:
        try:
            reply = input("\nType 'yes' to delete the above permanently: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            reply = ""
        if reply != "yes":
            print("Aborted — nothing deleted.")
            return

    removed = 0
    for _, p, _sz in plan:
        try:
            _remove(p)
            removed += 1
            kept = " (kept .gitkeep)" if p.exists() else ""
            print(f"  removed {_display(p)}{kept}")
        except OSError as e:
            print(f"  [error] could not remove {_display(p)}: {e}")
    print(f"\nDone — removed {removed}/{len(plan)} path(s), freed ~{_human(total)}.")
    print("Tracked .gitkeep markers are preserved (empty folders stay).")
    print("Fresh start: rdmca prepare → rdmca tokenizer → rdmca train.")


if __name__ == "__main__":
    main()
