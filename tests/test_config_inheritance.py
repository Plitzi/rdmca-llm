"""
Level-config inheritance from configs/levels/_base.yaml.

Levels declare only their diffs; load_config deep-merges them over the shared base. This
guards two things:
  1. REGRESSION — every level still resolves to EXACTLY the config it did before the base
     was introduced (a frozen snapshot), so trimming a level to its diffs changed nothing.
  2. The mechanism — a level that omits a shared block inherits it from the base, and the
     level-0 smoke tier (inherit_base: false) stays standalone.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.config import load_config, level_config_path, _deep_merge

SNAP = json.loads((Path(__file__).parent / "fixtures" /
                   "level_configs_snapshot.json").read_text())


def _strkeys(o):
    """Recursively stringify dict keys so YAML int keys (1) and JSON str keys ('1')
    compare equal — the snapshot is JSON (str keys), load_config yields YAML (int keys)."""
    if isinstance(o, dict):
        return {str(k): _strkeys(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_strkeys(v) for v in o]
    return o


def test_every_level_resolves_to_its_frozen_snapshot():
    """Each level's EFFECTIVE (base-merged) config must match the frozen fixture — a guard
    against ACCIDENTAL drift while trimming/refactoring level yamls. Regenerate the fixture
    (tests/fixtures/level_configs_snapshot.json from load_config) on an INTENTIONAL change."""
    for lvl, expected in SNAP.items():
        got = load_config(level_config_path(int(lvl)))
        assert _strkeys(got) == _strkeys(expected), f"level {lvl} merged config drifted"


def test_levels_inherit_shared_base_blocks():
    """A production level (1) inherits the shared moe/skip_gate/training knobs + the
    curriculum stage structure from the base even if its own yaml omits them."""
    base = yaml.safe_load((Path("configs/levels/_base.yaml")).read_text())
    cfg = load_config(level_config_path(1))
    assert cfg["moe"] == base["moe"]                      # full moe block from base
    assert cfg.get("skip_gate") is False
    # Every stage carries a name + entry_level (from base, possibly overridden).
    for s in range(1, 8):
        st = cfg["curriculum"][f"stage{s}"]
        assert st.get("name") and "entry_level" in st


def test_level0_smoke_tier_does_not_inherit_base():
    """level 0 sets inherit_base: false → it is loaded standalone (no base merge)."""
    raw = yaml.safe_load(Path(level_config_path(0)).read_text())
    raw.pop("inherit_base", None)
    got = load_config(level_config_path(0))
    assert _strkeys(got) == _strkeys(raw)


def test_deep_merge_override_wins_and_dicts_merge():
    base = {"a": 1, "b": {"x": 1, "y": 2}, "l": [1, 2]}
    over = {"b": {"y": 9, "z": 3}, "l": [9]}
    out = _deep_merge(base, over)
    assert out == {"a": 1, "b": {"x": 1, "y": 9, "z": 3}, "l": [9]}   # nested merge, list replace
