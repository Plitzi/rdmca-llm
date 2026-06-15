"""
Stage registry parity tests — lock the plugin metadata to the values the curriculum
used to hardcode in scattered dicts and level configs, so the plugin migration can't
silently change curriculum behavior.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import plugins as stages
from src.plugins import StageKind

# The known-good values before the refactor (from the old stages.py + _base.yaml).
EXPECTED_NAMES = {
    1: "Language and communication",
    2: "Perception and pattern recognition",
    3: "Abstraction and symbolic composition",
    4: "Causal and procedural reasoning",
    5: "Reasoning",
    6: "Memory management",
    7: "Cognitive ethics and BCF",
    8: "Action and tool use",
    9: "Model Context Protocol (MCP)",
    10: "Skills",
}
EXPECTED_GATES = {
    1: ("blim_accuracy", 0.70),
    2: ("arc_easy_accuracy", 0.60),
    3: ("gsm8k_accuracy", 0.15),
    4: ("causal_accuracy", 0.65),
    5: ("reasoning_accuracy", 0.20),
    6: ("memory_accuracy", 0.50),
    7: ("bcf_accuracy", 0.90),
}
EXPECTED_REHEARSAL = {1: 0.15, 2: 0.35, 3: 0.45, 4: 0.35, 5: 0.45, 6: 0.35, 7: 0.35}
EXPECTED_LR_SCALE = {1: 1.0, 2: 0.7, 3: 0.5, 4: 0.7, 5: 0.5, 6: 0.7, 7: 0.7}
EXPECTED_ENTRY_LEVEL = {1: 1, 2: 1, 3: 1, 4: 1, 5: 0, 6: 1, 7: 1, 8: 0, 9: 0, 10: 0}


def test_all_ten_stages_discovered():
    numbers = [p.number for p in stages.all_stages()]
    assert numbers == list(range(1, 11))


def test_names_match():
    assert {p.number: p.name for p in stages.all_stages()} == EXPECTED_NAMES


def test_gates_match():
    for n, (metric, threshold) in EXPECTED_GATES.items():
        gate = stages.get_stage(n).gate
        assert gate is not None and gate.metric_key == metric
        assert gate.threshold == threshold
    # behavioral stages carry no gate
    for n in (8, 9, 10):
        assert stages.get_stage(n).gate is None


def test_rehearsal_and_lr_match():
    for n, reh in EXPECTED_REHEARSAL.items():
        assert stages.get_stage(n).rehearsal_fraction == reh
    for n, lr in EXPECTED_LR_SCALE.items():
        assert stages.get_stage(n).lr_scale == lr


def test_entry_levels_match():
    assert {p.number: p.entry_level for p in stages.all_stages()} == EXPECTED_ENTRY_LEVEL


def test_freeze_point_and_kinds():
    assert stages.bcf_stage() == 7
    for n in range(1, 8):
        # Each stage DECLARES base membership via frozen_base; kind/behavioral derive.
        assert stages.get_stage(n).frozen_base is True
        assert stages.get_stage(n).kind is StageKind.COGNITIVE
        assert not stages.is_behavioral(n)
    for n in (8, 9, 10):
        assert stages.get_stage(n).frozen_base is False
        assert stages.get_stage(n).kind is StageKind.BEHAVIORAL
        assert stages.is_behavioral(n)


def test_active_stages_respects_entry_level():
    # entry_level is the FIRST level a stage becomes active (active when entry_level
    # <= level). Every stage's entry_level is 0 or 1, so all are active at level 1.
    assert {p.number for p in stages.active_stages(1)} == set(range(1, 11))
    # a hypothetical level 0 sees only the entry_level-0 stages.
    assert {p.number for p in stages.active_stages(0)} == {5, 8, 9, 10}
