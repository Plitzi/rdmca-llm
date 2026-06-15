"""Corpus writer + completeness validator (src/data/jsonl_writer.py)."""

import json

from src.data import jsonl_writer as J


def test_estimate_tokens():
    assert J.estimate_tokens("") == 0
    assert J.estimate_tokens("a" * 39) == 10  # ~3.9 chars/token


def test_validate_missing_or_empty(tmp_path):
    ok, why = J.validate_jsonl(tmp_path / "nope.jsonl", 1)
    assert not ok and "empty or missing" in why


def test_validate_prefers_meta_exhausted(tmp_path):
    p = tmp_path / "src.jsonl"
    p.write_text(json.dumps({"text": "x"}) + "\n")
    p.with_suffix(".meta.json").write_text(json.dumps({"tokens": 5_000, "exhausted": True}))
    ok, why = J.validate_jsonl(p, 100)
    assert ok and "exhausted" in why


def test_validate_meta_partial_and_complete(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text(json.dumps({"text": "x"}) + "\n")
    p.with_suffix(".meta.json").write_text(json.dumps({"tokens": 1_000_000}))  # 1M < 90% of 10M
    ok, _ = J.validate_jsonl(p, 10)
    assert not ok
    p.with_suffix(".meta.json").write_text(json.dumps({"tokens": 10_000_000}))  # ≥ 90% of 10M
    ok, _ = J.validate_jsonl(p, 10)
    assert ok


def test_validate_size_heuristic_and_corruption(tmp_path):
    # no meta → first-line + size heuristic
    big = tmp_path / "big.jsonl"
    big.write_text("".join(json.dumps({"text": "word " * 50}) + "\n" for _ in range(2000)))
    ok, _ = J.validate_jsonl(big, 1)  # plenty vs a 1M budget
    assert ok
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json at all\n")
    ok, why = J.validate_jsonl(bad, 1)
    assert not ok and "corrupted" in why
    nokey = tmp_path / "nokey.jsonl"
    nokey.write_text(json.dumps({"nope": 1}) + "\n")
    ok, why = J.validate_jsonl(nokey, 1)
    assert not ok and "wrong format" in why


def test_write_jsonl_exhausted_vs_budget(tmp_path, capsys):
    # source smaller than budget → exhausted True
    recs = ({"text": "some words here to count"} for _ in range(5))
    n, exhausted = J.write_jsonl(recs, tmp_path / "a.jsonl", token_budget_m=999, verbose=True)
    assert exhausted is True and n > 0
    assert (tmp_path / "a.jsonl").exists()
    # tiny budget → stops on budget, exhausted False
    many = ({"text": "word " * 200} for _ in range(100000))
    _, exhausted2 = J.write_jsonl(many, tmp_path / "b.jsonl", token_budget_m=0.001, verbose=False)
    assert exhausted2 is False
