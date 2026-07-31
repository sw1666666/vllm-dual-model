import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_json(relative_path: str):
    return json.loads((REPO_ROOT / relative_path).read_text())


def load_trace(name: str):
    path = REPO_ROOT / f"artifacts/traces/{name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def assert_lazy_invariants(rows):
    assert rows
    for row in rows:
        assert row["accepted_token"] == row["selected_argmax"]
        assert row["small_after"] == row["history_len_before"]
        assert 0 <= row["large_after"] <= row["small_after"]
        if row["route"] == "large":
            assert row["large_argmax"] is not None
            assert row["large_after"] == row["history_len_before"]
            assert row["large_delta"] == row["history_len_before"] - row["large_before"]
        else:
            assert row["large_argmax"] is None
            assert row["large_delta"] == 0
            assert row["large_after"] == row["large_before"]


def test_all_small_matches_stock_and_skips_large() -> None:
    result = load_json("artifacts/m5-all-small.json")
    stock = load_json("artifacts/stock_small.json")
    rows = load_trace("m5-all-small")
    assert result["output_token_ids"] == stock["output_token_ids"]
    assert all(row["route"] == "small" for row in rows)
    assert all(row["large_block_ids"] == [] for row in rows)
    assert_lazy_invariants(rows)


def test_all_large_matches_stock_and_advances_one_at_a_time() -> None:
    result = load_json("artifacts/m5-all-large.json")
    stock = load_json("artifacts/stock_large.json")
    rows = load_trace("m5-all-large")
    assert result["output_token_ids"] == stock["output_token_ids"]
    assert [row["large_delta"] for row in rows] == [10, *([1] * 15)]
    assert_lazy_invariants(rows)


def test_forced_routes_lazy_matches_sync() -> None:
    lazy = load_json("artifacts/m5-forced.json")
    sync = load_json("artifacts/m4-forced.json")
    rows = load_trace("m5-forced")
    sync_rows = load_trace("m4-forced")
    assert lazy["output_token_ids"] == sync["output_token_ids"]
    assert [row["route"] for row in rows] == [
        "small", "small", "large", "small", "large", "large", "small"
    ]
    assert [row["large_delta"] for row in rows] == [0, 0, 12, 0, 2, 1, 0]
    assert [row["large_argmax"] for row in rows if row["route"] == "large"] == [
        row["large_argmax"] for row in sync_rows if row["route"] == "large"
    ]
    assert_lazy_invariants(rows)


def test_late_first_large_catches_up_full_history() -> None:
    rows = load_trace("m5-late-first-large")
    assert len(rows) == 16
    assert all(row["route"] == "small" for row in rows[:15])
    assert all(row["large_after"] == 0 for row in rows[:15])
    last = rows[-1]
    assert last["route"] == "large"
    assert last["history_len_before"] == 25
    assert last["large_before"] == 0
    assert last["large_delta"] == 25
    assert last["large_after"] == 25
    assert last["large_block_ids"] == last["small_block_ids"] == [1, 2]
    assert_lazy_invariants(rows)
