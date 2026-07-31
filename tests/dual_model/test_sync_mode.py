import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_json(relative_path: str):
    return json.loads((REPO_ROOT / relative_path).read_text())


def load_trace(name: str):
    path = REPO_ROOT / f"artifacts/traces/{name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def assert_common_sync_invariants(rows):
    assert rows
    assert all(row["accepted_token"] == row["selected_argmax"] for row in rows)
    assert all(row["small_block_ids"] == row["large_block_ids"] for row in rows)
    assert all(
        len(row["small_block_ids"]) == len(set(row["small_block_ids"]))
        for row in rows
    )
    assert all(row["small_kv_ptr"] != row["large_kv_ptr"] for row in rows)
    assert [(row["small_delta"], row["large_delta"]) for row in rows] == [
        (10, 10),
        *[(1, 1)] * (len(rows) - 1),
    ]


def test_all_small_matches_stock() -> None:
    stock = load_json("artifacts/stock_small.json")
    result = load_json("artifacts/m4-all-small.json")
    rows = load_trace("m4-all-small")
    assert result["output_token_ids"] == stock["output_token_ids"]
    assert all(row["route"] == "small" for row in rows)
    assert all(row["accepted_token"] == row["small_argmax"] for row in rows)
    assert_common_sync_invariants(rows)


def test_all_large_matches_stock() -> None:
    stock = load_json("artifacts/stock_large.json")
    result = load_json("artifacts/m4-all-large.json")
    rows = load_trace("m4-all-large")
    assert result["output_token_ids"] == stock["output_token_ids"]
    assert all(row["route"] == "large" for row in rows)
    assert all(row["accepted_token"] == row["large_argmax"] for row in rows)
    assert_common_sync_invariants(rows)


def test_forced_routes_match_reference() -> None:
    reference = load_json("artifacts/m1-sync.json")
    result = load_json("artifacts/m4-forced.json")
    rows = load_trace("m4-forced")
    assert result["output_token_ids"] == reference["output_token_ids"]
    assert [row["route"] for row in rows] == [
        "small",
        "small",
        "large",
        "small",
        "large",
        "large",
        "small",
    ]
    assert_common_sync_invariants(rows)
