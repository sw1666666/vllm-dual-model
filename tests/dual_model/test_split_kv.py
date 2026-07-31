"""Acceptance tests for the M6 split KV-cache pools."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
TRACES = ARTIFACTS / "traces"


def _json(name: str) -> dict:
    return json.loads((ARTIFACTS / f"{name}.json").read_text())


def _trace(name: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (TRACES / f"{name}.jsonl").read_text().splitlines()
    ]


def _assert_common(rows: list[dict]) -> None:
    assert rows
    for row in rows:
        assert row["small_allocator"] == "small"
        assert row["large_allocator"] == "large"
        assert row["small_total_blocks"] == 292
        assert row["large_total_blocks"] == 910
        assert 0 <= row["small_used_blocks"] <= row["small_total_blocks"]
        assert 0 <= row["large_used_blocks"] <= row["large_total_blocks"]
        assert row["small_kv_ptr"] != row["large_kv_ptr"]
        assert row["accepted_token"] == row["selected_argmax"]


def test_all_small_uses_no_large_blocks_and_matches_stock() -> None:
    result = _json("m6-all-small")
    rows = _trace("m6-all-small")
    _assert_common(rows)

    assert result["output_token_ids"] == _json("stock_small")["output_token_ids"]
    assert result["kv_cache_memory_bytes"] == (512 << 20) + (2 << 30)
    assert result["small_kv_cache_bytes"] == 512 << 20
    assert result["large_kv_cache_bytes"] == 2 << 30
    assert all(row["route"] == "small" for row in rows)
    assert all(row["large_used_blocks"] == 0 for row in rows)
    assert all(row["large_block_ids"] == [] for row in rows)


def test_all_large_allocates_its_own_pool_and_matches_stock() -> None:
    result = _json("m6-all-large")
    rows = _trace("m6-all-large")
    _assert_common(rows)

    assert result["output_token_ids"] == _json("stock_large")["output_token_ids"]
    assert all(row["route"] == "large" for row in rows)
    assert rows[0]["large_delta"] == rows[0]["history_len_before"]
    assert all(row["large_delta"] == 1 for row in rows[1:])
    assert all(row["large_used_blocks"] > 0 for row in rows)
    assert all(row["large_block_ids"] for row in rows)


def test_forced_lazy_split_pool_matches_shared_pool_and_sync() -> None:
    result = _json("m6-forced")
    rows = _trace("m6-forced")
    _assert_common(rows)

    expected_routes = ["small", "small", "large", "small", "large", "large", "small"]
    assert [row["route"] for row in rows] == expected_routes
    assert result["output_token_ids"] == _json("m5-forced")["output_token_ids"]
    assert result["output_token_ids"] == _json("m4-forced")["output_token_ids"]
    assert rows[0]["large_block_ids"] == []
    assert rows[1]["large_block_ids"] == []
    assert rows[2]["large_delta"] == rows[2]["history_len_before"]
    assert rows[2]["large_input_token_ids"]
    assert all(row["large_block_ids"] for row in rows[2:])


def test_hundred_requests_restore_both_pools() -> None:
    cleanup = _json("m6-cleanup")
    assert cleanup["num_requests"] == 100
    assert cleanup["small_initial_free_blocks"] == 291
    assert cleanup["small_final_free_blocks"] == 291
    assert cleanup["large_initial_free_blocks"] == 909
    assert cleanup["large_final_free_blocks"] == 909
