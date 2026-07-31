"""Acceptance tests for M7 compact routed large-model sub-batches."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from vllm_dual.router import DeterministicRouter, ROUTE_NAMES


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


def _by_request(rows: list[dict]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        result[row["request_id"]].append(row)
    return result


def test_batch_two_and_four_cover_every_request() -> None:
    for name, size in (("m7-batch2-smoke", 2), ("m7-batch4-smoke", 4)):
        result = _json(name)
        rows = _trace(name)
        assert result["max_num_seqs"] == size
        assert len(result["outputs"]) == size
        assert len(_by_request(rows)) == size
        assert all(len(output["output_token_ids"]) == 4 for output in result["outputs"])


def test_batch_four_matches_each_isolated_request() -> None:
    outputs = _json("m7-batch4-smoke")["outputs"]
    for index, output in enumerate(outputs):
        isolated = _json(f"m7-single{index}")
        assert output["prompt_token_ids"] == isolated["prompt_token_ids"]
        assert output["output_token_ids"] == isolated["output_token_ids"]


def test_routes_are_stable_and_independent_of_batch_order() -> None:
    rows = _trace("m7-batch4-smoke")
    router = DeterministicRouter(seed=17, p_large=0.5)
    for row in rows:
        route = router.route(row["request_id"], row["output_index"])
        assert row["route"] == ROUTE_NAMES[route]
    expected = {
        req_id: [router.route(req_id, i) for i in range(4)]
        for req_id in ("0", "1", "2", "3")
    }
    reversed_order = {
        req_id: [router.route(req_id, i) for i in range(4)]
        for req_id in reversed(("0", "1", "2", "3"))
    }
    assert expected == reversed_order


def test_mixed_gather_scatter_and_ragged_catchup() -> None:
    rows = _trace("m7-batch4-smoke")
    by_step: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_step[row["output_index"]].append(row)
        assert row["accepted_token"] == row["selected_argmax"]
        assert row["large_sub_batch_size"] == sum(
            candidate["route"] == "large"
            for candidate in rows
            if candidate["output_index"] == row["output_index"]
        )
        if row["route"] == "small":
            assert row["large_row"] is None
            assert row["large_delta"] == 0
        else:
            assert row["large_row"] is not None
            assert row["large_argmax"] == row["accepted_token"]
    assert any(
        {row["route"] for row in step_rows} == {"small", "large"}
        for step_rows in by_step.values()
    )
    assert any(
        len({row["large_delta"] for row in step_rows if row["route"] == "large"}) > 1
        for step_rows in by_step.values()
    )
    assert any(
        row["large_row"] != row["global_row"]
        for row in rows
        if row["route"] == "large"
    )


def test_hundred_batched_requests_restore_both_pools() -> None:
    cleanup = _json("m7-cleanup")
    assert cleanup["num_requests"] == 100
    assert cleanup["max_num_seqs"] == 4
    assert cleanup["small_initial_free_blocks"] == cleanup["small_final_free_blocks"]
    assert cleanup["large_initial_free_blocks"] == cleanup["large_final_free_blocks"]


def test_finishing_one_request_does_not_remove_its_batch_peer() -> None:
    result = _json("m7-uneven-finish")
    rows = _by_request(_trace("m7-uneven-finish"))
    assert [len(output["output_token_ids"]) for output in result["outputs"]] == [1, 4]
    assert len(rows["0"]) == 1
    assert len(rows["1"]) == 4
    assert [row["output_index"] for row in rows["1"]] == [0, 1, 2, 3]
