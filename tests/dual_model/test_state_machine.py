from vllm_dual.router import LARGE, SMALL
from vllm_dual.state import DualRequestState


def run_routes(prompt_len: int, routes: list[int]) -> list[dict[str, int]]:
    history_len = prompt_len
    state = DualRequestState()
    rows = []
    for route in routes:
        plan = state.plan(history_len, route)
        state.commit_forwards(plan)
        rows.append(
            {
                "history": history_len,
                "small_before": plan.small_before,
                "small_delta": plan.small_delta,
                "small_after": state.small_num_computed_tokens,
                "large_before": plan.large_before,
                "large_delta": plan.large_delta,
                "large_after": state.large_num_computed_tokens,
            }
        )
        state.commit_sample()
        history_len += 1
    return rows


def test_forced_route_timeline() -> None:
    rows = run_routes(10, [SMALL, SMALL, LARGE, SMALL, LARGE, LARGE, SMALL])
    assert [(r["small_delta"], r["large_delta"]) for r in rows] == [
        (10, 0),
        (1, 0),
        (1, 12),
        (1, 0),
        (1, 2),
        (1, 1),
        (1, 0),
    ]
    assert [r["large_after"] for r in rows] == [0, 0, 12, 12, 14, 15, 15]


def test_late_first_large_catches_up_exact_slice() -> None:
    prompt_len = 10
    rows = run_routes(prompt_len, [SMALL] * 15 + [LARGE])
    late = rows[-1]
    assert late["history"] == prompt_len + 15
    assert late["large_before"] == 0
    assert late["large_delta"] == prompt_len + 15
    assert late["large_after"] == prompt_len + 15


def test_counters_stay_one_behind_after_sampling() -> None:
    state = DualRequestState()
    history_len = 10
    for route in [LARGE, SMALL, LARGE]:
        plan = state.plan(history_len, route)
        state.commit_forwards(plan)
        assert state.small_num_computed_tokens == history_len
        state.commit_sample()
        history_len += 1
        assert state.small_num_computed_tokens == history_len - 1


def test_sync_updates_both_progress_counters() -> None:
    state = DualRequestState()
    history_len = 10
    for route in [SMALL, SMALL, LARGE]:
        plan = state.plan(history_len, route, sync_large=True)
        state.commit_forwards(plan, sync_large=True)
        assert state.small_num_computed_tokens == history_len
        assert state.large_num_computed_tokens == history_len
        state.commit_sample()
        history_len += 1
