from vllm_dual.batch_gate import LargeBatchGate


def test_small_requests_run_while_large_requests_accumulate() -> None:
    gate = LargeBatchGate(min_batch_size=3, max_wait_steps=4)
    assert gate.select(["l0", "s0", "s1"], {"l0"}) == {"s0", "s1"}
    assert gate.select(["l0", "l1", "s1"], {"l0", "l1"}) == {"s1"}
    assert gate.select(["l0", "l1", "l2"], {"l0", "l1", "l2"}) == {
        "l0",
        "l1",
        "l2",
    }


def test_wait_limit_flushes_partial_large_batch() -> None:
    gate = LargeBatchGate(min_batch_size=4, max_wait_steps=2)
    assert gate.select(["l0", "s0"], {"l0"}) == {"s0"}
    assert gate.select(["l0", "s0"], {"l0"}) == {"l0"}


def test_all_large_flushes_immediately_to_avoid_stall() -> None:
    gate = LargeBatchGate(min_batch_size=8, max_wait_steps=8)
    assert gate.select(["l0", "l1"], {"l0", "l1"}) == {"l0", "l1"}


def test_discard_removes_finished_waiters() -> None:
    gate = LargeBatchGate(min_batch_size=4, max_wait_steps=4)
    gate.select(["l0", "s0"], {"l0"})
    gate.discard({"l0"})
    assert gate.wait_steps == {}


def test_grpo_flush_splits_by_private_positions_and_returns_one_public_sample():
    from types import SimpleNamespace

    from vllm_dual.scheduler import DualScheduler

    pending = {("a", 3): (130, 1), ("b", 4): (100, 1), ("c", 2): (50, 2)}
    scheduler = SimpleNamespace(
        pending_large=pending.copy(),
        max_num_scheduled_tokens=256,
        large_kv_cache_manager=SimpleNamespace(num_kv_cache_groups=1),
    )
    first = DualScheduler._schedule_large_flush(scheduler)
    assert first.dual_flush == [("a", 3), ("b", 4)]
    assert first.num_scheduled_tokens == {"a": 1, "b": 1}
    assert first.total_num_scheduled_tokens == 2
    assert first.num_common_prefix_blocks == [0]
    assert scheduler.draining_large
    assert scheduler.pending_large == pending
    for key in first.dual_flush:
        del scheduler.pending_large[key]
    second = DualScheduler._schedule_large_flush(scheduler)
    assert second.dual_flush == [("c", 2)]
    assert not scheduler.draining_large
