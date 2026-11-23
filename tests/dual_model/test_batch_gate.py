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
