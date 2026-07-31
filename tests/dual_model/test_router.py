from vllm_dual.router import LARGE, SMALL, DeterministicRouter, route_for_position


def test_route_is_stable() -> None:
    expected = [route_for_position("r0", i, seed=17, p_large=0.1) for i in range(100)]
    assert expected == [route_for_position("r0", i, seed=17, p_large=0.1) for i in range(100)]


def test_batch_order_does_not_change_routes() -> None:
    request_ids = ["r0", "r1", "r2", "r3"]
    forward = {rid: [route_for_position(rid, i) for i in range(32)] for rid in request_ids}
    reverse = {rid: [route_for_position(rid, i) for i in range(32)] for rid in reversed(request_ids)}
    assert forward == reverse


def test_probability_extremes() -> None:
    assert {route_for_position("r0", i, p_large=0) for i in range(100)} == {SMALL}
    assert {route_for_position("r0", i, p_large=1) for i in range(100)} == {LARGE}


def test_forced_routes_override_hash() -> None:
    forced = [SMALL, SMALL, LARGE, SMALL, LARGE, LARGE, SMALL]
    router = DeterministicRouter(forced_routes=forced, p_large=0)
    assert [router.route("r0", i) for i in range(len(forced))] == forced


def test_invalid_probability_is_rejected() -> None:
    for probability in (-0.1, 1.1):
        try:
            route_for_position("r0", 0, p_large=probability)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid probability was accepted")
