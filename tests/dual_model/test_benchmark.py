"""Unit tests for the M8 dual-model benchmark helpers."""

from __future__ import annotations

from benchmarks.benchmark_dual_model import clustered_routes, percentile
from vllm_dual.router import LARGE, SMALL


def test_clustered_routes_preserve_probability_endpoints() -> None:
    assert clustered_routes(0.0) == [SMALL] * 32
    assert clustered_routes(1.0) == [LARGE] * 32


def test_clustered_routes_are_contiguous_and_deterministic() -> None:
    routes = clustered_routes(0.1)
    assert routes == [LARGE] * 3 + [SMALL] * 29
    assert clustered_routes(0.1) == routes


def test_percentile_uses_nearest_observed_value() -> None:
    values = [4.0, 1.0, 3.0, 2.0]
    assert percentile(values, 0.50) == 3.0
    assert percentile(values, 0.95) == 4.0
