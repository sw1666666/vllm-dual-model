import pytest

from vllm_dual.config import DualModelConfig


def test_config_round_trip_from_additional_config() -> None:
    expected = DualModelConfig(enabled=True, mode="lazy", p_large=0.25)
    actual = DualModelConfig.from_additional_config({"dual_model": expected.to_dict()})
    assert actual == expected


def test_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        DualModelConfig(mode="unknown")
    with pytest.raises(ValueError):
        DualModelConfig(p_large=1.1)
    with pytest.raises(ValueError):
        DualModelConfig(forced_routes=[0, 2])


def test_default_kv_budgets_keep_total_and_balance_block_capacity() -> None:
    config = DualModelConfig()
    assert config.small_kv_cache_bytes == 1120 << 20
    assert config.large_kv_cache_bytes == 1440 << 20
    assert config.small_kv_cache_bytes + config.large_kv_cache_bytes == 2560 << 20


def test_grpo_threshold_cli():
    import argparse
    from vllm_dual.config import add_grpo_threshold_args

    parser = argparse.ArgumentParser()
    add_grpo_threshold_args(parser)
    assert parser.parse_args([]).router_large_threshold == 0.5
    assert parser.parse_args(["--router-large-threshold", "0.6"]).router_large_threshold == 0.6
    assert parser.parse_args(["--router-thresholds", "0.6/0.4"]).router_large_threshold == 0.4
    for value in ("0.1", "0.375", "0.9"):
        assert parser.parse_args(["--router-large-threshold", value]).router_large_threshold == float(value)
    for value in ("0", "0.0999", "0.9001", "1", "inf", "nan"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--router-large-threshold", value])
        with pytest.raises(ValueError, match="router_large_threshold"):
            DualModelConfig(router_large_threshold=float(value))
    for value in ("0.05/0.95", "0.7/0.7", "-0.1/1.1", "nan/0.5", "0.6"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--router-thresholds", value])
    with pytest.raises(ValueError, match="router_large_threshold"):
        DualModelConfig(router_large_threshold=float("nan"))


def test_margin_v4_requires_queued_lazy_routing():
    cfg = DualModelConfig(margin_v4=True, enabled=True, sync_forward_enabled=True,
                          mode="lazy", queue_large=True)
    assert cfg.grpo_variant == "v4"
    assert DualModelConfig.from_additional_config({"dual_model": cfg.to_dict()}) == cfg
    for override in ({"queue_large": False}, {"router_checkpoint": "policy.pt"},
                     {"mode": "sync"}, {"forced_routes": [0]}):
        with pytest.raises(ValueError):
            DualModelConfig(**{**cfg.to_dict(), **override})
