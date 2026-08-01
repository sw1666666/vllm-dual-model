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
