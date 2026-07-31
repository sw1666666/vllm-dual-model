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
