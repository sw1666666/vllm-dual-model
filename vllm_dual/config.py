"""Configuration shared by dual-model experiment components."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DualModelConfig:
    enabled: bool = False
    large_model: str = "Qwen/Qwen3-8B"
    router_seed: int = 17
    p_large: float = 0.10
    mode: str = "sync"
    sync_forward_enabled: bool = False
    forced_routes: list[int] | None = None
    small_kv_cache_bytes: int = 512 << 20
    large_kv_cache_bytes: int = 2 << 30
    trace_path: str = "artifacts/traces/dual.jsonl"

    def __post_init__(self) -> None:
        if self.mode not in {"disabled", "sync", "lazy"}:
            raise ValueError(f"unsupported dual-model mode: {self.mode}")
        if not 0.0 <= self.p_large <= 1.0:
            raise ValueError("p_large must be in [0, 1]")
        if self.small_kv_cache_bytes <= 0 or self.large_kv_cache_bytes <= 0:
            raise ValueError("KV cache byte budgets must be positive")
        if self.forced_routes is not None and any(
            route not in (0, 1) for route in self.forced_routes
        ):
            raise ValueError("forced routes must contain only 0 (small) or 1 (large)")

    @classmethod
    def from_additional_config(cls, additional_config: Any) -> "DualModelConfig":
        if not isinstance(additional_config, dict):
            return cls()
        value = additional_config.get("dual_model", {})
        if not isinstance(value, dict):
            raise TypeError("additional_config['dual_model'] must be a dict")
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
