"""Configuration shared by dual-model experiment components."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from typing import Any


DEFAULT_LARGE_THRESHOLD = 0.5


def validate_large_threshold(value: float) -> float:
    if not 0.1 <= value <= 0.9:
        raise ValueError("router_large_threshold must be in [0.1, 0.9]")
    return value


def add_grpo_threshold_args(parser: argparse.ArgumentParser) -> None:
    def probability(text: str) -> float:
        try:
            return validate_large_threshold(float(text))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc

    def small_large_pair(text: str) -> float:
        parts = text.split("/")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError("expected SMALL/LARGE, e.g. 0.4/0.6")
        small, large = map(probability, parts)
        if not math.isclose(small + large, 1.0, rel_tol=0, abs_tol=1e-9):
            raise argparse.ArgumentTypeError("SMALL/LARGE probabilities must sum to 1")
        return large

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--router-large-threshold", type=probability, default=DEFAULT_LARGE_THRESHOLD,
                       help="Use the large model when its GRPO probability exceeds this value in [0.1, 0.9] (default: 0.5).")
    group.add_argument("--router-thresholds", type=small_large_pair, metavar="SMALL/LARGE",
                       dest="router_large_threshold", default=argparse.SUPPRESS,
                       help="Equivalent boundary as SMALL/LARGE, e.g. 0.4/0.6 for P(large) > 0.6.")


@dataclass(frozen=True)
class DualModelConfig:
    enabled: bool = False
    large_model: str = "Qwen/Qwen3-8B"
    router_seed: int = 17
    p_large: float = 0.10
    mode: str = "sync"
    sync_forward_enabled: bool = False
    forced_routes: list[int] | None = None
    large_min_batch_size: int = 1
    large_max_wait_steps: int = 1
    small_kv_cache_bytes: int = 1120 << 20
    large_kv_cache_bytes: int = 1440 << 20
    trace_path: str | None = "artifacts/traces/dual.jsonl"
    router_checkpoint: str | None = None
    router_large_threshold: float = DEFAULT_LARGE_THRESHOLD
    queue_large: bool = False
    merge_route_readback: bool = False

    margin_v4: bool = False

    def __post_init__(self) -> None:
        if self.merge_route_readback and not (self.queue_large and self.router_checkpoint):
            raise ValueError("GRPO V3 requires a router checkpoint and queued 8B execution")
        validate_large_threshold(self.router_large_threshold)
        if self.margin_v4 and (not self.enabled or not self.sync_forward_enabled or self.mode != "lazy"
                               or not self.queue_large or self.router_checkpoint or self.forced_routes is not None):
            raise ValueError("Margin V4 requires queued lazy dual forward")
        if self.queue_large and not (self.router_checkpoint or self.margin_v4):
            raise ValueError("Queued large execution requires the online GRPO router")
        if self.router_checkpoint and (
            not self.enabled
            or not self.sync_forward_enabled
            or self.mode != "lazy"
            or self.forced_routes is not None
        ):
            raise ValueError("GRPO requires enabled lazy dual forward without forced routes")
        if self.mode not in {"disabled", "sync", "lazy"}:
            raise ValueError(f"unsupported dual-model mode: {self.mode}")
        if not 0.0 <= self.p_large <= 1.0:
            raise ValueError("p_large must be in [0, 1]")
        if self.small_kv_cache_bytes <= 0 or self.large_kv_cache_bytes <= 0:
            raise ValueError("KV cache byte budgets must be positive")
        if self.large_min_batch_size <= 0:
            raise ValueError("large_min_batch_size must be positive")
        if self.large_max_wait_steps <= 0:
            raise ValueError("large_max_wait_steps must be positive")
        if self.forced_routes is not None and any(
            route not in (0, 1) for route in self.forced_routes
        ):
            raise ValueError("forced routes must contain only 0 (small) or 1 (large)")

    @property
    def grpo_variant(self) -> str:
        if self.margin_v4:
            return "v4"
        if self.merge_route_readback:
            return "v3"
        return "v2" if self.queue_large else "v1"

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
