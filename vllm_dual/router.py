"""Order-independent deterministic token-level routing."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from typing import Sequence

SMALL = 0
LARGE = 1
ROUTE_NAMES = {SMALL: "small", LARGE: "large"}


def route_for_position(
    request_id: str,
    output_index: int,
    *,
    seed: int = 17,
    p_large: float = 0.10,
) -> int:
    if output_index < 0:
        raise ValueError("output_index must be non-negative")
    if not 0.0 <= p_large <= 1.0:
        raise ValueError("p_large must be in [0, 1]")
    payload = f"{seed}:{request_id}:{output_index}".encode()
    value = int.from_bytes(blake2b(payload, digest_size=8).digest(), "little")
    u = value / float(1 << 64)
    return LARGE if u < p_large else SMALL


@dataclass(frozen=True)
class DeterministicRouter:
    seed: int = 17
    p_large: float = 0.10
    forced_routes: Sequence[int] | None = None

    def route(self, request_id: str, output_index: int) -> int:
        if self.forced_routes is not None and output_index < len(self.forced_routes):
            route = self.forced_routes[output_index]
            if route not in (SMALL, LARGE):
                raise ValueError(f"invalid forced route: {route}")
            return route
        return route_for_position(
            request_id,
            output_index,
            seed=self.seed,
            p_large=self.p_large,
        )
