"""Request-level batching gate for lazy large-model forwards."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LargeBatchGate:
    """Hold LARGE-routed requests until a useful sub-batch is available."""

    min_batch_size: int
    max_wait_steps: int
    wait_steps: dict[str, int] = field(default_factory=dict)

    def select(
        self, running_ids: list[str], large_ids: set[str]
    ) -> set[str] | None:
        running = set(running_ids)
        large = running & large_ids
        for req_id in list(self.wait_steps):
            if req_id not in large:
                self.wait_steps.pop(req_id, None)
        if not large:
            return None

        for req_id in large:
            self.wait_steps[req_id] = self.wait_steps.get(req_id, 0) + 1
        flush = (
            len(large) >= self.min_batch_size
            or len(large) == len(running)
            or max(self.wait_steps[req_id] for req_id in large)
            >= self.max_wait_steps
        )
        if flush:
            for req_id in large:
                self.wait_steps.pop(req_id, None)
            return large
        return running - large

    def discard(self, req_ids: set[str]) -> None:
        for req_id in req_ids:
            self.wait_steps.pop(req_id, None)
