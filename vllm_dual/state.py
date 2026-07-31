"""The two-counter state machine shared by reference and vLLM experiments."""

from __future__ import annotations

from dataclasses import dataclass

from .router import LARGE, SMALL


@dataclass(frozen=True)
class StepPlan:
    route: int
    history_len_before: int
    small_before: int
    small_delta: int
    large_before: int
    large_delta: int


@dataclass
class DualRequestState:
    small_num_computed_tokens: int = 0
    large_num_computed_tokens: int = 0
    output_index: int = 0

    def plan(self, history_len: int, route: int, *, sync_large: bool = False) -> StepPlan:
        if route not in (SMALL, LARGE):
            raise ValueError(f"invalid route: {route}")
        if not 0 <= self.large_num_computed_tokens <= self.small_num_computed_tokens:
            raise AssertionError("invalid computed-token ordering")
        if history_len < self.small_num_computed_tokens:
            raise AssertionError("public history moved backwards")
        large_should_run = sync_large or route == LARGE
        return StepPlan(
            route=route,
            history_len_before=history_len,
            small_before=self.small_num_computed_tokens,
            small_delta=history_len - self.small_num_computed_tokens,
            large_before=self.large_num_computed_tokens,
            large_delta=(history_len - self.large_num_computed_tokens)
            if large_should_run
            else 0,
        )

    def commit_forwards(self, plan: StepPlan, *, sync_large: bool = False) -> None:
        self.small_num_computed_tokens = plan.history_len_before
        if sync_large or plan.route == LARGE:
            self.large_num_computed_tokens = plan.history_len_before
        self._assert_after_forward(plan, sync_large=sync_large)

    def commit_sample(self) -> None:
        self.output_index += 1

    def _assert_after_forward(self, plan: StepPlan, *, sync_large: bool) -> None:
        assert 0 <= self.large_num_computed_tokens <= self.small_num_computed_tokens
        assert self.small_num_computed_tokens == plan.history_len_before
        if sync_large or plan.route == LARGE:
            assert self.large_num_computed_tokens == plan.history_len_before
        else:
            assert self.large_num_computed_tokens == plan.large_before
