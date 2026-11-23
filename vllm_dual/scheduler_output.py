"""Metadata planned by DualScheduler for one dual-model step."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DualStepMetadata:
    route: int | None  # None until the worker evaluates the GRPO head.
    output_index: int
    history_len_before: int
    large_before: int
    large_delta: int
    large_block_ids: tuple[list[int], ...]
    large_new_block_ids: tuple[list[int], ...]
    small_allocator: str = "small"
    large_allocator: str = "large"
    small_total_blocks: int = 0
    large_total_blocks: int = 0
    small_used_blocks: int = 0
    large_used_blocks: int = 0
