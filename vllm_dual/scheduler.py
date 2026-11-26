"""M2 passthrough scheduler hook."""

from __future__ import annotations

from copy import deepcopy

from transformers import AutoConfig
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler

from .batch_gate import LargeBatchGate
from .config import DualModelConfig
from .router import LARGE, DeterministicRouter
from .request_view import RequestIdView, RequestProgressView
from .scheduler_output import DualStepMetadata

logger = init_logger(__name__)


class DualScheduler(Scheduler):
    """A behavior-identical Scheduler that proves the custom hook is active."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dual_model_config = DualModelConfig.from_additional_config(
            self.vllm_config.additional_config
        )
        self.dual_router = DeterministicRouter(
            seed=self.dual_model_config.router_seed,
            p_large=self.dual_model_config.p_large,
            forced_routes=self.dual_model_config.forced_routes,
        )
        self.large_num_computed_tokens: dict[str, int] = {}
        # (request, public output index) -> (private input positions, small round).
        self.pending_large: dict[tuple[str, int], tuple[int, int]] = {}
        self.small_round = 0
        self.draining_large = False
        self.large_batch_gate = LargeBatchGate(
            min_batch_size=self.dual_model_config.large_min_batch_size,
            max_wait_steps=self.dual_model_config.large_max_wait_steps,
        )
        large_hf_config = AutoConfig.from_pretrained(
            self.dual_model_config.large_model
        )
        head_dim = getattr(
            large_hf_config,
            "head_dim",
            large_hf_config.hidden_size // large_hf_config.num_attention_heads,
        )
        large_block_bytes = (
            large_hf_config.num_hidden_layers
            * 2
            * large_hf_config.num_key_value_heads
            * head_dim
            * 2  # FP16 bytes.
            * self.block_size
        )
        large_num_blocks = (
            self.dual_model_config.large_kv_cache_bytes // large_block_bytes
        )
        if large_num_blocks <= 0:
            raise ValueError("large KV byte budget cannot hold one block")
        large_kv_cache_config = deepcopy(self.kv_cache_config)
        small_num_blocks = large_kv_cache_config.num_blocks
        outer_tensor_bytes = sum(
            tensor.size for tensor in self.kv_cache_config.kv_cache_tensors
        )
        if outer_tensor_bytes % small_num_blocks:
            raise AssertionError("outer KV tensors are not block aligned")
        small_block_bytes = outer_tensor_bytes // small_num_blocks
        configured_small_num_blocks = (
            self.dual_model_config.small_kv_cache_bytes // small_block_bytes
        )
        if configured_small_num_blocks <= 0:
            raise ValueError("small KV byte budget cannot hold one block")
        small_kv_cache_config = deepcopy(self.kv_cache_config)
        small_kv_cache_config.num_blocks = configured_small_num_blocks
        for tensor in small_kv_cache_config.kv_cache_tensors:
            tensor.size = (
                tensor.size // small_num_blocks * configured_small_num_blocks
            )

        large_kv_cache_config.num_blocks = large_num_blocks
        for tensor in large_kv_cache_config.kv_cache_tensors:
            if tensor.size % small_num_blocks:
                raise AssertionError("small KV tensor is not block aligned")
            tensor.size = tensor.size // small_num_blocks * large_num_blocks
        self.large_kv_cache_manager = KVCacheManager(
            kv_cache_config=large_kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=False,
            use_eagle=False,
            log_stats=False,
            enable_kv_cache_events=False,
            dcp_world_size=self.dcp_world_size,
            pcp_world_size=self.pcp_world_size,
            hash_block_size=self.block_size,
        )
        self.kv_cache_config = small_kv_cache_config
        self.small_total_blocks = configured_small_num_blocks
        self.large_total_blocks = large_num_blocks
        self.dual_cleanup_count = 0
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=small_kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=False,
            use_eagle=False,
            log_stats=False,
            enable_kv_cache_events=False,
            dcp_world_size=self.dcp_world_size,
            pcp_world_size=self.pcp_world_size,
            hash_block_size=self.block_size,
        )
        self.small_initial_free_blocks = (
            self.kv_cache_manager.block_pool.get_num_free_blocks()
        )
        self.large_initial_free_blocks = (
            self.large_kv_cache_manager.block_pool.get_num_free_blocks()
        )
        logger.info(
            "DUAL_ALLOCATORS small_blocks=%d large_blocks=%d "
            "small_block_bytes=%d large_block_bytes=%d",
            self.kv_cache_config.num_blocks,
            large_num_blocks,
            small_block_bytes,
            large_block_bytes,
        )
        logger.info(
            "DUAL_HOOK scheduler_loaded class=%s mode=%s enabled=%s",
            type(self).__name__,
            self.dual_model_config.mode,
            self.dual_model_config.enabled,
        )

    def schedule(self):
        if self.dual_model_config.queue_large and self.pending_large:
            for key in tuple(self.pending_large):
                if key[0] not in self.requests:
                    del self.pending_large[key]
            if self.pending_large:
                positions = sum(delta for delta, _ in self.pending_large.values())
                _, oldest = next(iter(self.pending_large.values()))
                no_small = len(self.running) == len(self.pending_large) and (
                    not self.waiting or len(self.running) >= self.max_num_running_reqs
                )
                if (self.draining_large or positions >= 2 * self.max_num_running_reqs
                        or self.small_round > oldest or no_small):
                    return self._schedule_large_flush()
        output = self._schedule_with_large_batch_gate()
        if not (
            self.dual_model_config.enabled
            and self.dual_model_config.sync_forward_enabled
        ):
            return output

        for finished_req_id in output.finished_req_ids:
            self.large_kv_cache_manager.free(RequestIdView(finished_req_id))
            self.large_num_computed_tokens.pop(finished_req_id, None)
        self.large_batch_gate.discard(set(output.finished_req_ids))
        if output.finished_req_ids:
            self.dual_cleanup_count += len(output.finished_req_ids)
            if self.dual_cleanup_count % 25 == 0:
                logger.info(
                    "DUAL_POOL_CLEANUP finished=%d small_free=%d large_free=%d",
                    self.dual_cleanup_count,
                    self.kv_cache_manager.block_pool.get_num_free_blocks(),
                    self.large_kv_cache_manager.block_pool.get_num_free_blocks(),
                )

        metadata: dict[str, DualStepMetadata] = {}
        sync_large = self.dual_model_config.mode == "sync"
        for req_id in output.num_scheduled_tokens:
            request = self.requests[req_id]
            history_len = len(request.all_token_ids)
            output_index = request.num_output_tokens
            # GRPO decides on this step's small hidden state in the worker.
            # Reserve pages up to the public history; reservation does not
            # imply that the large runner has populated those pages.
            learned = bool(self.dual_model_config.router_checkpoint)
            route = None if learned else self.dual_router.route(req_id, output_index)
            large_before = self.large_num_computed_tokens.get(req_id, 0)
            large_delta = (
                history_len - large_before
                if sync_large or route == LARGE
                else 0
            )
            reservation = history_len if learned else large_delta
            if reservation:
                large_view = RequestProgressView(request, large_before)
                new_large_blocks = self.large_kv_cache_manager.allocate_slots(
                    large_view,
                    reservation,
                )
                if new_large_blocks is None:
                    free_blocks = (
                        self.large_kv_cache_manager.block_pool.get_num_free_blocks()
                    )
                    raise RuntimeError(
                        "large KV pool exhausted: "
                        f"request_id={req_id} reservation={reservation} "
                        f"free_blocks={free_blocks}"
                    )
                large_new_block_ids = new_large_blocks.get_block_ids()
            else:
                large_new_block_ids = tuple(
                    []
                    for _ in range(
                        self.large_kv_cache_manager.num_kv_cache_groups
                    )
                )
            large_block_ids = self.large_kv_cache_manager.get_blocks(
                req_id
            ).get_block_ids()
            metadata[req_id] = DualStepMetadata(
                route=route,
                output_index=output_index,
                history_len_before=history_len,
                large_before=large_before,
                large_delta=large_delta,
                large_block_ids=large_block_ids,
                large_new_block_ids=large_new_block_ids,
                small_total_blocks=self.small_total_blocks,
                large_total_blocks=self.large_total_blocks,
                small_used_blocks=(
                    self.small_initial_free_blocks
                    - self.kv_cache_manager.block_pool.get_num_free_blocks()
                ),
                large_used_blocks=(
                    self.large_initial_free_blocks
                    - self.large_kv_cache_manager.block_pool.get_num_free_blocks()
                ),
            )
        output.dual_model_metadata = metadata
        return output

    def _schedule_large_flush(self):
        # Public small KV was computed on the routing round. Return a sample
        # for that same position without calling Scheduler._update_after_schedule.
        out = SchedulerOutput.make_empty()
        keys = []
        positions = 0
        for key, (delta, _) in self.pending_large.items():
            if delta > self.max_num_scheduled_tokens:
                raise ValueError("One pending history exceeds the large flush budget")
            if positions + delta > self.max_num_scheduled_tokens:
                break
            keys.append(key)
            positions += delta
        self.draining_large = len(keys) < len(self.pending_large)
        out.num_common_prefix_blocks = [0] * self.large_kv_cache_manager.num_kv_cache_groups
        out.dual_flush = keys
        out.num_scheduled_tokens = {rid: 1 for rid, _ in keys}
        out.total_num_scheduled_tokens = len(keys)
        return out

    def _schedule_with_large_batch_gate(self):
        if (
            self.dual_model_config.mode != "lazy"
            or not self.dual_model_config.sync_forward_enabled
            or self.dual_model_config.router_checkpoint
            or self.dual_model_config.large_min_batch_size == 1
        ):
            return super().schedule()

        original_running = list(self.running)
        running_ids = [request.request_id for request in original_running]
        large_ids = {
            request.request_id
            for request in original_running
            if self.dual_router.route(
                request.request_id, request.num_output_tokens
            ) == LARGE
        }
        selected_ids = self.large_batch_gate.select(running_ids, large_ids)
        if selected_ids is None or selected_ids == set(running_ids):
            return super().schedule()

        withheld_ids = set(running_ids) - selected_ids
        self.running = [
            request
            for request in original_running
            if request.request_id in selected_ids
        ]
        original_max_running = self.max_num_running_reqs
        # Keep the WAITING queue closed while RUNNING is partitioned. New
        # admissions here could overflow max_num_seqs when withheld requests
        # are restored after the scheduling pass.
        self.max_num_running_reqs = len(self.running)
        try:
            output = super().schedule()
            retained = {request.request_id: request for request in self.running}
        finally:
            self.max_num_running_reqs = original_max_running

        restored = []
        for request in original_running:
            req_id = request.request_id
            if req_id in withheld_ids:
                restored.append(request)
            elif req_id in retained:
                restored.append(retained.pop(req_id))
        restored.extend(retained.values())
        self.running = restored
        return output

    def update_from_output(self, scheduler_output, model_runner_output):
        if self.dual_model_config.queue_large:
            flush = getattr(scheduler_output, "dual_flush", ())
            if flush:
                for key in flush:
                    del self.pending_large[key]
            elif scheduler_output.total_num_scheduled_tokens:
                self.small_round += 1
                for key, delta in model_runner_output.dual_deferred:
                    if key in self.pending_large:
                        raise RuntimeError(f"Output position was queued twice: {key}")
                    self.pending_large[key] = (delta, self.small_round)
        result = super().update_from_output(scheduler_output, model_runner_output)
        metadata = getattr(scheduler_output, "dual_model_metadata", {})
        sync_large = self.dual_model_config.mode == "sync"
        for req_id, step in metadata.items():
            if sync_large or step.route == LARGE:
                self.large_num_computed_tokens[req_id] = step.history_len_before
            if req_id not in self.requests:
                self.large_kv_cache_manager.free(RequestIdView(req_id))
                self.large_num_computed_tokens.pop(req_id, None)
        return result
