"""M2 passthrough worker hook."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import torch
from transformers import AutoConfig, AutoTokenizer
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker

from .config import DualModelConfig
from .grpo_router import GrpoRouter
from .model_runner import DualRuntime
from .router import LARGE, ROUTE_NAMES, DeterministicRouter
from .scheduler_output import DualStepMetadata
from .state import DualRequestState, StepPlan
from .trace import JsonlTraceWriter

logger = init_logger(__name__)


class DualWorker(Worker):
    """A behavior-identical Worker that proves the custom hook is active."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dual_model_config = DualModelConfig.from_additional_config(
            self.vllm_config.additional_config
        )
        logger.info(
            "DUAL_HOOK worker_loaded class=%s mode=%s enabled=%s",
            type(self).__name__,
            self.dual_model_config.mode,
            self.dual_model_config.enabled,
        )
        self.dual_router = DeterministicRouter(
            seed=self.dual_model_config.router_seed,
            p_large=self.dual_model_config.p_large,
            forced_routes=self.dual_model_config.forced_routes,
        )
        self.grpo_router: GrpoRouter | None = None
        self.dual_request_states: dict[str, DualRequestState] = {}
        self.large_model_runner: GPUModelRunner | None = None
        self.pending_dual_steps: dict[
            str, tuple[StepPlan, DualStepMetadata, list[int]]
        ] = {}
        self.dual_cleanup_count = 0
        self.dual_trace = JsonlTraceWriter(
            self.dual_model_config.trace_path
            if self.dual_model_config.sync_forward_enabled
            else None
        )

    def _memory_snapshot(self, stage: str) -> None:
        torch.cuda.synchronize(self.device)
        free, total = torch.cuda.mem_get_info(self.device)
        logger.info(
            "DUAL_MEMORY stage=%s allocated_gib=%.3f reserved_gib=%.3f "
            "free_gib=%.3f total_gib=%.3f",
            stage,
            torch.cuda.memory_allocated(self.device) / 2**30,
            torch.cuda.memory_reserved(self.device) / 2**30,
            free / 2**30,
            total / 2**30,
        )

    def _check_model_compatibility(self) -> None:
        small_path = self.model_config.model
        large_path = self.dual_model_config.large_model
        small_config = AutoConfig.from_pretrained(small_path)
        large_config = AutoConfig.from_pretrained(large_path)
        for field in ("vocab_size", "eos_token_id", "bos_token_id"):
            small_value = getattr(small_config, field)
            large_value = getattr(large_config, field)
            if small_value != large_value:
                raise ValueError(
                    f"dual model {field} mismatch: {small_value} != {large_value}"
                )
        small_tokenizer = AutoTokenizer.from_pretrained(small_path)
        large_tokenizer = AutoTokenizer.from_pretrained(large_path)
        if small_tokenizer.get_vocab() != large_tokenizer.get_vocab():
            raise ValueError("dual model tokenizer vocabularies differ")
        logger.info("DUAL_COMPAT tokenizer_and_config=true")

    def load_model(self) -> None:
        large_vllm_config = None
        if self.dual_model_config.enabled:
            # Copy before loading small: model loading populates the compilation
            # config's static forward context with attention modules.
            large_vllm_config = deepcopy(self.vllm_config)
            large_model_config = replace(
                large_vllm_config.model_config,
                model=self.dual_model_config.large_model,
                tokenizer=self.dual_model_config.large_model,
                served_model_name=["large"],
            )
            # A compact large sub-batch can catch up multiple long histories
            # even though the public scheduler only advances one token per
            # request. Size the private runner's token buffers for that worst
            # case instead of inheriting the public per-step token budget.
            large_scheduler_config = replace(
                large_vllm_config.scheduler_config,
                max_model_len=large_vllm_config.model_config.max_model_len,
                is_encoder_decoder=(
                    large_vllm_config.model_config.is_encoder_decoder
                ),
                max_num_batched_tokens=(
                    large_vllm_config.scheduler_config.max_num_seqs
                    * large_vllm_config.model_config.max_model_len
                ),
            )
            large_vllm_config = replace(
                large_vllm_config,
                model_config=large_model_config,
                scheduler_config=large_scheduler_config,
            )

        self._memory_snapshot("before_small")
        super().load_model()
        self.small_runtime = DualRuntime(
            name="small",
            model=self.model_runner.model,
            model_config=self.model_config,
            static_forward_context=self.compilation_config.static_forward_context,
            kv_namespace="small::",
        )
        self._memory_snapshot("after_small")
        if not self.dual_model_config.enabled:
            self.large_runtime = None
            return

        assert large_vllm_config is not None
        self._check_model_compatibility()
        if self.dual_model_config.router_checkpoint:
            self.grpo_router = GrpoRouter(
                self.dual_model_config.router_checkpoint, self.model_config.model,
                self.model_config.hf_config.hidden_size, self.device,
                large_threshold=self.dual_model_config.router_large_threshold,
            )
        self.large_model_runner = GPUModelRunner(large_vllm_config, self.device)
        self.large_model_runner.load_model()
        self.large_runtime = DualRuntime(
            name="large",
            model=self.large_model_runner.model,
            model_config=large_vllm_config.model_config,
            static_forward_context=(
                large_vllm_config.compilation_config.static_forward_context
            ),
            kv_namespace="large::",
        )
        self._memory_snapshot("after_large")
        logger.info(
            "DUAL_RESIDENT small=true large=true large_forward_enabled=%s",
            self.dual_model_config.sync_forward_enabled,
        )

    def initialize_from_config(self, kv_cache_config) -> None:
        if self.large_model_runner is None:
            super().initialize_from_config(kv_cache_config)
            return

        small_spec = self.model_runner.get_kv_cache_spec()
        small_kv_config = get_kv_cache_configs(
            self.vllm_config,
            [small_spec],
            [self.dual_model_config.small_kv_cache_bytes],
        )[0]
        super().initialize_from_config(small_kv_config)
        large_spec = self.large_model_runner.get_kv_cache_spec()
        large_kv_config = get_kv_cache_configs(
            self.large_model_runner.vllm_config,
            [large_spec],
            [self.dual_model_config.large_kv_cache_bytes],
        )[0]
        small_bytes = sum(t.size for t in small_kv_config.kv_cache_tensors)
        large_bytes = sum(t.size for t in large_kv_config.kv_cache_tensors)
        outer_bytes = sum(t.size for t in kv_cache_config.kv_cache_tensors)
        outer_block_bytes = outer_bytes // kv_cache_config.num_blocks
        if small_bytes > self.dual_model_config.small_kv_cache_bytes:
            raise AssertionError("small KV allocation exceeds configured budget")
        if large_bytes > self.dual_model_config.large_kv_cache_bytes:
            raise AssertionError("large KV allocation exceeds configured budget")
        if small_bytes + large_bytes > outer_bytes + outer_block_bytes:
            raise AssertionError("split KV allocations exceed outer budget")
        self.large_model_runner.initialize_kv_cache(large_kv_config)
        logger.info(
            "DUAL_KV split_pools=true small_blocks=%d large_blocks=%d "
            "small_bytes=%d large_bytes=%d",
            small_kv_config.num_blocks,
            large_kv_config.num_blocks,
            small_bytes,
            large_bytes,
        )

    def compile_or_warm_up_model(self) -> None:
        super().compile_or_warm_up_model()
        if (
            self.large_model_runner is not None
            and not self.model_config.enforce_eager
        ):
            self.large_model_runner.capture_model()

    @torch.inference_mode()
    def execute_model(self, scheduler_output):
        for finished_req_id in scheduler_output.finished_req_ids:
            self.dual_request_states.pop(finished_req_id, None)
            if self.grpo_router is not None:
                self.grpo_router.discard(finished_req_id)

        if (
            self.dual_model_config.mode == "lazy"
            and self.large_model_runner is not None
            and scheduler_output.finished_req_ids
        ):
            cleanup_output = SchedulerOutput.make_empty()
            cleanup_output.finished_req_ids = set(scheduler_output.finished_req_ids)
            self.large_model_runner.execute_model(cleanup_output)
            self.dual_cleanup_count += len(scheduler_output.finished_req_ids)
            if self.dual_cleanup_count % 25 == 0:
                logger.info(
                    "DUAL_CLEANUP finished=%d dual_states=%d large_requests=%d "
                    "allocated_gib=%.3f reserved_gib=%.3f",
                    self.dual_cleanup_count,
                    len(self.dual_request_states),
                    len(self.large_model_runner.requests),
                    torch.cuda.memory_allocated(self.device) / 2**30,
                    torch.cuda.memory_reserved(self.device) / 2**30,
                )

        output = super().execute_model(scheduler_output)
        if not self.dual_model_config.sync_forward_enabled:
            return output
        if self.large_model_runner is None:
            raise RuntimeError("dual forward enabled without a large vLLM runner")

        if self.model_runner.input_batch.num_reqs == 0:
            return output
        scheduler_metadata = getattr(scheduler_output, "dual_model_metadata", None)
        if scheduler_metadata is None:
            raise RuntimeError("DualScheduler did not attach route metadata")
        sync_large = self.dual_model_config.mode == "sync"
        self.pending_dual_steps = {}
        large_req_ids: list[str] = []
        req_ids = self.model_runner.input_batch.req_ids
        actions = None
        if self.grpo_router is not None:
            hidden = self.model_runner.execute_model_state.sample_hidden_states
            actions = self.grpo_router.decide(req_ids, hidden)
        for row, req_id in enumerate(req_ids):
            if req_id not in scheduler_output.num_scheduled_tokens:
                continue
            request = self.model_runner.requests[req_id]
            if request.prompt_token_ids is None:
                raise AssertionError("M7 does not support prompt embeddings")
            history = list(request.prompt_token_ids) + list(request.output_token_ids)
            state = self.dual_request_states.setdefault(req_id, DualRequestState())
            if req_id not in scheduler_metadata:
                raise RuntimeError(f"missing route metadata for {req_id}")
            scheduled_step = scheduler_metadata[req_id]
            if actions is not None:
                before = state.large_num_computed_tokens
                action = actions[row]
                scheduled_step = replace(
                    scheduled_step, route=action, large_before=before,
                    large_delta=len(history) - before if action == LARGE else 0,
                )
            if scheduled_step.output_index != state.output_index:
                raise AssertionError("scheduler/worker output index mismatch")
            if scheduled_step.history_len_before != len(history):
                raise AssertionError("scheduler/worker public history mismatch")
            if scheduled_step.large_before != state.large_num_computed_tokens:
                raise AssertionError("scheduler/worker large progress mismatch")
            plan = state.plan(len(history), scheduled_step.route, sync_large=sync_large)
            if scheduled_step.large_delta != plan.large_delta:
                raise AssertionError("scheduler/worker large delta mismatch")
            self.pending_dual_steps[req_id] = (plan, scheduled_step, history)
            if sync_large or scheduled_step.route == LARGE:
                large_req_ids.append(req_id)

        if not large_req_ids:
            return output
        large_scheduler_output = self._make_lazy_large_output(
            scheduler_output, large_req_ids
        )
        large_output = self.large_model_runner.execute_model(large_scheduler_output)
        if large_output is not None:
            raise AssertionError("M4 expects deferred sampling from both runners")
        return output

    def _make_lazy_large_output(
        self,
        base: SchedulerOutput,
        req_ids: list[str],
    ) -> SchedulerOutput:
        new_reqs: list[NewRequestData] = []
        cached_ids: list[str] = []
        cached_block_ids = []
        cached_computed = []
        cached_output_tokens = []
        num_scheduled_tokens: dict[str, int] = {}
        for req_id in req_ids:
            plan, scheduled_step, history = self.pending_dual_steps[req_id]
            if plan.large_delta <= 0:
                raise AssertionError("a LARGE route must schedule at least one token")
            num_scheduled_tokens[req_id] = plan.large_delta
            small_request = self.model_runner.requests[req_id]
            if req_id not in self.large_model_runner.requests:
                new_reqs.append(
                    NewRequestData(
                        req_id=req_id,
                        prompt_token_ids=history.copy(),
                        mm_features=[],
                        sampling_params=small_request.sampling_params,
                        pooling_params=small_request.pooling_params,
                        block_ids=deepcopy(scheduled_step.large_block_ids),
                        num_computed_tokens=0,
                        lora_request=small_request.lora_request,
                    )
                )
            else:
                large_request = self.large_model_runner.requests[req_id]
                cached_ids.append(req_id)
                if self.grpo_router is None:
                    new_blocks = scheduled_step.large_new_block_ids
                else:
                    # Include every page reserved since this request's last
                    # large call, including pages reserved during small steps.
                    new_blocks = tuple(
                        full[len(previous):]
                        for full, previous in zip(
                            scheduled_step.large_block_ids, large_request.block_ids,
                            strict=True,
                        )
                    )
                cached_block_ids.append(deepcopy(new_blocks))
                cached_computed.append(plan.large_before)
                cached_output_tokens.append(len(large_request.output_token_ids))
        cached_reqs = CachedRequestData(
            req_ids=cached_ids,
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=cached_block_ids,
            num_computed_tokens=cached_computed,
            num_output_tokens=cached_output_tokens,
        )

        return replace(
            deepcopy(base),
            scheduled_new_reqs=new_reqs,
            scheduled_cached_reqs=cached_reqs,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0] * len(base.num_common_prefix_blocks),
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )

    @torch.inference_mode()
    def sample_tokens(self, grammar_output):
        if not self.dual_model_config.sync_forward_enabled:
            return super().sample_tokens(grammar_output)
        execute_state = self.model_runner.execute_model_state
        if execute_state is None:
            return super().sample_tokens(grammar_output)
        if not self.pending_dual_steps:
            raise RuntimeError("missing dual step planned before forward")
        sync_large = self.dual_model_config.mode == "sync"
        small_logits = execute_state.logits
        if self.large_model_runner is None:
            raise RuntimeError("large vLLM runner is not initialized")
        large_execute_state = self.large_model_runner.execute_model_state
        large_req_ids = [
            req_id
            for req_id, (plan, _, _) in self.pending_dual_steps.items()
            if sync_large or plan.route == LARGE
        ]
        if large_req_ids and large_execute_state is None:
            raise RuntimeError("large vLLM runner did not produce logits")
        large_logits = (
            large_execute_state.logits.to(
                device=small_logits.device,
                dtype=small_logits.dtype,
            )
            if large_execute_state is not None
            else None
        )
        if large_logits is not None and not torch.isfinite(large_logits).all():
            raise FloatingPointError("large catch-up logits contain NaN or Inf")
        if large_execute_state is not None:
            # The large runner never samples. Clear its deferred-sampling state
            # so it can accept the next SchedulerOutput.
            self.large_model_runner.execute_model_state = None
        selected_logits = small_logits.clone()
        large_to_global_row: dict[str, int] = {}
        if large_logits is not None:
            for large_row, req_id in enumerate(
                self.large_model_runner.input_batch.req_ids
            ):
                global_row = self.model_runner.input_batch.req_id_to_index[req_id]
                large_to_global_row[req_id] = global_row
                if self.pending_dual_steps[req_id][0].route == LARGE:
                    selected_logits[global_row].copy_(large_logits[large_row])
        self.model_runner.execute_model_state = execute_state._replace(
            logits=selected_logits
        )
        output = super().sample_tokens(grammar_output)
        small_kv_ptr = int(self.model_runner.kv_caches[0].data_ptr())
        large_kv_ptr = int(self.large_model_runner.kv_caches[0].data_ptr())
        assert small_kv_ptr != large_kv_ptr
        for req_id in self.model_runner.input_batch.req_ids:
            if req_id not in self.pending_dual_steps:
                continue
            plan, scheduled_step, history = self.pending_dual_steps[req_id]
            state = self.dual_request_states[req_id]
            state.commit_forwards(plan, sync_large=sync_large)
            global_row = self.model_runner.input_batch.req_id_to_index[req_id]
            small_argmax = int(small_logits.argmax(dim=-1)[global_row].item())
            large_argmax = None
            if req_id in large_to_global_row:
                large_row = self.large_model_runner.input_batch.req_id_to_index[req_id]
                large_argmax = int(large_logits.argmax(dim=-1)[large_row].item())
            selected_argmax = large_argmax if plan.route == LARGE else small_argmax
            accepted_token = output.sampled_token_ids[global_row][0]
            assert selected_argmax is not None and accepted_token == selected_argmax
            if req_id in self.large_model_runner.requests:
                large_batch = self.large_model_runner.input_batch
                if req_id in large_batch.req_id_to_index:
                    large_req_index = large_batch.req_id_to_index[req_id]
                    large_start = int(large_batch.num_tokens_no_spec[large_req_index])
                    large_end = large_start + 1
                    large_batch.token_ids_cpu[
                        large_req_index, large_start:large_end
                    ] = [accepted_token]
                    large_batch.is_token_ids[
                        large_req_index, large_start:large_end
                    ] = True
                    large_batch.num_tokens_no_spec[large_req_index] = large_end
                    large_batch.num_tokens[large_req_index] = large_end
                self.large_model_runner.requests[req_id].output_token_ids.append(
                    accepted_token
                )
            request = self.model_runner.requests[req_id]
            large_request = self.large_model_runner.requests.get(req_id)
            self.dual_trace.write({
                "request_id": req_id,
                "output_index": state.output_index,
                "route": ROUTE_NAMES[plan.route],
                "history_len_before": len(history),
                "small_before": plan.small_before,
                "small_delta": plan.small_delta,
                "small_after": state.small_num_computed_tokens,
                "large_before": plan.large_before,
                "large_delta": plan.large_delta,
                "large_input_token_ids": (
                    history[plan.large_before : plan.history_len_before]
                    if sync_large or plan.route == LARGE
                    else []
                ),
                "large_after": state.large_num_computed_tokens,
                "small_argmax": small_argmax,
                "large_argmax": large_argmax,
                "selected_argmax": selected_argmax,
                "accepted_token": accepted_token,
                "small_block_ids": list(request.block_ids[0]),
                "large_block_ids": (
                    list(large_request.block_ids[0])
                    if large_request is not None else []
                ),
                "global_row": global_row,
                "large_row": (
                    self.large_model_runner.input_batch.req_id_to_index[req_id]
                    if req_id in large_to_global_row else None
                ),
                "large_sub_batch_size": len(large_req_ids),
                "small_allocator": "small",
                "large_allocator": "large",
                "small_total_blocks": scheduled_step.small_total_blocks,
                "large_total_blocks": scheduled_step.large_total_blocks,
                "small_used_blocks": scheduled_step.small_used_blocks,
                "large_used_blocks": scheduled_step.large_used_blocks,
                "small_kv_ptr": small_kv_ptr,
                "large_kv_ptr": large_kv_ptr,
            })
            if self.grpo_router is not None:
                self.grpo_router.commit(req_id, state.output_index, plan.route)
            state.commit_sample()
        self.pending_dual_steps = {}
        return output
