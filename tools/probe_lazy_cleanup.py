#!/usr/bin/env python3
"""Run many short requests through one lazy dual-model engine."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")

from vllm import LLM, SamplingParams  # noqa: E402
import torch  # noqa: E402

from vllm_dual.config import DualModelConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--small-model", required=True)
    parser.add_argument("--large-model", required=True)
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--trace-path", required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    dual = DualModelConfig(
        enabled=True,
        large_model=args.large_model,
        mode="lazy",
        sync_forward_enabled=True,
        p_large=0.10,
        router_seed=17,
        trace_path=args.trace_path,
    )
    llm = LLM(
        model=args.small_model,
        worker_cls="vllm_dual.worker.DualWorker",
        scheduler_cls="vllm_dual.scheduler.DualScheduler",
        additional_config={"dual_model": dual.to_dict()},
        dtype="half",
        enforce_eager=True,
        max_model_len=256,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=256,
        block_size=16,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        async_scheduling=False,
        seed=17,
        generation_config="vllm",
        kv_cache_memory_bytes=(512 << 20) + (2 << 30),
    )
    prompts = [
        f"Give one concise fact about the Moon. Answer: request {i}"
        for i in range(args.num_requests)
    ]
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
        use_tqdm=False,
    )
    assert len(outputs) == args.num_requests
    assert all(len(output.outputs[0].token_ids) == args.max_tokens for output in outputs)
    scheduler = llm.llm_engine.engine_core.engine_core.scheduler
    small_free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    large_free = scheduler.large_kv_cache_manager.block_pool.get_num_free_blocks()
    assert small_free == scheduler.small_initial_free_blocks
    assert large_free == scheduler.large_initial_free_blocks
    print(f"completed_requests={len(outputs)} max_tokens={args.max_tokens}")
    print(f"final_free_blocks small={small_free} large={large_free}")
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "num_requests": len(outputs),
            "max_tokens": args.max_tokens,
            "max_num_seqs": args.max_num_seqs,
            "small_initial_free_blocks": scheduler.small_initial_free_blocks,
            "small_final_free_blocks": small_free,
            "large_initial_free_blocks": scheduler.large_initial_free_blocks,
            "large_final_free_blocks": large_free,
            "gpu_allocated_gib": torch.cuda.memory_allocated() / 2**30,
            "gpu_reserved_gib": torch.cuda.memory_reserved() / 2**30,
        }
        args.output_json.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
