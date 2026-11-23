#!/usr/bin/env python3
"""Comparable V100 benchmark for small-only, large-only, and dual-sync."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")

import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from vllm_dual.config import DualModelConfig
from vllm_dual.router import LARGE, SMALL


def prompts(batch_size: int, salt: int) -> list[TokensPrompt]:
    return [
        TokensPrompt(
            prompt_token_ids=[1000 + ((salt + row * 37 + i) % 1000) for i in range(128)]
        )
        for row in range(batch_size)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("small", "large", "dual-sync", "dual-lazy"), required=True
    )
    parser.add_argument("--small-model", required=True)
    parser.add_argument("--large-model", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--trace-path", type=Path)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--p-large", type=float, default=0.10)
    parser.add_argument("--large-min-batch-size", type=int, default=4)
    parser.add_argument("--large-max-wait-steps", type=int, default=4)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    args = parser.parse_args()

    common = dict(
        dtype="half",
        enforce_eager=args.enforce_eager,
        max_model_len=256,
        max_num_seqs=max(args.batch_sizes),
        max_num_batched_tokens=512,
        block_size=16,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        async_scheduling=False,
        seed=17,
        generation_config="vllm",
        kv_cache_memory_bytes=2560 << 20,
    )
    if args.mode.startswith("dual-"):
        if args.trace_path is None:
            raise ValueError("dual modes require --trace-path")
        args.trace_path.parent.mkdir(parents=True, exist_ok=True)
        args.trace_path.write_text("", encoding="utf-8")
        dual = DualModelConfig(
            enabled=True,
            large_model=args.large_model,
            mode="sync" if args.mode == "dual-sync" else "lazy",
            sync_forward_enabled=True,
            p_large=args.p_large,
            forced_routes=(
                [SMALL, LARGE] * 32 if args.mode == "dual-sync" else None
            ),
            large_min_batch_size=args.large_min_batch_size,
            large_max_wait_steps=args.large_max_wait_steps,
            small_kv_cache_bytes=1120 << 20,
            large_kv_cache_bytes=1440 << 20,
            trace_path=str(args.trace_path),
        )
        llm = LLM(
            model=args.small_model,
            worker_cls="vllm_dual.worker.DualWorker",
            scheduler_cls="vllm_dual.scheduler.DualScheduler",
            additional_config={"dual_model": dual.to_dict()},
            **common,
        )
    else:
        llm = LLM(
            model=args.small_model if args.mode == "small" else args.large_model,
            **common,
        )

    params = SamplingParams(temperature=0.0, max_tokens=32)
    llm.generate(prompts(max(args.batch_sizes), 0), params, use_tqdm=False)
    rows = []
    for batch_size in args.batch_sizes:
        repeats = []
        for repeat in range(3):
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            outputs = llm.generate(
                prompts(batch_size, 1000 + batch_size * 100 + repeat),
                params,
                use_tqdm=False,
            )
            torch.cuda.synchronize()
            latency = time.perf_counter() - started
            output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            repeats.append(
                {
                    "latency_seconds": latency,
                    "output_tokens": output_tokens,
                    "output_tokens_per_second": output_tokens / latency,
                    "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                }
            )
        rows.append(
            {
                "batch_size": batch_size,
                "prompt_tokens_per_request": 128,
                "output_tokens_per_request": 32,
                "latency_seconds_mean": statistics.mean(r["latency_seconds"] for r in repeats),
                "output_tokens_per_second_mean": statistics.mean(
                    r["output_tokens_per_second"] for r in repeats
                ),
                "peak_gpu_allocated_gib_max": max(r["peak_gpu_allocated_gib"] for r in repeats),
                "peak_gpu_reserved_gib_max": max(r["peak_gpu_reserved_gib"] for r in repeats),
                "repeats": repeats,
            }
        )
        print(args.mode, rows[-1], flush=True)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps({"mode": args.mode, "results": rows}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
