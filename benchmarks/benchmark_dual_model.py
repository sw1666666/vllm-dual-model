#!/usr/bin/env python3
"""Benchmark one dual-model routing configuration across the M8 matrix."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
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

import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

from vllm_dual.config import DualModelConfig  # noqa: E402
from vllm_dual.router import LARGE, SMALL  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--small-model", required=True)
    parser.add_argument("--large-model", required=True)
    parser.add_argument("--p-large", type=float, required=True)
    parser.add_argument("--route-pattern", choices=("random", "clustered"), required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--small-kv-cache-mib", type=int, default=1120)
    parser.add_argument("--large-kv-cache-mib", type=int, default=1440)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--trace-path", type=Path, required=True)
    return parser.parse_args()


def clustered_routes(p_large: float, length: int = 32) -> list[int]:
    count = round(p_large * length)
    return [LARGE] * count + [SMALL] * (length - count)


def make_prompts(prompt_len: int, batch_size: int, salt: int) -> list[TokensPrompt]:
    prompts = []
    for row in range(batch_size):
        token_ids = [1000 + ((salt + row * 37 + index) % 1000) for index in range(prompt_len)]
        prompts.append(TokensPrompt(prompt_token_ids=token_ids))
    return prompts


def read_trace(path: Path, request_ids: set[str]) -> list[dict]:
    return [
        row
        for row in (
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        )
        if row["request_id"] in request_ids
    ]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def main() -> None:
    args = parse_args()
    forced_routes = (
        clustered_routes(args.p_large)
        if args.route_pattern == "clustered"
        else None
    )
    dual = DualModelConfig(
        enabled=True,
        large_model=args.large_model,
        mode="lazy",
        sync_forward_enabled=True,
        p_large=args.p_large,
        router_seed=17,
        forced_routes=forced_routes,
        small_kv_cache_bytes=args.small_kv_cache_mib << 20,
        large_kv_cache_bytes=args.large_kv_cache_mib << 20,
        trace_path=str(args.trace_path),
    )
    llm = LLM(
        model=args.small_model,
        worker_cls="vllm_dual.worker.DualWorker",
        scheduler_cls="vllm_dual.scheduler.DualScheduler",
        additional_config={"dual_model": dual.to_dict()},
        dtype="half",
        enforce_eager=True,
        max_model_len=256,
        max_num_seqs=4,
        max_num_batched_tokens=256,
        block_size=16,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        async_scheduling=False,
        seed=17,
        generation_config="vllm",
        kv_cache_memory_bytes=(
            (args.small_kv_cache_mib + args.large_kv_cache_mib) << 20
        ),
    )

    warmup_params = SamplingParams(temperature=0.0, max_tokens=4)
    for index in range(args.warmup):
        llm.generate(make_prompts(32, 4, index), warmup_params, use_tqdm=False)

    results = []
    case_index = 0
    for prompt_len in (32, 64, 128):
        for output_len in (16, 32):
            for batch_size in (1, 2, 4):
                repeat_rows = []
                for repeat in range(args.repeats):
                    prompts = make_prompts(prompt_len, batch_size, 10000 + case_index * 100 + repeat)
                    params = SamplingParams(temperature=0.0, max_tokens=output_len)
                    torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    outputs = llm.generate(prompts, params, use_tqdm=False)
                    torch.cuda.synchronize()
                    latency = time.perf_counter() - started
                    accepted_tokens = sum(
                        len(output.outputs[0].token_ids) for output in outputs
                    )
                    request_ids = {output.request_id for output in outputs}
                    rows = read_trace(args.trace_path, request_ids)
                    assert len(rows) == accepted_tokens
                    small_forward_tokens = sum(row["small_delta"] for row in rows)
                    large_forward_tokens = sum(row["large_delta"] for row in rows)
                    large_route_count = sum(row["route"] == "large" for row in rows)
                    repeat_rows.append(
                        {
                            "repeat": repeat,
                            "accepted_output_tokens": accepted_tokens,
                            "latency_seconds": latency,
                            "accepted_output_tokens_per_second": accepted_tokens / latency,
                            "small_forward_tokens": small_forward_tokens,
                            "large_forward_tokens": large_forward_tokens,
                            "large_route_count": large_route_count,
                            "large_catchup_tokens": large_forward_tokens - large_route_count,
                            "large_forward_tokens_per_accepted_output_token": (
                                large_forward_tokens / accepted_tokens
                            ),
                            "peak_small_used_blocks": max(row["small_used_blocks"] for row in rows),
                            "peak_large_used_blocks": max(row["large_used_blocks"] for row in rows),
                            "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                            "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                        }
                    )
                latencies = [row["latency_seconds"] for row in repeat_rows]
                throughputs = [
                    row["accepted_output_tokens_per_second"] for row in repeat_rows
                ]
                results.append(
                    {
                        "prompt_length": prompt_len,
                        "output_length": output_len,
                        "batch_size": batch_size,
                        "p_large": args.p_large,
                        "route_pattern": args.route_pattern,
                        "latency_seconds_mean": statistics.mean(latencies),
                        "latency_seconds_p50": percentile(latencies, 0.50),
                        "latency_seconds_p95": percentile(latencies, 0.95),
                        "accepted_output_tokens_per_second_mean": statistics.mean(throughputs),
                        "repeats": repeat_rows,
                    }
                )
                args.output_json.parent.mkdir(parents=True, exist_ok=True)
                args.output_json.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "warmup": args.warmup,
                            "repeat_count": args.repeats,
                            "small_kv_cache_mib": args.small_kv_cache_mib,
                            "large_kv_cache_mib": args.large_kv_cache_mib,
                            "forced_routes": forced_routes,
                            "results": results,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                case_index += 1
                print(
                    f"case={case_index}/18 prompt={prompt_len} output={output_len} "
                    f"batch={batch_size} tok_s={statistics.mean(throughputs):.3f}"
                )


if __name__ == "__main__":
    main()
