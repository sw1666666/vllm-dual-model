#!/usr/bin/env python3
"""Single entry point for the staged dual-model V100 experiment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")

from vllm import LLM, SamplingParams  # noqa: E402

from vllm_dual.config import DualModelConfig  # noqa: E402
from vllm_dual.router import LARGE, SMALL  # noqa: E402
from vllm_dual.trace import JsonlTraceWriter  # noqa: E402

DEFAULT_PROMPT = "Give one concise fact about the Moon. Answer:"


def parse_forced_routes(value: str | None) -> list[int] | None:
    if value is None:
        return None
    routes = []
    for item in value.replace(",", " ").split():
        normalized = item.strip().lower()
        if normalized in {"s", "small", "0"}:
            routes.append(SMALL)
        elif normalized in {"l", "large", "1"}:
            routes.append(LARGE)
        else:
            raise ValueError(f"unknown forced route: {item}")
    return routes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--small-model", required=True)
    parser.add_argument("--large-model", required=True)
    parser.add_argument("--mode", choices=("disabled", "sync", "lazy"), default="disabled")
    parser.add_argument("--p-large", type=float, default=0.10)
    parser.add_argument("--router-seed", type=int, default=17)
    parser.add_argument("--forced-routes")
    parser.add_argument("--trace-path", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompts-json", type=Path)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--max-tokens-json", type=Path)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--kv-cache-memory-bytes", type=int)
    parser.add_argument("--small-kv-cache-bytes", type=int, default=1120 << 20)
    parser.add_argument("--large-kv-cache-bytes", type=int, default=1440 << 20)
    parser.add_argument("--enable-dual-forward", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    forced_routes = parse_forced_routes(args.forced_routes)
    config = DualModelConfig(
        enabled=args.mode != "disabled",
        large_model=args.large_model,
        router_seed=args.router_seed,
        p_large=args.p_large,
        mode=args.mode,
        sync_forward_enabled=args.enable_dual_forward,
        forced_routes=forced_routes,
        small_kv_cache_bytes=args.small_kv_cache_bytes,
        large_kv_cache_bytes=args.large_kv_cache_bytes,
        trace_path=str(args.trace_path),
    )
    llm_kwargs = dict(
        model=args.small_model,
        worker_cls="vllm_dual.worker.DualWorker",
        scheduler_cls="vllm_dual.scheduler.DualScheduler",
        additional_config={"dual_model": config.to_dict()},
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
    )
    outer_kv_cache_memory_bytes = args.kv_cache_memory_bytes
    if outer_kv_cache_memory_bytes is None and args.mode != "disabled":
        outer_kv_cache_memory_bytes = (
            args.small_kv_cache_bytes + args.large_kv_cache_bytes
        )
    if outer_kv_cache_memory_bytes is not None:
        llm_kwargs["kv_cache_memory_bytes"] = outer_kv_cache_memory_bytes
    llm = LLM(**llm_kwargs)
    prompts = [args.prompt]
    if args.prompts_json is not None:
        prompts = json.loads(args.prompts_json.read_text(encoding="utf-8"))
        if not isinstance(prompts, list) or not all(
            isinstance(prompt, str) for prompt in prompts
        ):
            raise ValueError("--prompts-json must contain a JSON list of strings")
    sampling_params: SamplingParams | list[SamplingParams]
    if args.max_tokens_json is None:
        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=args.max_tokens
        )
    else:
        max_tokens = json.loads(args.max_tokens_json.read_text(encoding="utf-8"))
        if (
            not isinstance(max_tokens, list)
            or len(max_tokens) != len(prompts)
            or not all(isinstance(value, int) and value > 0 for value in max_tokens)
        ):
            raise ValueError(
                "--max-tokens-json must contain one positive integer per prompt"
            )
        sampling_params = [
            SamplingParams(temperature=0.0, max_tokens=value)
            for value in max_tokens
        ]
    request_outputs = llm.generate(prompts, sampling_params)
    result = {
        "mode": args.mode,
        "small_model": args.small_model,
        "large_model_configured": args.large_model,
        "large_model_loaded": args.mode != "disabled",
        "large_model_forward_enabled": False,
        "kv_cache_memory_bytes": outer_kv_cache_memory_bytes,
        "small_kv_cache_bytes": args.small_kv_cache_bytes,
        "large_kv_cache_bytes": args.large_kv_cache_bytes,
        "worker_cls": "vllm_dual.worker.DualWorker",
        "scheduler_cls": "vllm_dual.scheduler.DualScheduler",
        "prompt": prompts[0] if len(prompts) == 1 else None,
        "prompts": prompts,
        "max_num_seqs": args.max_num_seqs,
        "prompt_token_ids": list(request_outputs[0].prompt_token_ids),
        "output_token_ids": list(request_outputs[0].outputs[0].token_ids),
        "text": request_outputs[0].outputs[0].text,
        "outputs": [
            {
                "request_id": request_output.request_id,
                "prompt": prompt,
                "prompt_token_ids": list(request_output.prompt_token_ids),
                "output_token_ids": list(request_output.outputs[0].token_ids),
                "text": request_output.outputs[0].text,
            }
            for prompt, request_output in zip(prompts, request_outputs)
        ],
        "forced_routes": forced_routes,
    }
    result["large_model_forward_enabled"] = args.enable_dual_forward
    if not args.enable_dual_forward:
        JsonlTraceWriter(args.trace_path).write(result)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
