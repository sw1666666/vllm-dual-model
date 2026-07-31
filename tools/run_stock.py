#!/usr/bin/env python3
"""Run a deterministic stock-vLLM baseline on the V100 test host."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

# These must be fixed before importing vLLM.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")

from vllm import LLM, SamplingParams  # noqa: E402


class GPUMemoryMonitor:
    """Poll total device memory usage, including the vLLM worker."""

    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "GPUMemoryMonitor":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=2,
                )
                self.peak_mib = max(
                    self.peak_mib,
                    *(int(line.strip()) for line in output.splitlines()),
                )
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self._stop.wait(self.interval_s)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.perf_counter()
    with GPUMemoryMonitor() as gpu_monitor:
        llm = LLM(
            model=args.model,
            dtype="half",
            enforce_eager=True,
            max_model_len=256,
            max_num_seqs=1,
            max_num_batched_tokens=256,
            block_size=16,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            async_scheduling=False,
            seed=17,
            generation_config="vllm",
        )
        request_output = llm.generate(
            [args.prompt],
            SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
        )[0]

    completion = request_output.outputs[0]
    result = {
        "model": args.model,
        "prompt": args.prompt,
        "prompt_token_ids": list(request_output.prompt_token_ids),
        "output_token_ids": list(completion.token_ids),
        "text": completion.text,
        "peak_gpu_memory_mib": gpu_monitor.peak_mib,
        "elapsed_seconds_including_load": time.perf_counter() - started_at,
        "config": {
            "dtype": "half",
            "enforce_eager": True,
            "max_model_len": 256,
            "max_num_seqs": 1,
            "max_num_batched_tokens": 256,
            "block_size": 16,
            "enable_prefix_caching": False,
            "enable_chunked_prefill": False,
            "async_scheduling": False,
            "seed": 17,
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
            "attention_backend": os.environ["VLLM_ATTENTION_BACKEND"],
            "v1_multiprocessing": os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"],
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
