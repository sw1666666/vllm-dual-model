"""Choose a GRPO threshold on a sample from the benchmark’s 128-token synthetic-input family."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from unittest.mock import patch

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / "benchmarks"))
from grpo_batch_experiment import pool_size
from vllm import LLM, SamplingParams
from vllm_dual.config import DualModelConfig
from vllm_dual.grpo_router import GrpoRouter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    small = args.models / "Qwen3-0.6B"
    large = args.models / "Qwen3-8B"
    batch, length = 128, 160
    small_bytes = pool_size(small, batch, length)
    large_bytes = pool_size(large, batch, length)
    config = DualModelConfig(enabled=True, mode="lazy", large_model=str(large),
                             sync_forward_enabled=True, trace_path=None,
                             router_checkpoint=str(args.models / "grpo_switch_policy_2600.pt"),
                             small_kv_cache_bytes=small_bytes, large_kv_cache_bytes=large_bytes)
    rng = random.Random(1532)
    inputs = []
    for start in rng.sample(range(1000), 768):
        seq = [1000 + (start + j) % 1000 for j in range(128)]
        inputs.append(seq)
    params = SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True)
    llm = LLM(model=str(small), dtype="half", seed=17, max_model_len=length,
              max_num_seqs=batch, max_num_batched_tokens=4096, block_size=16,
              kv_cache_memory_bytes=small_bytes + large_bytes, enable_chunked_prefill=False,
              enable_prefix_caching=False, async_scheduling=False, generation_config="vllm",
              worker_cls="vllm_dual.worker.DualWorker", scheduler_cls="vllm_dual.scheduler.DualScheduler",
              additional_config={"dual_model": config.to_dict()})
    original_decide = GrpoRouter.decide
    trials = []

    def evaluate(threshold, sequences):
        counts = [0, 0]

        def observe(router, ids, hidden):
            router.log_odds_threshold = math.log(threshold) - math.log1p(-threshold)
            actions = original_decide(router, ids, hidden)
            counts[0] += len(actions)
            counts[1] += sum(actions)
            return actions

        batch_prompts = [{"prompt_token_ids": seq} for seq in sequences]
        with patch.object(GrpoRouter, "decide", observe):
            outputs = llm.generate(batch_prompts, params, use_tqdm=False)
        assert len(outputs) == len(sequences) and all(len(o.outputs[0].token_ids) == 32 for o in outputs)
        assert counts[0] == 32 * len(sequences)
        return dict(threshold=threshold, outputs=counts[0], large_outputs=counts[1], share=counts[1] / counts[0])

    low, high = 0.5, 0.7
    threshold = 0.63
    for _ in range(10):
        trial = evaluate(threshold, inputs)
        share = trial["share"]
        print(trial, flush=True)
        trials.append(trial)
        if abs(share - 0.15) <= 0.002:
            break
        if share > 0.15:
            low = threshold
        else:
            high = threshold
        threshold = round((low + high) / 2, 6)
        if any(t["threshold"] == threshold for t in trials):
            break
    best = min(trials, key=lambda t: abs(t["share"] - 0.15))
    # Check the exact input sets before timing. This engine uses 128 active rows.
    checks = {}
    for target_batch in (128, 256):
        salt = 1000 + target_batch * 100
        seqs = [[1000 + (salt + row * 37 + j) % 1000 for j in range(128)] for row in range(4 * target_batch)]
        checks[target_batch] = evaluate(best["threshold"], seqs)
        print("Input-set check", target_batch, checks[target_batch], flush=True)
    profile = dict(input_set_checks=checks, target=0.15, calibration_seed=1532, requests=len(inputs),
                   input_sha256=hashlib.sha256(json.dumps(inputs, separators=(",", ":")).encode()).hexdigest(),
                   router_large_threshold=best["threshold"], calibration_share=best["share"], trials=trials)
    args.out.write_text(json.dumps(profile, indent=2) + "\n")


if __name__ == "__main__":
    main()
