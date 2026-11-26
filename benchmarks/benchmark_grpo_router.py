"""Compare same-step GRPO routing with native Qwen3 and immediate hash routing."""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")

if os.environ.get("DUAL_IMPL_ROOT"):
    sys.path.insert(0, os.environ["DUAL_IMPL_ROOT"])

import torch
from vllm import LLM, SamplingParams

from vllm_dual.config import DualModelConfig, add_grpo_threshold_args


def replay_policy(trace):
    """Bypass the online head for an identical-action timing control (one worker)."""
    actions = defaultdict(list)
    with trace.open() as stream:
        for event in map(json.loads, stream):
            sequence = actions[event["request_id"]]
            assert event["output_index"] == len(sequence)
            sequence.append(int(event["route"] == "large"))
    cursors = defaultdict(int)

    def decide(_router, req_ids, _hidden):
        return [actions[rid][cursors[rid]] for rid in req_ids]

    def commit(_router, rid, _step, _action):
        cursors[rid] += 1

    return patch.multiple("vllm_dual.grpo_router.GrpoRouter", decide=decide, commit=commit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("small", "large", "hash", "grpo", "replay"))
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=Path("artifacts/grpo-v100/questions.json"))
    parser.add_argument("--batch", type=int, help="Run the fixed-request large-batch experiment")
    parser.add_argument("--workload", choices=("synthetic", "gsm8k", "check"), default="synthetic")
    parser.add_argument("--queue-large", action="store_true",
                        help="Use the V2 pending queue in an online GRPO run")
    add_grpo_threshold_args(parser)
    args = parser.parse_args()
    if args.batch is not None:
        if __package__:
            from .grpo_batch_experiment import run
        else:
            from grpo_batch_experiment import run
        return run(args)
    if args.workload != "synthetic":
        parser.error("workload selection requires --batch")
    if args.queue_large and args.mode != "grpo":
        parser.error("--queue-large requires grpo mode")
    learned = args.mode in {"grpo", "replay"}
    if learned and not args.checkpoint:
        parser.error("grpo requires --checkpoint")
    args.out.mkdir(parents=True, exist_ok=True)
    trace = args.out / f"{args.mode}.trace.jsonl"
    small = str(args.models / "Qwen3-0.6B")
    large = str(args.models / "Qwen3-8B")
    dual = learned or args.mode == "hash"
    options = {}
    if dual:
        config = DualModelConfig(
            enabled=True, large_model=large, mode="lazy", sync_forward_enabled=True,
            router_large_threshold=args.router_large_threshold,
            p_large=0.1, router_checkpoint=args.checkpoint if learned else None,
            trace_path=str(trace),
            **({"queue_large": True} if args.queue_large else {}),
        )
        options.update(worker_cls="vllm_dual.worker.DualWorker",
                       scheduler_cls="vllm_dual.scheduler.DualScheduler",
                       additional_config={"dual_model": config.to_dict()})
    if args.mode == "replay":
        original = json.loads((args.out / "grpo.json").read_text())
        if any(r.get("router_large_threshold", 0.5) != args.router_large_threshold for r in original):
            parser.error("Replay must use the threshold recorded by the GRPO run")
        control = replay_policy(args.out / "grpo.trace.jsonl")
        control.start()
    llm = LLM(model=large if args.mode == "large" else small, dtype="half", seed=17,
              max_model_len=256, max_num_batched_tokens=512, max_num_seqs=32, block_size=16,
              kv_cache_memory_bytes=2560 << 20, enable_chunked_prefill=False,
              async_scheduling=False, enable_prefix_caching=False, generation_config="vllm",
              **options)
    tok = llm.get_tokenizer()
    questions = json.loads(args.questions.read_text())
    math_ids = [tok.apply_chat_template([{"role": "user", "content": q}],
                                        tokenize=True, add_generation_prompt=True,
                                        enable_thinking=False) for q in questions]
    assert len(math_ids) >= 96 and max(map(len, math_ids)) + 32 <= 256
    params = SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True)
    runs = []
    for workload, batch in [("synthetic", b) for b in (1, 8, 16, 32)] + [("gsm8k", 32)]:
        for rep in (-1, 0, 1, 2):
            if workload == "synthetic":
                salt = 0 if rep < 0 else 1000 + batch * 100 + rep
                ids = [[1000 + (salt + row*37 + j) % 1000 for j in range(128)]
                       for row in range(batch)]
            else:
                start = max(0, rep) * batch
                ids = math_ids[start:start + batch]
            prompts = [{"prompt_token_ids": row} for row in ids]
            torch.cuda.synchronize()
            started = time.perf_counter()
            outputs = llm.generate(prompts, params, use_tqdm=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            assert len(outputs) == batch and all(len(o.outputs[0].token_ids) == 32 for o in outputs)
            if rep < 0:
                continue
            requests = [dict(id=o.request_id, prompt=list(o.prompt_token_ids),
                             tokens=list(o.outputs[0].token_ids)) for o in outputs]
            runs.append(dict(workload=workload, batch=batch, repeat=rep, seconds=elapsed,
                             tok_s=batch*32/elapsed, requests=requests))
            print(workload, batch, rep, round(batch*32/elapsed, 2), flush=True)
    if dual:
        events = defaultdict(list)
        with trace.open() as stream:
            for event in map(json.loads, stream):
                events[event["request_id"]].append(event)
        for run in runs:
            wide = catchup = 0
            for req in run["requests"]:
                records = events[req["id"]]
                assert [r["output_index"] for r in records] == list(range(32))
                assert [r["accepted_token"] for r in records] == req["tokens"]
                history = req["prompt"][:]
                end = 0
                req["actions"] = []
                for r in records:
                    action = int(r["route"] == "large")
                    req["actions"].append(action)
                    assert r["large_before"] == end
                    if action:
                        assert r["large_input_token_ids"] == history[end:]
                        assert r["accepted_token"] == r["large_argmax"]
                        end = len(history)
                        catchup += r["large_delta"] - 1
                    else:
                        assert r["large_delta"] == 0 and r["accepted_token"] == r["small_argmax"]
                    assert r["large_after"] == end
                    history.append(r["accepted_token"])
                    wide += action
            run.update(large_outputs=wide, large_history_tokens=catchup)
    if learned:
        for run in runs:
            run["router_large_threshold"] = args.router_large_threshold
    if args.mode == "replay":
        control.stop()
        assert len(original) == len(runs)
        for ref, run in zip(original, runs):
            assert ref["requests"] == run["requests"]
    (args.out / f"{args.mode}.json").write_text(json.dumps(runs) + "\n")


if __name__ == "__main__":
    main()
