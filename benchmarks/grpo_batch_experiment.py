"""Fixed-input, replenished-batch comparison for the single-GPU GRPO worker."""
import hashlib
import json
import statistics
import time
from collections import defaultdict
from contextlib import ExitStack
from unittest.mock import patch

import torch
from vllm import LLM, SamplingParams
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.core.sched.scheduler import Scheduler
from vllm_dual.config import DualModelConfig
from vllm_dual.trace import JsonlTraceWriter


def digest(data):
    return hashlib.sha256(json.dumps(data, separators=(",", ":")).encode()).hexdigest()


def pool_size(path, batch, length):
    cfg = json.loads((path / "config.json").read_text())
    block_bytes = 16 * cfg["num_hidden_layers"] * 2 * cfg["num_key_value_heads"] * cfg["head_dim"] * 2
    return (batch * ((length + 15) // 16) + 2) * block_bytes


def run(args):
    batch = args.batch
    check = args.workload == "check"
    if batch < 1 or args.mode not in ("large", "grpo"):
        raise ValueError("Use a positive batch and mode large or grpo")
    if args.mode == "grpo" and not args.checkpoint:
        raise ValueError("GRPO requires --checkpoint")
    if args.mode != "grpo" and (check or args.queue_large):
        raise ValueError("Token alignment and the pending queue require GRPO mode")
    args.out.mkdir(parents=True, exist_ok=False)
    small, large = (args.models / name for name in ("Qwen3-0.6B", "Qwen3-8B"))
    length = 256 if args.workload == "gsm8k" else 160
    step_budget = max(length, 32 * batch)
    small_bytes = pool_size(small, batch, length)
    large_bytes = pool_size(large, batch, length)
    dual = args.mode == "grpo"
    trace = args.out / "check.jsonl"
    options = {}
    if dual:
        extra = {"queue_large": True} if args.queue_large else {}
        if args.merge_route_readback:
            extra["merge_route_readback"] = True
        cfg = DualModelConfig(enabled=True, large_model=str(large), mode="lazy",
                              sync_forward_enabled=True, router_checkpoint=args.checkpoint,
                              router_large_threshold=args.router_large_threshold,
                              small_kv_cache_bytes=small_bytes, large_kv_cache_bytes=large_bytes,
                              trace_path=str(trace) if check else None, **extra)
        options.update(worker_cls="vllm_dual.worker.DualWorker", scheduler_cls="vllm_dual.scheduler.DualScheduler",
                       additional_config={"dual_model": cfg.to_dict()})
    llm = LLM(model=str(small if dual else large), dtype="half", seed=17, block_size=16,
              max_model_len=length, max_num_seqs=batch, max_num_batched_tokens=step_budget,
              kv_cache_memory_bytes=(small_bytes if dual else 0) + large_bytes,
              enable_chunked_prefill=False, enable_prefix_caching=False, async_scheduling=False,
              generation_config="vllm", **options)
    if args.workload == "gsm8k":
        questions = json.loads(args.questions.read_text())[:96]
        assert len(questions) == batch == 96
        ids = [llm.get_tokenizer().apply_chat_template([{"role": "user", "content": q}],
               tokenize=True, add_generation_prompt=True, enable_thinking=False) for q in questions]
    else:
        count = batch if check else 4 * batch
        salt = 1000 + batch * 100
        ids = [[1000 + (salt + row * 37 + j) % 1000 for j in range(128)] for row in range(count)]
    assert max(map(len, ids)) + 32 <= length
    prompts = [{"prompt_token_ids": seq} for seq in ids]
    params = SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True)
    original_execute = GPUModelRunner.execute_model
    original_schedule = Scheduler.schedule
    stats = {}

    def observe_forward(runner, scheduler_output, *a, **kw):
        n = scheduler_output.num_scheduled_tokens
        if n and runner.model_config.model == str(large):
            stats["large_calls"] += 1
            if dual:
                stats["large_outputs"] += len(n)
                stats["large_history_tokens"] += scheduler_output.total_num_scheduled_tokens - len(n)
        if not dual or runner.model_config.model == str(small):
            stats["peak_decode_rows"] = max(stats["peak_decode_rows"], sum(v == 1 for v in n.values()))
        return original_execute(runner, scheduler_output, *a, **kw)

    def observe_schedule(scheduler):
        out = original_schedule(scheduler)
        stats["peak_active_decode"] = max(stats["peak_active_decode"],
                                          sum(req.num_output_tokens > 0 for req in scheduler.running))
        return out

    records = []
    with ExitStack() as stack:
        stack.enter_context(patch.object(GPUModelRunner, "execute_model", observe_forward))
        stack.enter_context(patch.object(Scheduler, "schedule", observe_schedule))
        if not check:
            stack.enter_context(patch.object(JsonlTraceWriter, "write", lambda self, row: None))
        for rep in ([0] if check else [-1, 0, 1, 2]):
            stats = dict(large_calls=0, large_outputs=0, large_history_tokens=0,
                         peak_decode_rows=0, peak_active_decode=0)
            torch.cuda.synchronize()
            start = None if check else time.perf_counter()
            outputs = llm.generate(prompts, params, use_tqdm=False)
            torch.cuda.synchronize()
            elapsed = None if check else time.perf_counter() - start
            tokens = [list(o.outputs[0].token_ids) for o in outputs]
            assert len(outputs) == len(ids) and all(len(t) == 32 for t in tokens)
            assert all(list(o.prompt_token_ids) == inp for o, inp in zip(outputs, ids))
            assert stats["peak_active_decode"] == stats["peak_decode_rows"] == batch, stats
            record = dict(rep=rep, output_sha256=digest(tokens), **stats)
            if elapsed is not None:
                record.update(seconds=elapsed, tok_s=len(ids) * 32 / elapsed)
            print(record, flush=True)
            if rep >= 0:
                records.append(record)
            if check:
                events = defaultdict(list)
                for line in trace.read_text().splitlines():
                    e = json.loads(line)
                    assert e["output_index"] == len(events[e["request_id"]])
                    events[e["request_id"]].append((e["route"], e["accepted_token"]))
                ordered = [events[o.request_id] for o in outputs]
                assert [[t for _, t in row] for row in ordered] == tokens
                (args.out / "actions-tokens.json").write_text(json.dumps(ordered))
    result = dict(batch=batch, workload=args.workload, requests=len(ids), input_sha256=digest(ids),
                  mode=args.mode, queue_large=args.queue_large, router_large_threshold=args.router_large_threshold,
                  small_kv_bytes=small_bytes if dual else 0, large_kv_bytes=large_bytes,
                  max_model_len=length, public_token_budget=step_budget, runs=records)
    if dual:
        result["variant"] = "v3" if args.merge_route_readback else ("v2" if args.queue_large else "v1")
    if not check:
        result["median_tok_s"] = statistics.median(r["tok_s"] for r in records)
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
