"""Generate or reproduce the Margin I natural-EOS benchmark on a 96 GB GPU."""
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from .config import DualModelConfig


def run(args):
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")
    from vllm import LLM, SamplingParams
    import torch

    benchmark = args.command == "benchmark"
    if benchmark:
        source = Path(__file__).resolve().parents[1] / "artifacts/margin-v4/questions.json"
        questions = json.loads(source.read_text())
    elif args.prompt is not None:
        questions = [args.prompt]
    else:
        questions = json.loads(args.prompts_json.read_text())
    if not isinstance(questions, list) or not questions or not all(isinstance(q, str) for q in questions):
        raise ValueError("Expected a nonempty JSON list of text prompts")
    config = DualModelConfig(enabled=True, mode="lazy", sync_forward_enabled=True,
                             large_model=str(args.models / "Qwen3-8B"), queue_large=True,
                             margin_v4=True, small_kv_cache_bytes=49 << 29,
                             large_kv_cache_bytes=63 << 29, trace_path=None)
    llm = LLM(model=str(args.models / "Qwen3-0.6B"), dtype="half", seed=17, block_size=16,
              max_num_seqs=512, max_model_len=8192, max_num_batched_tokens=8192,
              enable_chunked_prefill=False, enable_prefix_caching=False, async_scheduling=False,
              kv_cache_memory_bytes=config.small_kv_cache_bytes + config.large_kv_cache_bytes,
              generation_config="vllm", worker_cls="vllm_dual.worker.DualWorker",
              scheduler_cls="vllm_dual.scheduler.DualScheduler",
              additional_config={"dual_model": config.to_dict()})
    tokenizer = llm.get_tokenizer()
    prompts = [{"prompt_token_ids": tokenizer.apply_chat_template(
        [{"role": "user", "content": q}], tokenize=True, add_generation_prompt=True,
        enable_thinking=False)} for q in questions]

    def generate(limit):
        params = SamplingParams(temperature=0, repetition_penalty=1.15, max_tokens=limit,
                                stop_token_ids=[151643, 151645], ignore_eos=False)
        return llm.generate(prompts, params, use_tqdm=False)

    worker = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker
    if benchmark:
        generate(32)
        worker.grpo_router.committed[:] = [0, 0]
    torch.cuda.synchronize()
    start = perf_counter()
    outputs = generate(None if benchmark else args.max_tokens)
    torch.cuda.synchronize()
    seconds = perf_counter() - start
    tokens = [list(o.outputs[0].token_ids) for o in outputs]
    count = sum(map(len, tokens))
    small, large = worker.grpo_router.committed
    assert small + large == count
    result = dict(variant="v4", requests=len(outputs), seconds=seconds, tokens=count,
                  tok_s=count / seconds, large_commits=large,
                  token_digest=hashlib.sha256(json.dumps(tokens).encode()).hexdigest())
    if not benchmark:
        result["outputs"] = [dict(prompt=q, text=o.outputs[0].text, output_token_ids=t)
                             for q, o, t in zip(questions, outputs, tokens)]
    directory = args.output_root / f"{args.command}-v4-{uuid4().hex}"
    directory.mkdir(parents=True)
    (directory / f"{args.command}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))
    print(f"Run directory: {directory}")
