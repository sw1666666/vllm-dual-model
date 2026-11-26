"""Untimed first-batch audit of GRPO decisions, logits and private replay inputs."""
import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / "benchmarks"))
sys.path.insert(0, os.environ.get("DUAL_IMPL_ROOT", str(repo)))
import torch
from vllm import LLM
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm_dual.grpo_router import GrpoRouter
from grpo_batch_experiment import run

parser = argparse.ArgumentParser()
parser.add_argument("--models", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--batch", type=int, required=True)
parser.add_argument("--queue-large", action="store_true")
parser.add_argument("--workload", default="synthetic")
parser.add_argument("--router-large-threshold", type=float, default=0.5)
args = parser.parse_args()
args.mode = "grpo"
args.checkpoint = str(args.models / "grpo_switch_policy_2600.pt")
args.questions = Path("artifacts/grpo-v100/questions.json")
execute = GPUModelRunner.execute_model
decide = GrpoRouter.decide
generate = LLM.generate
rows, history, cursor = {}, {}, {}

def audit_forward(runner, sched, *a, **kw):
    output = execute(runner, sched, *a, **kw)
    if not sched.total_num_scheduled_tokens:
        return output
    small = runner.model_config.model.endswith("0.6B")
    logits = runner.execute_model_state.logits
    values, indices = logits.topk(2, dim=-1)
    tops = indices.tolist()
    vals = values.tolist()
    inputs = runner.input_ids.gpu[:sched.total_num_scheduled_tokens].tolist() if not small else []
    offset = 0
    for i, rid in enumerate(runner.input_batch.req_ids):
        if small:
            req = runner.requests[rid]
            history[rid] = list(req.prompt_token_ids) + list(req.output_token_ids)
            seq = rows.setdefault(rid, [])
            assert len(seq) == len(req.output_token_ids)
            seq.append(dict(small=[tops[i], vals[i]]))
        else:
            n = sched.num_scheduled_tokens[rid]
            before = int(runner.input_batch.num_computed_tokens_cpu[i])
            assert before == cursor.get(rid, 0) and before + n == len(history[rid])
            assert inputs[offset:offset+n] == history[rid][before:]
            offset += n
            cursor[rid] = before + n
            rows[rid][-1]["large"] = [tops[i], vals[i]]
    return output

def audit_decide(router, ids, hidden):
    actions = decide(router, ids, hidden)
    usage = torch.stack([router.usage.get(rid,router.zero) for rid in ids])
    features = torch.cat((hidden.float(), usage), dim=1)
    logits = torch.nn.functional.linear(features, router.weight, router.bias)
    margins = (logits[:,1] - logits[:,0]).tolist()
    for rid,action,margin in zip(ids,actions,margins):
        rows[rid][-1].update(action=action,router_margin=margin)
    return actions

def one_batch(llm,*a,**kw):
    outputs = generate(llm,*a,**kw)
    result = []
    for out in outputs:
        seq = rows[out.request_id]
        tokens = out.outputs[0].token_ids
        assert len(tokens) == len(seq) == 32
        for e,t in zip(seq,tokens):
            e["token"] = t
            assert bool(e["action"]) == ("large" in e)
        result.append(seq)
    (args.out/"audit.json").write_text(json.dumps(result))
    print("Verified every private input ID and contiguous large KV cursor.",flush=True)
    # Stop after this independent validation submission; it produces no timing.
    raise SystemExit(0)

with patch.object(GPUModelRunner,"execute_model",audit_forward), patch.object(GrpoRouter,"decide",audit_decide), patch.object(LLM,"generate",one_batch):
    run(args)
