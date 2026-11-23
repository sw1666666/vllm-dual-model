"""Fixed-cohort, single-GPU v1 replay for the native-KV upper bound."""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"

import torch
from vllm import LLM, SamplingParams
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm_dual.config import DualModelConfig
from vllm_dual.router import DeterministicRouter, LARGE
from vllm_dual.scheduler_output import DualStepMetadata


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def save_rows(path, data):
    with path.open("w") as stream:
        for row in data:
            print(json.dumps(row, separators=(",", ":")), file=stream)


def engine(args, dual, large_only=False):
    extra = {}
    if dual:
        config = DualModelConfig(
            enabled=True, mode="lazy", large_model=args.large, sync_forward_enabled=True,
            trace_path=str(args.out / "record-worker.jsonl") if args.job == "record" else None,
        )
        extra.update(worker_cls="vllm_dual.worker.DualWorker",
                     scheduler_cls="vllm_dual.scheduler.DualScheduler",
                     additional_config={"dual_model": config.to_dict()})
    return LLM(model=args.large if large_only else args.small, dtype="half",
               seed=17, max_model_len=512, max_num_seqs=32, max_num_batched_tokens=1024,
               enable_prefix_caching=False, enable_chunked_prefill=False,
               async_scheduling=False, block_size=16, kv_cache_memory_bytes=2560 << 20,
               generation_config="vllm", **extra)


def record(args):
    llm = engine(args, True)
    tokenizer = llm.get_tokenizer()
    prompts = []
    for item in rows(args.questions)[:100]:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": item["question"] + "\nGive only the numerical answer."}],
            tokenize=True, add_generation_prompt=True)
        prompts.append({"prompt_token_ids": ids})
    outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=48),
                           use_tqdm=False)
    by_id = defaultdict(list)
    for row in rows(args.out / "record-worker.jsonl"):
        by_id[row["request_id"]].append(row)
    trace = []
    for idx, result in enumerate(outputs):
        completion = result.outputs[0]
        steps = by_id[result.request_id]
        assert [s["output_index"] for s in steps] == list(range(len(steps)))
        assert [s["accepted_token"] for s in steps] == list(completion.token_ids)
        trace.append(dict(request_id=result.request_id, gsm8k_train_index=idx,
                          prompt_ids=list(result.prompt_token_ids), output_ids=list(completion.token_ids),
                          route_for_output=["L" if s["route"] == "large" else "S" for s in steps],
                          route_source="online:DeterministicRouter/blake2b(seed=17,p_large=0.1)",
                          stop_reason=completion.finish_reason, stop_token=completion.stop_reason,
                          groups={str(b):divmod(idx,b) for b in (1,8,32)}))
    save_rows(args.out / "trace.jsonl", trace)


def run_check(args, replay, trace):
    patterns = ["SSSLLSLSL","SLSLSL"]
    selected = [r for r in trace if len(r["output_ids"]) >= 9][:2]
    fixture = [dict(r,output_ids=r["output_ids"][:len(pattern)],route_for_output=pattern)
               for r,pattern in zip(selected,patterns)]
    result, logits = replay.cohort(fixture,check=True)
    if args.mode == "large":
        _, teacher = replay.cohort(fixture,check=True,teacher=True)
        torch.save((logits,teacher),args.out/"native-check.pt")
        return
    expected,teacher = torch.load(args.out/"native-check.pt",map_location="cpu")
    events = result["requests"]
    assert len(events) == sum(len(r["output_ids"])-1 for r in fixture)
    keys = [(r["request_id"],r["step"]) for r in events if r["action"]=="L"]
    assert set(keys)==set(logits)
    expected_gaps = sum(r["gap"] for r in events if r["action"]=="L")
    actual = sum(s["catchup"] for s in result["steps"])
    assert actual == (expected_gaps if args.mode=="routed_exact" else 0)
    def compare(reference,candidate):
        errors = {k:(reference[k]-candidate[k]).abs().max().item() for k in keys}
        worst = max(errors,key=errors.get)
        return dict(max_abs=errors[worst],worst_request=worst[0],
                    worst_pos=len(next(r for r in fixture if r["request_id"]==worst[0])["prompt_ids"])+worst[1]-1,
                    top1=sum(int(reference[k].argmax()==candidate[k].argmax()) for k in keys)/len(keys))
    calibration = compare(expected,teacher)
    tolerance = max(calibration["max_abs"], torch.finfo(torch.float16).eps)
    checks = []
    def add(label,a,b,tol):
        row = dict(kind=label,n=len(keys),tolerance=tol,**compare(a,b))
        row["passed"] = row["max_abs"]<=tol and row["top1"]>=calibration["top1"]
        checks.append(row)
    add("native_decode_vs_teacher",expected,teacher,tolerance)
    add("native_decode_vs_"+args.mode,expected,logits,tolerance)
    if args.mode=="routed_exact":
        torch.save(logits,args.out/"exact-check.pt")
    else:
        add("exact_vs_free",torch.load(args.out/"exact-check.pt"),logits,2*tolerance)
        disturbed = {}
        for t in sorted({t for _,t in keys}):
            _, altered = replay.cohort(fixture,check=True,poison_at=t)
            disturbed.update({k:v for k,v in altered.items() if k[1]==t})
        add("free_vs_future_slots_changed",logits,disturbed,0)
    for row in checks:
        row.update(mode=args.mode,logical_gap=expected_gaps,actual_history=actual,
                   current_positions=len(keys),invariants=True,
                   forced_fixture_router_disagreements=sum(r["action"]!=r["predicted"] for r in events))
    save_rows(args.out/(args.mode+"-checks.jsonl"), checks)
    if not all(r["passed"] for r in checks):
        raise ValueError(checks)


def run(args):
    trace = rows(args.out/"trace.jsonl")[:args.limit]
    replay = Replay(args)
    if args.job=="check":
        return run_check(args,replay,trace)
    for mode in ("routed_exact","routed_free_kv"):
        assert all(x["passed"] for x in rows(args.out/(mode+"-checks.jsonl")))
    for b in args.sizes:
        cohorts = [trace[i:i+b] for i in range(0,len(trace),b)]
        warm_start = time.perf_counter()
        for cohort in cohorts:
            replay.cohort(cohort)
        warm_s = time.perf_counter()-warm_start
        records = []
        for idx,cohort in enumerate(cohorts):
            result,_ = replay.cohort(cohort)
            result.update(run=args.job,mode=args.mode,B=b,repeat=args.repeat,cohort=idx,warmup_s=warm_s)
            records.append(result)
        save_rows(args.out/f"{args.job}-{args.mode}-b{b}-r{args.repeat}.jsonl",records)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=["record","check","time","diag"])
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--small", default="/root/autodl-tmp/models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--large", default="/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--mode", choices=["small","large","routed_exact","routed_free_kv"])
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1,8,32])
    return parser.parse_args()


def schedule_new(key, ids, blocks):
    out = SchedulerOutput.make_empty()
    out.num_common_prefix_blocks = [0]
    out.scheduled_new_reqs = [NewRequestData(key, ids, [], SamplingParams(temperature=0),
                                             None, (blocks,), 0, None)]
    out.num_scheduled_tokens = {key:len(ids)}
    out.total_num_scheduled_tokens = len(ids)
    return out


def forget(runner, keys):
    out = SchedulerOutput.make_empty()
    out.finished_req_ids = set(keys)
    runner.execute_model(out)


def commit(runner, key, token, sampled):
    batch = runner.input_batch
    state = runner.requests[key]
    if sampled:
        state.output_token_ids[-1] = token
    else:
        state.output_token_ids.append(token)
    i = batch.req_id_to_index.get(key)
    if i is not None:
        n = state.num_tokens
        batch.token_ids_cpu[i, n-1] = token
        batch.is_token_ids[i, n-1] = True
        batch.num_tokens_no_spec[i] = batch.num_tokens[i] = n


class Replay:
    def __init__(self, args):
        self.args = args
        self.mode = args.mode
        self.dual = self.mode.startswith("routed")
        self.llm = engine(args, self.dual, self.mode == "large")
        self.worker = self.llm.llm_engine.model_executor.driver_worker.worker
        self.main = self.worker.model_runner
        self.large = self.worker.large_model_runner if self.dual else None
        self.router = DeterministicRouter()
        self.shape = {}
        self.events = []
        self.capture = False
        self.large_actual = {}
        if self.dual:
            build = self.worker._make_lazy_large_output

            def large_output(base, keys):
                out = build(base, keys)
                if self.mode in ("routed_free_kv","routed_replay_logits") and out.scheduled_cached_reqs.req_ids:
                    cached = out.scheduled_cached_reqs
                    assert not out.scheduled_new_reqs
                    cached.num_computed_tokens = [
                        base.dual_model_metadata[k].history_len_before-1 for k in cached.req_ids
                    ]
                    out.num_scheduled_tokens = dict.fromkeys(keys, 1)
                    out.total_num_scheduled_tokens = len(keys)
                self.large_actual = out.num_scheduled_tokens
                return out

            self.worker._make_lazy_large_output = large_output
        if args.job == "diag":
            sampler = self.main.sample_tokens

            def profiled_sample(grammar):
                if not self.capture:
                    return sampler(grammar)
                first = torch.cuda.Event(enable_timing=True)
                last = torch.cuda.Event(enable_timing=True)
                first.record()
                result = sampler(grammar)
                last.record()
                self.events.append(("sample",first,last))
                return result

            self.main.sample_tokens = profiled_sample
            if self.dual:
                construct = self.worker._make_lazy_large_output

                def profiled_group(base, keys):
                    if not self.capture:
                        return construct(base, keys)
                    tick = time.perf_counter()
                    output = construct(base, keys)
                    self.shape["group_build_host_ms"] = 1000*(time.perf_counter()-tick)
                    return output

                self.worker._make_lazy_large_output = profiled_group
            for name, runner in [("main",self.main), ("large",self.large)]:
                if runner is None:
                    continue
                original = runner._determine_batch_execution_and_padding

                def shape(*a, _original=original, _name=name, **kw):
                    result = _original(*a, **kw)
                    self.shape[_name] = (result[0].name,result[1].num_tokens)
                    return result

                runner._determine_batch_execution_and_padding = shape
                execute = runner.execute_model

                def measured(*a, _execute=execute, _name=name, **kw):
                    if not self.capture:
                        return _execute(*a, **kw)
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    result = _execute(*a, **kw)
                    end.record()
                    self.events.append((_name,start,end))
                    return result

                runner.execute_model = measured

    @torch.inference_mode()
    def cohort(self, records, check=False, poison_at=None, teacher=False):
        reserve_start = time.perf_counter()
        data = {r["request_id"]:r for r in records}
        ids = list(data)
        end = {}
        blocks = {}
        next_block = 1  # vLLM reserves block zero.
        for key, r in data.items():
            count = (len(r["prompt_ids"]) + len(r["output_ids"]) + 15)//16
            blocks[key] = list(range(next_block,next_block+count))
            next_block += count
        for runner in (self.main,self.large):
            if runner is not None:
                assert next_block <= runner.kv_caches[0].shape[0]
        torch.cuda.synchronize()
        prep_start = time.perf_counter()
        if self.args.job == "diag":
            torch.cuda.reset_peak_memory_stats()
            before_prepare = torch.cuda.memory_allocated()
        if self.mode in ("routed_free_kv","routed_replay_logits"):
            for key,r in data.items():
                tokens = r["prompt_ids"] + r["output_ids"][:-1]
                self.large.execute_model(schedule_new(key,tokens,blocks[key]))
                self.large.execute_model_state = None
                forget(self.large,[key])
        torch.cuda.synchronize()
        prefill_start = time.perf_counter()
        prepare_extra = (torch.cuda.max_memory_allocated()-before_prepare
                         if self.args.job=="diag" else None)
        prompt_commit = {}
        for key,r in data.items():
            p = len(r["prompt_ids"])
            out = schedule_new(key,r["prompt_ids"],blocks[key])
            if self.dual:
                out.dual_model_metadata = {key:DualStepMetadata(
                    LARGE,0,p,0,p,(blocks[key],),(blocks[key],))}
                self.worker.execute_model(out)
                self.large.execute_model_state = None
                self.worker.dual_request_states[key].commit_forwards(
                    self.worker.pending_dual_steps[key][0])
                self.worker.dual_request_states[key].commit_sample()
                commit(self.large,key,r["output_ids"][0],False)
            else:
                self.main.execute_model(out)
            self.main.sample_tokens(None)
            commit(self.main,key,r["output_ids"][0],True)
            if self.args.job=="diag":
                prompt_commit[key]=time.perf_counter()
            end[key] = p-1
        probe = getattr(self, "benchmark_probe", None)
        if probe:
            probe.begin(self, records)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        logs, steps, checks = [], [], {}
        policy = getattr(self, "ready_queue", None)
        if policy:
            policy.reset(ids)
        max_ticks = (sum(len(r["output_ids"]) for r in records) if policy
                     else max(len(r["output_ids"]) for r in records))
        for t in range(1,max_ticks):
            if not policy:
                active = [k for k in ids if len(data[k]["output_ids"])>t]
            self.events = []
            self.shape.clear()
            self.capture = self.args.job == "diag"
            router_start = time.perf_counter()
            if policy:
                active,pred,when,wait_ms = policy.select(data,self.router)
                if not active:
                    break
            else:
                pred = {k:("L" if self.router.route(k,t)==LARGE else "S") for k in active} if self.dual else {}
            group_start = time.perf_counter()
            pos = {k:len(data[k]["prompt_ids"])+(when[k] if policy else t)-1 for k in active}
            heavy = [k for k in active if self.dual and data[k]["route_for_output"][when[k] if policy else t]=="L"]
            gaps = {k:pos[k]-end[k]-1 for k in heavy}
            out = SchedulerOutput.make_empty()
            out.num_common_prefix_blocks = [0]
            out.num_scheduled_tokens = dict.fromkeys(active,1)
            out.total_num_scheduled_tokens = len(active)
            c = out.scheduled_cached_reqs
            c.req_ids = active
            c.num_computed_tokens = list(pos.values())
            c.num_output_tokens = [when[k] for k in active] if policy else [t]*len(active)
            c.new_block_ids = [None]*len(active)
            self.large_actual = {}
            if self.dual:
                out.dual_model_metadata = {
                    k:DualStepMetadata(int(k in gaps),when[k] if policy else t,pos[k]+1,end[k]+1,
                                       gaps[k]+1 if k in gaps else 0,(blocks[k],),([],))
                    for k in active}
                if poison_at == t:
                    for k in heavy:
                        for j in range(pos[k]+1,len(data[k]["prompt_ids"])+len(data[k]["output_ids"])):
                            for kv in self.large.kv_caches:
                                kv[blocks[k][j//16],:,j%16].fill_(3.25)
                execute_start = time.perf_counter()
                self.worker.execute_model(out)
                if probe and probe.inspect and heavy:
                    probe.inspect_rows(self.large,t,pos)
                scatter_start = time.perf_counter() if self.capture else 0
                state = self.main.execute_model_state
                selected = state.logits.clone()
                for k in heavy:
                    logits = self.large.execute_model_state.logits[self.large.input_batch.req_id_to_index[k]]
                    selected[self.main.input_batch.req_id_to_index[k]].copy_(logits)
                    if check:
                        checks[k,t] = logits.detach().float().cpu()
                self.main.execute_model_state = state._replace(logits=selected)
                self.large.execute_model_state = None
                if self.capture:
                    self.shape["scatter_host_ms"] = (time.perf_counter()-scatter_start)*1000
            else:
                execute_start = time.perf_counter()
                self.main.execute_model(out)
                if check:
                    for k in active:
                        checks[k,t] = self.main.execute_model_state.logits[
                            self.main.input_batch.req_id_to_index[k]].detach().float().cpu()
            sample_start = time.perf_counter()
            sampled = self.main.sample_tokens(None)
            if probe and probe.inspect:
                probe.inspect_samples(self.main, sampled, t)
            for k in active:
                token = data[k]["output_ids"][when[k] if policy else t]
                commit(self.main,k,token,True)
                if self.dual:
                    commit(self.large,k,token,False)
                    state = self.worker.dual_request_states[k]
                    state.commit_forwards(self.worker.pending_dual_steps[k][0])
                    state.commit_sample()
                if k in gaps:
                    end[k] = pos[k]
            stamp = (time.perf_counter()-start)*1000
            actual = sum(self.large_actual.values())-len(heavy) if self.dual else None
            assert not self.dual or actual == (0 if self.mode in ("routed_free_kv","routed_replay_logits") else sum(gaps.values()))
            steps.append(dict(step=t,active=len(active),heavy=len(heavy) if self.dual else None,
                              small_positions=len(active) if self.mode!="large" else None,
                              large_positions=len(active) if self.mode=="large" else len(heavy) if self.dual else None,
                              gap=sum(gaps.values()) if self.dual else None,catchup=actual,
                              graph=self.shape.copy(),_events=self.events,
                              router_host_ms=(group_start-router_start)*1000 if self.dual else None,
                              group_host_ms=(execute_start-group_start)*1000,
                              sample_commit_host_ms=stamp-(sample_start-start)*1000))
            if policy:
                logs.extend(dict(step=t,output_index=when[k],request_id=k,pos=pos[k],
                                 action=data[k]["route_for_output"][when[k]],predicted=pred[k],
                                 gap=gaps.get(k),wait_ms=wait_ms.get(k),commit_ms=stamp) for k in active)
            else:
                logs.extend(dict(step=t,request_id=k,pos=pos[k],action=data[k]["route_for_output"][t],
                                 predicted=pred.get(k),gap=gaps.get(k),commit_ms=stamp) for k in active)
            if poison_at == t:
                break
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        if probe:
            probe.end(self)
        self.capture = False
        if prompt_commit:
            for row in logs:
                if row["step"]==1:
                    row["prefill_to_first_decode_ms"]=(
                        row["commit_ms"]+(start-prompt_commit[row["request_id"]])*1000)
        for s in steps:
            for name,a,b in s.pop("_events"):
                s[name+"_forward_ms"] = a.elapsed_time(b)
        peak = torch.cuda.max_memory_allocated()
        if teacher:
            for k,t in list(checks):
                tokens = data[k]["prompt_ids"]+data[k]["output_ids"][:t]
                self.main.execute_model(schedule_new("teacher",tokens,blocks[k]))
                checks[k,t] = self.main.execute_model_state.logits[0].detach().float().cpu()
                self.main.execute_model_state = None
                forget(self.main,["teacher"])
        forget(self.main,ids)
        if self.dual:
            forget(self.large,ids)
            self.worker.dual_request_states.clear()
            self.worker.pending_dual_steps.clear()
        return dict(seconds=elapsed,precompute_s=prefill_start-prep_start,
                    block_table_prepare_s=prep_start-reserve_start,
                    prepare_peak_extra_bytes=prepare_extra,
                    prefill_s=start-prefill_start,peak_bytes=peak,steps=steps,requests=logs),checks


if __name__ == "__main__":
    args = arguments()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.job == "record":
        record(args)
    else:
        run(args)
