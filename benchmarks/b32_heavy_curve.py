"""Time native 1.5B decoder + row selection + LM head at trace-derived contexts."""
import argparse
import csv
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

from free_kv_replay import Replay, SchedulerOutput, commit, forget, rows, schedule_new, torch


def bucket(pos):
    return 0 if pos<96 else 96 if pos<128 else 128


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=Path("artifacts/b32-logits-replay"))
    parser.add_argument("--small",default="/root/autodl-tmp/models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--large",default="/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct")
    args=parser.parse_args()
    args.job="curve"
    args.mode="routed_free_kv"
    engine=Replay(args)
    runner=engine.large
    groups=defaultdict(list)
    for r in rows(Path("artifacts/free-kv-v1/trace.jsonl")):
        for t,a in enumerate(r["route_for_output"][1:],1):
            if a=="L":
                pos=len(r["prompt_ids"])+t-1
                groups[bucket(pos)].append((r,t,pos))
    rng=random.Random(17)
    measurements=[]
    active=False
    native=runner._model_forward
    lm=runner.model.compute_logits
    dispatch=runner._determine_batch_execution_and_padding
    def shape(*a,**kw):
        nonlocal graph,pad
        result=dispatch(*a,**kw)
        graph,pad=result[0].name,result[1].num_tokens
        return result
    def forward(*a,**kw):
        nonlocal start
        if active:
            torch.cuda.synchronize()
            start=time.perf_counter_ns()
            first.record()
        return native(*a,**kw)
    def logits(*a,**kw):
        nonlocal elapsed
        result=lm(*a,**kw)
        if active:
            last.record()
            last.synchronize()
            elapsed=(time.perf_counter_ns()-start)/1e6
        return result
    runner._determine_batch_execution_and_padding=shape
    runner._model_forward=forward
    runner.model.compute_logits=logits
    for k in range(1,9):
        for lower,population in sorted(groups.items()):
            for batch in range(6):
                candidates=list(population)
                rng.shuffle(candidates)
                chosen={}
                for r,t,pos in candidates:
                    chosen.setdefault(r["request_id"],(r,t,pos))
                    if len(chosen)==k:
                        break
                assert len(chosen)==k
                next_block=1
                out=SchedulerOutput.make_empty()
                out.num_common_prefix_blocks=[0]
                cached=out.scheduled_cached_reqs
                for key,(r,t,pos) in chosen.items():
                    prefix=r["prompt_ids"]+r["output_ids"][:t-1]
                    blocks=list(range(next_block,next_block+(pos+16)//16))
                    next_block+=len(blocks)
                    runner.execute_model(schedule_new(key,prefix,blocks))
                    runner.execute_model_state=None
                    commit(runner,key,r["output_ids"][t-1],False)
                    cached.req_ids.append(key)
                    cached.num_computed_tokens.append(pos)
                    cached.num_output_tokens.append(1)
                    cached.new_block_ids.append(None)
                out.num_scheduled_tokens=dict.fromkeys(chosen,1)
                out.total_num_scheduled_tokens=k
                for repeat in range(6):
                    first,last=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    active=True
                    graph=pad=start=elapsed=None
                    runner.execute_model(out)
                    active=False
                    runner.execute_model_state=None
                    if repeat:
                        measurements.append(dict(k=k,context_bucket_min=lower,
                            context_min=min(x[2] for x in chosen.values()),context_max=max(x[2] for x in chosen.values()),
                            batch=batch,repeat=repeat,graph=graph,padded_rows=pad,
                            gpu_ms=first.elapsed_time(last),wall_ms=elapsed))
                forget(runner,chosen)
    with (args.out/"heavy_curve_samples.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(measurements[0]));w.writeheader();w.writerows(measurements)
    result=[]
    for k in range(1,9):
        for lower in sorted(groups):
            group=[s for s in measurements if s["k"]==k and s["context_bucket_min"]==lower]
            result.append(dict(k=k,context_bucket_min=lower,padded_rows=group[0]["padded_rows"],
                               graph=group[0]["graph"],samples=len(group),
                               gpu_ms=statistics.median(s["gpu_ms"] for s in group),
                               wall_ms=statistics.median(s["wall_ms"] for s in group)))
    with (args.out/"heavy_curve.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)

if __name__=="__main__":
    main()
