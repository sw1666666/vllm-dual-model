"""Stage one only: stock, single-model vLLM decode at fixed FFN widths."""
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from export_static_ffn import OUT, model, sha
from free_kv_replay import Replay, rows, torch

TRACE = Path("artifacts/free-kv-v1/trace.jsonl")


def pct(values, q):
    s = sorted(values)
    p = q*(len(s)-1)
    i = int(p)
    return s[i]+(p-i)*(s[min(len(s)-1,i+1)]-s[i])


def groups(trace,b):
    assert all(r["groups"][str(b)]==list(divmod(i,b)) for i,r in enumerate(trace))
    return [trace[i:i+b] for i in range(0,len(trace),b)]


@torch.inference_mode()
def diagnose(engine, trace, width):
    runner=engine.main
    layers=runner.get_model().model.layers
    assert len(layers)==len(runner.kv_caches)==28
    for layer in layers:
        assert list(layer.mlp.gate_up_proj.weight.shape)==[2*width,1536]
        assert list(layer.mlp.down_proj.weight.shape)==[1536,width]
    # Execute the loaded, unmodified vLLM MLP directly in a separate profiler
    # pass: captures actual GEMM operands (CUDA graph replays hide aten ops).
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=True) as prof:
        for n in (1,32):
            layers[0].mlp(torch.zeros(n,1536,dtype=torch.float16,device="cuda"))
        torch.cuda.synchronize()
    ops=[(e.name,e.input_shapes) for e in prof.events() if e.name in ("aten::mm","aten::linear","aten::matmul","aten::addmm")]
    for n in (1,32):
        for k,out in ((1536,width*2),(width,1536)):
            assert any([n,k] in dims and ([k,out] in dims or [out,k] in dims) for _,dims in ops),(n,k,out,ops)
    fixture=dict(trace[0],prompt_ids=trace[0]["prompt_ids"][:12],output_ids=trace[0]["output_ids"][:5])
    positions=range(12,16)
    for kv in runner.kv_caches:
        for pos in positions:kv[1,:,pos].fill_(float("nan"))
    sample, logits=engine.cohort([fixture],check=True)
    assert len(logits)==4 and all(v.isfinite().all() for v in logits.values())
    assert all(kv[1,:,12:16].isfinite().all() for kv in runner.kv_caches)
    shapes={}
    for b in (1,32):
        hist=Counter()
        for g in ([ [fixture] ] if b==1 else groups(trace,b)):
            rec,_=engine.cohort(g)
            hist.update((s["active"],*s["graph"]["main"]) for s in rec["steps"])
        shapes[b]=[dict(active=a,mode=m,padded=p,count=c) for (a,m,p),c in sorted(hist.items())]
    (OUT/f"diag-{width}.json").write_text(json.dumps(dict(width=width,gemm_ops=ops,graph=shapes,
            all_loaded_layers=28,finite_logits_steps=4,finite_new_kv_layers=28,preemptions=0),indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("job",choices=("time","diag"))
    p.add_argument("width",type=int)
    p.add_argument("--repeat",type=int,default=0)
    args=p.parse_args()
    assert sha(TRACE)=="393281ea1a8eee5613d4533be07d193fe1043cb3e8d8190f429e04df83ad4586"
    args.mode="large";args.large=str(model(args.width));args.out=OUT
    trace=rows(TRACE)
    engine=Replay(args)
    if args.job=="diag":
        diagnose(engine,trace,args.width)
        return
    assert (OUT/f"diag-{args.width}.json").exists()
    for b in ((1,32) if args.repeat%2==0 else (32,1)):
        cohorts=groups(trace,b)
        # B=1 has only a single graph shape. B=32 warmup covers every tail.
        for g in (cohorts[:1] if b==1 else cohorts):
            engine.cohort(g)
        wall=prefill=0.
        itl=[];cohort_rows=[];n=0
        for i,g in enumerate(cohorts):
            c,_=engine.cohort(g)
            wall+=c["seconds"];prefill+=c["prefill_s"]
            last={};seq={}
            for r in c["requests"]:
                key=r["request_id"];t=r["step"];stamp=r["commit_ms"]
                if key in last:itl.append(stamp-last[key])
                last[key]=stamp;seq.setdefault(key,[]).append((t,r["pos"]))
            for x in g:
                assert seq.get(x["request_id"],[])==[(j,len(x["prompt_ids"])+j-1) for j in range(1,len(x["output_ids"]))]
            n+=len(c["requests"])
            cohort_rows.append(dict(cohort=i,seconds=c["seconds"],prefill_s=c["prefill_s"],N=len(c["requests"]),
                                    commits=[(r["request_id"],r["step"],r["commit_ms"]) for r in c["requests"]]))
        assert n==4091 and len(itl)==3991
        data=dict(repeat=args.repeat,B=b,width=args.width,N=n,wall_ms=wall*1000,prefill_ms=prefill*1000,
                  tok_s=n/wall,itl_n=len(itl),itl_p95_ms=pct(itl,.95),preemptions=0,cohorts=cohort_rows)
        (OUT/f"time-{args.repeat}-{args.width}-{b}.json").write_text(json.dumps(data))
        print({k:v for k,v in data.items() if k!="cohorts"},flush=True)


if __name__=="__main__":
    main()
