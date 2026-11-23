"""Paired B=32 oracle-logits experiment on the free-KV replay driver."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

from free_kv_replay import Replay, rows, save_rows, torch

L, F, O = "large", "routed_free_kv", "routed_replay_logits"


class Probe:
    def __init__(self, job, out):
        self.job = job
        self.out = out
        self.inspect = job in ("capture", "verify")

    def begin(self, replay, records):
        self.cohort = records[0]["groups"]["32"][0]
        self.path = self.out/f"native-{self.cohort}.pt"
        self.output, self.samples, self.observations, self.spans = [], [], [], []
        self.calls = self.native_rows = self.lookup_ns = self.buffer_bytes = 0
        self.real = replay.large
        self.hooks = []
        if self.real is None:
            return
        if replay.mode == O:
            saved = torch.load(self.path, map_location="cpu")
            self.views = []
            for row in saved:
                host = row["logits"]
                device = torch.empty_strided(host.shape, row["stride"], dtype=host.dtype, device="cuda")
                device.copy_(host)
                self.buffer_bytes += device.untyped_storage().nbytes()
                self.views.append(device)
            self.iterator = iter(self.views)
            self.real._benchmark_ready_logits = self if self.job=="diag" else self.iterator
        if self.inspect or self.job=="diag":
            for obj, attr in [(self.real, "_model_forward"), (self.real.model,"compute_logits")]:
                original = getattr(obj, attr)
                def observed(*a, _orig=original, _attr=attr, **kw):
                    if _attr=="_model_forward":
                        self.calls += 1
                        self.native_rows += self.real.input_batch.num_reqs
                    if self.job!="diag":
                        return _orig(*a,**kw)
                    first,last = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    first.record()
                    started=time.perf_counter_ns()
                    value=_orig(*a,**kw)
                    last.record()
                    self.spans.append((_attr,first,last,(time.perf_counter_ns()-started)/1e6))
                    return value
                self.hooks.append((obj,attr,original))
                setattr(obj,attr,observed)

    def __next__(self):
        start=time.perf_counter_ns()
        value=next(self.iterator)
        ns=time.perf_counter_ns()-start
        self.lookup_ns+=ns
        self.spans.append(("view",None,None,ns/1e6))
        return value

    def inspect_rows(self, runner, step, pos):
        logits=runner.execute_model_state.logits
        ids=list(runner.input_batch.req_ids)
        self.output.append(dict(step=step,ids=ids,pos=[pos[k] for k in ids],
                                stride=logits.stride(),logits=logits.detach().cpu()))

    def inspect_samples(self, runner, sampled, t):
        self.samples.append([(k,t,tuple(sampled.sampled_token_ids[i]))
                             for i,k in enumerate(runner.input_batch.req_ids)])

    def end(self, replay):
        for obj,attr,orig in self.hooks:
            setattr(obj,attr,orig)
        if replay.mode==O:
            assert next(self.iterator,None) is None
            self.real._benchmark_ready_logits=None
            self.views.clear()
        if self.job=="capture":
            torch.save(self.output,self.path)
        if self.inspect and replay.mode!=L:
            assert self.calls==(0 if replay.mode==O else len(self.output))
            assert self.native_rows==(0 if replay.mode==O else sum(len(x["ids"]) for x in self.output))


def audit(probe, index):
    saved=torch.load(probe.path,map_location="cpu")
    assert len(saved)==len(probe.output)
    diffs=[]
    for ref,run in zip(saved,probe.output):
        assert all(ref[k]==run[k] for k in ("step","ids","pos","stride"))
        a,b=ref["logits"],run["logits"]
        assert a.dtype==b.dtype and a.shape==b.shape and torch.isfinite(b).all()
        for i,k in enumerate(run["ids"]):
            err=(a[i]-b[i]).abs().max().item()
            diffs.append(dict(cohort=index,step=run["step"],request_id=k,input_pos=run["pos"][i],
                              heavy_row=i,max_abs=err,top1=int(a[i].argmax()==b[i].argmax()),
                              dtype=str(b.dtype),vocab=b.shape[1],row_stride=run["stride"][0]))
    return diffs


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("job",choices=["capture","verify","time","diag"])
    parser.add_argument("--mode",choices=[L,F,O],required=True)
    parser.add_argument("--repeat",type=int,default=0)
    parser.add_argument("--out",type=Path,default=Path("artifacts/b32-logits-replay"))
    parser.add_argument("--small",default="/root/autodl-tmp/models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--large",default="/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct")
    args=parser.parse_args()
    trace=Path("artifacts/free-kv-v1/trace.jsonl")
    assert hashlib.sha256(trace.read_bytes()).hexdigest()=="393281ea1a8eee5613d4533be07d193fe1043cb3e8d8190f429e04df83ad4586"
    records=rows(trace)
    groups=[records[i:i+32] for i in range(0,len(records),32)]
    args.out.mkdir(parents=True,exist_ok=True)
    if args.job in ("time","diag"):
        for m in (F,O):
            assert json.loads((args.out/f"check-{m}.json").read_text())["passed"]
    replay=Replay(args)
    probe=Probe(args.job,args.out)
    replay.benchmark_probe=probe
    if args.job in ("time","diag"):
        for group in groups:
            replay.cohort(group)
    results=[]; signatures=[]; samples=[]; errors=[]
    for idx,group in enumerate(groups):
        result,_=replay.cohort(group)
        result.update(cohort=idx,buffer_bytes=probe.buffer_bytes,lookup_ns=probe.lookup_ns,
                      native_calls=probe.calls if args.job!="time" else None,
                      native_rows=probe.native_rows if args.job!="time" else None)
        if args.job=="diag":
            spans=iter(probe.spans)
            for s in result["steps"]:
                if s["heavy"]:
                    if args.mode==O:
                        kind,_,_,s["view_host_ms"]=next(spans)
                        assert kind=="view"
                    else:
                        for attr in ("_model_forward","compute_logits"):
                            kind,a,b,wall=next(spans)
                            assert kind==attr
                            s[attr+"_gpu_ms"]=a.elapsed_time(b)
                            s[attr+"_submit_ms"]=wall
            assert next(spans,None) is None
        if probe.inspect:
            signatures.extend((idx,r["request_id"],r["step"],r["pos"],r["action"]) for r in result["requests"])
            samples.extend(probe.samples)
            errors.extend(audit(probe,idx))
        results.append(result)
    n=sum(len(x["requests"]) for x in results)
    assert n==4091
    if args.mode!=L:
        assert sum(s["heavy"] for x in results for s in x["steps"])==384
        assert sum(s["catchup"] for x in results for s in x["steps"])==0
        assert all(r["action"]==r["predicted"] for x in results for r in x["requests"])
    if args.job=="capture":
        torch.save((signatures,samples),args.out/"capture-schedule.pt")
    if probe.inspect:
        reference=torch.load(args.out/"capture-schedule.pt")
        assert reference==(signatures,samples) and len(errors)==384
        worst=max(errors,key=lambda r:r["max_abs"])
        save_rows(args.out/f"rows-{args.job}-{args.mode}.jsonl",errors)
        assert all(r["top1"] for r in errors) and worst["max_abs"]==0,worst
        (args.out/f"{'check' if args.job=='verify' else 'capture'}-{args.mode}.json").write_text(json.dumps(
            dict(passed=True,N=n,steps=len(samples),heavy_rows=len(errors),native_calls=sum(x["native_calls"] for x in results),
                 native_rows=sum(x["native_rows"] for x in results),catchup=0,ordered_rows_and_positions=True,
                 stop_steps_and_active_ids=True,samples_match=n,max_abs=worst["max_abs"],top1=1,
                 worst_request=worst["request_id"],worst_pos=worst["input_pos"],max_buffer_bytes=max(x["buffer_bytes"] for x in results))))
    save_rows(args.out/f"{args.job}-{args.mode}-{args.repeat}.jsonl",results)


if __name__=="__main__":
    main()
