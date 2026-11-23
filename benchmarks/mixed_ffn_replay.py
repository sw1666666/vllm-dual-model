"""B=32 shared-attention stage two. Run from repository root in the V100 env."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from free_kv_replay import Replay, rows, torch
from mixed_ffn_engine import Paths, get_forward_context

OUT=Path("artifacts/mixed-ffn-v1")
TRACE=Path("artifacts/free-kv-v1/trace.jsonl")
MODES=("F","C","W","M")


def save(name,data):
    (OUT/name).write_text(json.dumps(data,allow_nan=False))


def write(name,rows):
    with (OUT/name).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def fixture(trace):
    patterns=["SSLLSLS","SSSLSS","SSSLL","SSSLSLS"]
    return [dict(r,output_ids=r["output_ids"][:len(a)],route_for_output=a)
            for r,a in zip([trace[i] for i in (3,1,4,0)],patterns)]


def sizes(cohorts):
    result=defaultdict(set)
    for g in cohorts:
        for t in range(1,max(len(r["output_ids"]) for r in g)):
            active=[r for r in g if t<len(r["output_ids"])]
            k=sum(r["route_for_output"][t]=="L" for r in active)
            n=len(active);p=next(v for v in (1,2,4,8,16,24,32) if n<=v)
            result[k].add(p)
    return result


def prepare(paths,plan):
    # The stock startup dummy runner assumes empty block tables. A preceding
    # real tail has -1 padding pages: it cannot be reused as a 32-row dummy
    # with seq_len=1 for *every* row. No real requests may own these tables.
    r=paths.runner
    assert r.input_batch.num_reqs==0 and not r.requests
    table=r.input_batch.block_table[0]
    table.get_device_tensor(32).zero_()
    table.slot_mapping.gpu.fill_(-1)
    paths.prepare(plan)


class Driver:
    inspect=False  # Replay itself samples normally; check=True exports logits only on fixture.
    def __init__(self,replay):
        self.replay=replay
        self.r=replay.main
        self.paths=Paths(self.r)
        self.orig=self.r._model_forward
        self.host=torch.empty((64,32),dtype=torch.int64,pin_memory=True)
        self.np=self.host.numpy()
        self.detail=[]

    def kv(self,r,first,last):
        pos=torch.arange(first,last,device="cuda")
        pages=self.block[r]+pos//16
        return torch.stack([v[pages,:,pos%16] for v in self.r.kv_caches])

    def begin(self,replay,records):
        self.data={r["request_id"]:r for r in records}
        self.step=0
        self.sigs=[]
        self.block={};p=1
        for k,r in self.data.items():
            self.block[k]=p;p+=(len(r["prompt_ids"])+len(r["output_ids"])+15)//16
        if self.audit:
            self.prefix={}
            for k,r in self.data.items():
                n=len(r["prompt_ids"])
                if self.audit=="fixture":self.prefix[k]=self.kv(k,0,n).cpu()
                pos=torch.arange(n,n+len(r["output_ids"])-1,device="cuda")
                for v in self.r.kv_caches:v[self.block[k]+pos//16,:,pos%16]=float("nan")
        if self.mode!="F" or self.audit:
            self.r._model_forward=self.forward

    def end(self,replay):
        self.r._model_forward=self.orig
        if self.audit:
            self.written={}
            for k,r in self.data.items():
                n=len(r["prompt_ids"])
                x=self.kv(k,n,n+len(r["output_ids"])-1)
                assert x.shape[0]==28 and x.isfinite().all().item()
                if self.audit=="fixture":self.written[k]=x.cpu()

    def forward(self,**kw):
        mode=self.mode
        self.step+=1;t=self.step
        b=self.r.input_batch
        n=b.num_reqs
        ctx=get_forward_context()
        p=ctx.batch_descriptor.num_tokens
        if mode in ("M","dense") or self.audit:
            ids=b.req_ids
            idx=[]
            for i,rid in enumerate(ids):
                r=self.data[rid]
                pos=int(b.num_computed_tokens_cpu[i]);j=pos-len(r["prompt_ids"])+1
                if self.audit: assert j==t,(rid,i,pos,j,t)
                if r["route_for_output"][j]=="L":idx.append(i)
            if mode in ("M","dense") and idx:
                self.np[t,:len(idx)]=idx
                self.paths.indices[:len(idx)].copy_(self.host[t,:len(idx)],non_blocking=True)
            if mode=="dense":
                self.paths.mask.zero_()
                if idx:self.paths.mask.index_fill_(0,self.paths.indices[:len(idx)],True)
        k=len(idx) if mode=="M" else 0
        diagnostic=self.audit=="trace" and mode=="M" and k>0
        key=mode,k,diagnostic
        if self.audit:
            assert ctx.cudagraph_runtime_mode.name=="FULL"
            self.sigs.append((list(ids),[ids[i] for i in idx]))
        z=self.orig(**kw) if mode=="F" else self.paths.forward(key,kw)
        if self.audit=="trace":
            times=[0.,0.,0.]
            if diagnostic:
                torch.cuda.synchronize()
                for ev in self.paths.clocks[key,p]:
                    for j in range(3):times[j]+=ev[j].elapsed_time(ev[j+1])
            self.detail.append(dict(mode=mode,cohort=self.cohort,step=t,active=n,base_padded=p,
                k=len(idx),E_rows=(p if mode=="W" else k if mode=="M" else 0),
                graph=ctx.cudagraph_runtime_mode.name,gather_ms=times[0],E_ms=times[1],scatter_ms=times[2],
                physical_ids=list(ids),wide_ids=[ids[i] for i in idx]))
        return z

    def run(self,mode,cohorts,audit=False):
        self.mode=mode;self.audit=audit
        results=[]
        for i,group in enumerate(cohorts):
            self.cohort=i
            result,logits=self.replay.cohort(group,check=audit=="fixture")
            if audit=="fixture":
                return logits,self.prefix,self.written,self.sigs
            rr=result["requests"]
            assert len(rr)==sum(len(r["output_ids"])-1 for r in group)
            results.append(dict(wall_s=result["seconds"],prefill_s=result["prefill_s"],
                                commits=[(r["request_id"],r["step"],r["pos"],r["commit_ms"]) for r in rr]))
        return results


def err(a,b):
    a=a.float();b=b.float()
    return (a-b).abs().max().item(),((a-b).norm()/a.norm().clamp_min(1e-12)).item()


def checks(ref,test,which,fx):
    a,p,kv,s=ref;b,q,kt,u=test
    assert p.keys()==q.keys() and all(torch.equal(p[k],q[k]) for k in p)
    assert s==u and a.keys()==b.keys()
    assert {len(x[1]) for x in s}>={0,1,2}
    assert any(len(ids)==len(w) for ids,w in s)
    assert any(ids!=[r["request_id"] for r in fx if t<len(r["output_ids"])] for t,(ids,_) in enumerate(s,1))
    lens={r["request_id"]:len(r["prompt_ids"]) for r in fx}
    vals=[]
    for (rid,t),x in a.items():
        y=b[rid,t]
        ab,rel=err(x,y)
        vals.append(dict(type="logits",request=rid,pos=lens[rid]+t-1,layer=None,max_abs=ab,rel=rel,top1=int(x.argmax()==y.argmax())))
    for rid,x in kv.items():
        for i in range(28):
            for t in range(x.shape[1]):
                ab,rel=err(x[i,t],kt[rid][i,t])
                vals.append(dict(type="KV",request=rid,pos=lens[rid]+t,layer=i,max_abs=ab,rel=rel,top1=None))
    result=[]
    for name in ("logits","KV"):
        v=[x for x in vals if x["type"]==name]
        worst=max(v,key=lambda x:x["max_abs"])
        relative=max(x["rel"] for x in v)
        top=sum(x["top1"] for x in v)/len(v) if name=="logits" else None
        result.append(dict(pair=which,type=name,n=len(v),max_abs=worst["max_abs"],max_rel=relative,
             worst_request=worst["request"],worst_pos=worst["pos"],worst_layer=worst["layer"],top1=top,
             prompt_KV_bitwise=True,ordered_rows=True,passed=relative<.025 and (top is None or top>=.90)))
    return result


@torch.inference_mode()
def main():
    OUT.mkdir(exist_ok=True,parents=True)
    assert hashlib.sha256(TRACE.read_bytes()).hexdigest()=="393281ea1a8eee5613d4533be07d193fe1043cb3e8d8190f429e04df83ad4586"
    tr=rows(TRACE);fx=fixture(tr);groups=[tr[i:i+32] for i in range(0,len(tr),32)]
    replay=Replay(SimpleNamespace(mode="large",large="/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct",job="mix"))
    d=Driver(replay);replay.benchmark_probe=d
    fs=sizes([fx]);ts=sizes(groups)
    plan={}
    for m in ("C","W","Cref","dense"):
        plan[m,0,False]=set.union(*fs.values())
    for k,pads in fs.items():plan["M",k,False]=set(pads)
    prepare(d.paths,plan)
    print("fixture graphs ready",flush=True)
    tests={m:d.run(m,[fx],"fixture") for m in ("F","W","C","Cref","M","dense")}
    data=[]
    for ref,test in (("F","W"),("Cref","C"),("dense","M")):
        data+=checks(tests[ref],tests[test],ref+"/"+test,fx)
    write("checks.csv",data);save("fixture_rows.json",tests["M"][3])
    assert all(x["passed"] for x in data),data
    print("checks passed",data,flush=True)
    plan={("C",0,False):set.union(*ts.values()),("W",0,False):set.union(*ts.values())}
    plan.update({("M",k,False):pads for k,pads in ts.items()})
    prepare(d.paths,plan)
    for m in MODES:d.run(m,groups)
    for rep in range(5):
        for j in range(4):
            m=MODES[(j+rep)%4]
            res=d.run(m,groups)
            save(f"time-{rep}-{m}.json",res)
            print("formal",rep,m,sum(c["wall_s"] for c in res),flush=True)
    prepare(d.paths,{("M",k,True):p for k,p in ts.items() if k})
    for m in MODES:d.run(m,groups,"trace")
    save("steps.json",d.detail)
    for m in MODES:
        rr=[x for x in d.detail if x["mode"]==m]
        assert len(rr)==188 and sum(x["active"] for x in rr)==4091
        assert sum(x["k"] for x in rr)==384 and sum(x["k"]>0 for x in rr)==150
    print("diagnostics complete",flush=True)


if __name__=="__main__":main()
