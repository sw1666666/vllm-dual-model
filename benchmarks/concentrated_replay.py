"""2B: use exactly the 2A runner and conditional graphs. Only trace flags differ."""
import json
import sys
from collections import defaultdict
from types import SimpleNamespace

from concentrated_masks import BASE, OUT, TRACE, dump, sha
from free_kv_replay import Replay, rows, torch
from mixed_ffn_engine import get_forward_context
from mixed_ffn_replay import Driver, err, prepare, sizes

MODES=("F","M150","M96","M80")


def inputs():
    from pathlib import Path
    for f,h in json.loads((OUT/"inputs.json").read_text())["hashes"].items():assert sha(Path(f))==h,f
    original=rows(TRACE)
    locked=json.loads((OUT/"locked_steps.json").read_text())
    all_groups={};selections={}
    for m in MODES:
        sel=[s["ids"] if m=="F" else s["original"] for s in locked] if m in ("F","M150") else json.loads((OUT/f"{m}.json").read_text())
        recs={r["request_id"]:dict(r,route_for_output=list(r["route_for_output"])) for r in original}
        for s,chosen in zip(locked,sel):
            for rid in s["ids"]:
                if m!="F":recs[rid]["route_for_output"][s["step"]]="L" if rid in chosen else "S"
        g=list(recs.values())
        if m in ("F","M150"):assert g==original
        all_groups[m]=[g[i:i+32] for i in range(0,len(g),32)]
        selections[m]=sel
    return locked,all_groups,selections


@torch.inference_mode()
def main():
    locked,groups,sel=inputs()
    # Full first cohort, real prefixes, same synthetic history up to the first
    # few wide steps. Limit the audit size only; all formal runs use 100 requests.
    samples={}
    for mode in ("M96","M80"):
        t=next(s["step"] for s,w in zip(locked,sel[mode]) if s["cohort"]==0 and w)
        length=max(5,t+2)
        samples[mode]=[dict(r,output_ids=r["output_ids"][:length],route_for_output=r["route_for_output"][:length]) for r in groups[mode][0]]
    replay=Replay(SimpleNamespace(mode="large",large="/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct",job="mix"))
    d=Driver(replay);replay.benchmark_probe=d
    plan=defaultdict(set)
    for mode in MODES[1:]:
        for k,pads in sizes(groups[mode]).items():plan["M",k,False].update(pads)
    for g in samples.values():
        for k,pads in sizes([g]).items():plan["dense",0,False].update(pads);plan["M",k,False].update(pads)
    prepare(d.paths,dict(plan))
    checks=[]
    for mode,g in samples.items():
        ref,p0,v0,s0=d.run("dense",[g],"fixture")
        test,p1,v1,s1=d.run("M",[g],"fixture")
        expect=[(s,w) for s,w in zip(locked,sel[mode]) if s["cohort"]==0][:len(s1)]
        assert s0==s1 and s1==[(s["ids"],w) for s,w in expect]
        assert all(torch.equal(p0[k],p1[k]) for k in p0)
        assert set(ref)==set(test)
        vals=[]
        for (rid,t),a in ref.items():
            b=test[rid,t];ab,rel=err(a,b);s,w=expect[t-1]
            vals.append(dict(request=rid,pos=s["pos"][s["ids"].index(rid)],wide=rid in w,max_abs=ab,relative_l2=rel,top1=int(a.argmax()==b.argmax())))
        rate=sum(x["top1"] for x in vals)/len(vals)
        assert any(x["wide"] for x in vals) and max(x["relative_l2"] for x in vals)<.025 and rate>=.90
        checks.append(dict(mode=mode,n=len(vals),wide=sum(x["wide"] for x in vals),top1=rate,max_relative_l2=max(x["relative_l2"] for x in vals),worst_abs=max(vals,key=lambda x:x["max_abs"]),full_prefix_KV_equal=True))
        del ref,test,p0,p1,v0,v1
    dump("checks.json",checks)
    print("dense checks",checks,flush=True)
    audit=[]
    # Untimed physical-row + graph-dispatch audit. No GPU events/profiling.
    direct=d.paths.forward;calls=[]
    def observed(key,kwargs):
        c=get_forward_context();calls.append((c.cudagraph_runtime_mode.name,c.batch_descriptor.num_tokens,key[1]))
        return direct(key,kwargs)
    d.paths.forward=observed
    for m in MODES:
        j=0
        for i,g in enumerate(groups[m]):
            calls.clear();d.detail.clear()
            d.mode="F" if m=="F" else "M";d.audit="trace" if m=="F" else "mapping";d.cohort=i
            replay.cohort(g)
            for t,(ids,wide) in enumerate(d.sigs):
                s=locked[j];w=sel[m][j]
                assert ids==s["ids"] and s["step"]==t+1 and s["cohort"]==i
                if m=="F":
                    detail=d.detail[t];graph,p,k=detail["graph"],detail["base_padded"],len(ids)
                else:
                    graph,p,k=calls[t];assert wide==w and k==len(w)
                assert graph=="FULL"
                audit.append(dict(mode=m,cohort=i,step=t+1,active=len(ids),padded_main=p,k=len(w),executed_wide=p if m=="F" else k,graph=graph))
                j+=1
        assert j==188
    d.paths.forward=direct
    dump("audit.json",audit)
    for m in MODES:d.run("F" if m=="F" else "M",groups[m])
    for rep in range(5):
        for j in range(4):
            m=MODES[(rep+j)%4];rec=d.run("F" if m=="F" else "M",groups[m])
            dump(f"time-{rep}-{m}.json",rec)
            print("formal",rep,m,round(1000*sum(c["wall_s"] for c in rec),2),flush=True)


if __name__=="__main__":main()
