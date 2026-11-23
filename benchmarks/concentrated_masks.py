"""Offline only: redistribute wide flags on the locked physical 2A schedule."""
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

OUT=Path("artifacts/mixed-ffn-2b")
BASE=Path("artifacts/mixed-ffn-v1")
TRACE=Path("artifacts/free-kv-v1/trace.jsonl")
SEED=170219


def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,separators=(",",":"),allow_nan=False)+"\n")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def locked():
    tr={r["request_id"]:r for r in map(json.loads,TRACE.read_text().splitlines())}
    logs=json.loads((BASE/"time-0-M.json").read_text())
    pos={(i,t,r):p for i,c in enumerate(logs) for r,t,p,_ in c["commits"]}
    table=[]
    for s in json.loads((BASE/"steps.json").read_text()):
        if s["mode"]!="M":continue
        i,t=s["cohort"],s["step"];ids=s["physical_ids"]
        assert len(ids)==len(set(ids)) and set(ids)=={r for r,x in tr.items() if x["groups"]["32"][0]==i and t<len(x["output_ids"])}
        pp=[pos[i,t,r] for r in ids]
        assert all(p==len(tr[r]["prompt_ids"])+t-1 for r,p in zip(ids,pp))
        assert s["wide_ids"]==[r for r in ids if tr[r]["route_for_output"][t]=="L"]
        table.append(dict(cohort=i,step=t,ids=ids,pos=pp,original=s["wide_ids"]))
    assert len(table)==188 and len(pos)==sum(len(s["ids"]) for s in table)==4091
    return table


def cohort_H(parts,total):
    # First minimize sum(k^2), i.e. distance from ideal k at fixed total H;
    # ties favor proportional cohort allocation. Keep the tail out of the
    # two bigger batch steps if possible (they originally had zero wide rows).
    dp={0:(0,0,[])}
    for group in parts:
        w=sum(len(s["original"]) for s in group)
        opts=[]
        for h in range(1,min(len(group),w)+1):
            q,r=divmod(w,h);ks=[q]*(h-r)+[q+1]*r
            caps=sorted((len(s["ids"]) for s in group),reverse=True)
            if all(k<=n for k,n in zip(reversed(ks),caps)):
                opts.append((h,sum(k*k for k in ks),(384*h-total*w)**2))
        nxt={}
        for j,(cost,prop,hs) in dp.items():
            for h,a,b in opts:
                if j+h>total:continue
                z=cost+a,prop+b,hs+[h]
                if j+h not in nxt or z<nxt[j+h]:nxt[j+h]=z
        dp=nxt
    return dp[total][2]


def allocate(group,h,rng):
    by=defaultdict(list)
    for s in group:by[len(s["ids"])].append(s)
    w=sum(len(s["original"]) for s in group);q,r=divmod(w,h)
    # Assign counts of q and q+1 steps to exact active-size strata, minimizing
    # total absolute deviation in wide-row counts from 2A.
    dp={(0,0):(0,[])}
    for n,ss in sorted(by.items()):
        cap=len(ss);target=sum(len(s["original"]) for s in ss);nxt={}
        for (l,u),(cost,choices) in dp.items():
            for a in range(1+min(cap if n>=q else 0,h-r-l)):
                for b in range(1+min(cap-a if n>=q+1 else 0,r-u)):
                    score=cost+abs(a*q+b*(q+1)-target)
                    ix=l+a,u+b
                    if ix not in nxt or score<nxt[ix][0]:nxt[ix]=score,choices+[(n,a,b)]
        dp=nxt
    selections=dp[h-r,r][1]
    target=Counter();orig_ctx=[]
    for s in group:
        for rid,p in zip(s["ids"],s["pos"]):
            if rid in s["original"]:
                target[len(s["ids"]),(p+1)//32]+=1;orig_ctx.append(p+1)
    best=None
    for attempt in range(96):
        picks=[]
        for n,a,b in selections:
            if a+b==0:continue
            ss=by[n];weights=[len(s["original"])+0.1 for s in ss]
            # Uniform quantiles of original wide mass within this active size.
            chosen=set()
            for j in range(a+b):
                point=(j+rng.uniform(.12,.88))*sum(weights)/(a+b);acc=0;opts=[]
                for i,wt in enumerate(weights):
                    if i not in chosen:opts.append((abs(acc+wt/2-point),i))
                    acc+=wt
                chosen.add(min(opts)[1])
            ks=[q]*a+[q+1]*b;rng.shuffle(ks)
            picks.extend((ss[i],k) for i,k in zip(sorted(chosen),ks))
        rng.shuffle(picks);counts=Counter();sel={};contexts=[]
        for s,k in picks:
            rows=list(zip(s["ids"],s["pos"]));n=len(rows);selected=[]
            for _ in range(k):
                j=max(range(len(rows)),key=lambda i:target[n,(rows[i][1]+1)//32]-counts[n,(rows[i][1]+1)//32]+rng.random())
                rid,p=rows.pop(j);selected.append(rid);contexts.append(p+1);counts[n,(p+1)//32]+=1
            sel[s["step"]]=[rid for rid in s["ids"] if rid in selected]
        score=sum(abs(target[key]-counts[key]) for key in set(target)|set(counts)),sum(abs(i-j) for i,j in zip(sorted(contexts),sorted(orig_ctx)))
        if best is None or score<best[0]:best=score,sel
    return best[1]


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    table=locked();dump("locked_steps.json",table)
    parts=[[s for s in table if s["cohort"]==i] for i in range(4)]
    plans={"M150":[s["original"] for s in table]}
    for h in (96,80):
        hs=cohort_H(parts,h);rng=random.Random(SEED+h)
        plans[f"M{h}"]=[sel.get(s["step"],[]) for g,n in zip(parts,hs) for sel in [allocate(g,n,rng)] for s in g]
    counts={};distributions={}
    for m,sels in plans.items():
        by=Counter();hist=Counter();joint=Counter();ctx=[];overlap=0
        for s,ids in zip(table,sels):
            assert len(ids)==len(set(ids)) and set(ids)<=set(s["ids"])
            by[s["cohort"]]+=len(ids);hist[len(ids)]+=1
            overlap+=len(set(ids)&set(s["original"]))
            for rid in ids:
                length=s["pos"][s["ids"].index(rid)]+1;ctx.append(length)
                joint[s["cohort"],len(s["ids"]),length//32]+=1
        assert list(by.values())==[121,119,126,18] and 188-hist[0]==int(m[1:])
        counts[m]=dict(cohorts=dict(by),hist=dict(hist),H=188-hist[0],mean_context=sum(ctx)/len(ctx),retained=overlap)
        distributions[m]=[dict(cohort=i,active=n,context_bucket_min=b*32,wide=v) for (i,n,b),v in sorted(joint.items())]
        if m!="M150":dump(f"{m}.json",sels)
    dump("distribution.json",distributions);dump("mask_stats.json",counts)
    dump("inputs.json",dict(seed=SEED,hashes={str(p):sha(p) for p in (TRACE,BASE/"steps.json",BASE/"time-0-M.json",OUT/"locked_steps.json",OUT/"M96.json",OUT/"M80.json")}))
    print(counts)


if __name__=="__main__":main()
