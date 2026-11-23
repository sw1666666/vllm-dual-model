"""Recompute stage 2B latency, paired wall-clock budgets and net savings."""
import csv
import json
import statistics as st
from collections import Counter
from pathlib import Path
from concentrated_masks import BASE, OUT, TRACE, dump, sha

MODES=("F","M150","M96","M80")


def load(name):return json.loads((OUT/name).read_text())


def write(name,rows):
    with (OUT/name).open("w",newline="") as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)


def span(a,d):return f"{st.median(a):.{d}f} [{min(a):.{d}f}, {max(a):.{d}f}]"


def main():
    trace=[json.loads(l) for l in TRACE.read_text().splitlines()]
    groups=[trace[i:i+32] for i in range(0,len(trace),32)]
    expected=[[(r["request_id"],t,len(r["prompt_ids"])+t-1) for t in range(1,max(len(r["output_ids"]) for r in g))
               for r in g if t<len(r["output_ids"])] for g in groups]
    a=load("audit.json");stats={};mask=load("mask_stats.json")
    for m in MODES:
        rr=[x for x in a if x["mode"]==m]
        assert len(rr)==188 and sum(r["active"] for r in rr)==4091
        hist=Counter(r["k"] for r in rr);h=188-hist[0];w=sum(r["k"] for r in rr);real=sum(r["executed_wide"] for r in rr)
        assert (h,w)==((188,4091) if m=="F" else (int(m[1:]),384))
        if m!="F":assert real==384 and {int(k):v for k,v in mask[m]["hist"].items()}==hist
        stats[m]=dict(H=h,wide=w,executed=real,avg=w/h,hist=dict(sorted(hist.items())),FULL=sum(r["graph"]=="FULL" for r in rr),eager=sum(r["graph"]=="NONE" for r in rr))
    runs=[];pairs=[]
    for rep in range(5):
        times={}
        for m in MODES:
            rec=load(f"time-{rep}-{m}.json");assert len(rec)==4
            wall=sum(c["wall_s"] for c in rec)*1000;itl=[]
            for c,ex in zip(rec,expected):
                assert [(r,t,p) for r,t,p,_ in c["commits"]]==ex
                last={}
                for r,t,p,stamp in c["commits"]:
                    if r in last:itl.append(stamp-last[r])
                    last[r]=stamp
            assert len(itl)==3991 and min(itl)>=0
            itl.sort();p=.95*(len(itl)-1);i=int(p)
            times[m]=wall
            runs.append(dict(repeat=rep,mode=m,N=4091,wall_ms=wall,tok_s=4091000/wall,P95_ms=itl[i]+(p-i)*(itl[i+1]-itl[i]),
                             itl_n=len(itl),paired_speedup=times["F"]/wall,budget_ms=times["F"]/1.2-wall,
                             H=stats[m]["H"],executed_wide_rows=stats[m]["executed"],full_steps=stats[m]["FULL"],eager_steps=stats[m]["eager"]))
        pairs.append(dict(repeat=rep,M150_minus_M96=times["M150"]-times["M96"],M96_minus_M80=times["M96"]-times["M80"],
                          **{m+"_target_budget_ms":times["F"]/1.2-times[m] for m in MODES}))
    write("timings.csv",runs);write("pairs.csv",pairs)
    txt=["# 阶段 2B：固定宽位置数，改变非空步数","",
         "单张 Tesla V100-PCIE-32GB、Qwen2.5-1.5B-Instruct、FP16，原始全宽 prefill；只计 decode。每组 188 步、4,091 个强制提交输出，P95 为 3,991 个同请求 decode 间隔的分位数。以下为五轮中位数，速度比与预算先按同轮 F 配对。","",
         "| 模式 | H | 非空平均 k | k:步数（含空步） | decode ms | tok/s | F/被测 [范围] | P95 ITL ms | F/1.2−被测 ms [范围] |",
         "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for m in MODES:
        z=[r for r in runs if r["mode"]==m];v=lambda k:[r[k] for r in z];s=stats[m]
        hist=", ".join(f"{k}:{n}" for k,n in s["hist"].items())
        txt.append(f"| {m} | {s['H']} | {s['avg']:.2f} | {hist} | {st.median(v('wall_ms')):.1f} | {st.median(v('tok_s')):.1f} | {span(v('paired_speedup'),3)} | {st.median(v('P95_ms')):.2f} | {span(v('budget_ms'),1)} |")
    txt+=["",f"配对净节省 ms（中位数 [最小, 最大]）：M150→M96 {span([r['M150_minus_M96'] for r in pairs],1)}；M96→M80 {span([r['M96_minus_M80'] for r in pairs],1)}。",
          "",f"每组 188 FULL / 0 eager。F 对 4,091 个有效行执行原生全宽（主干 Graph 共 {stats['F']['executed']} 个含填充行）；三个 M 组实际各执行 384 个互补行，无额外填充。第二阶段原有 FFN 代码和 kernel 未改，图内的 k 与离线计划完全相符；未另跑 GPU 事件诊断。","",
          "| mask | 分 cohort 宽行 | 分 cohort 非空步 | 宽行平均上下文 | 与原始宽位置重合 |","|---|---|---|---:|---:|"]
    for m in MODES[1:]:
        co=[sum(x["k"]>0 for x in a if x["mode"]==m and x["cohort"]==i) for i in range(4)]
        txt.append(f"| {m} | 121 / 119 / 126 / 18 | {' / '.join(map(str,co))} | {mask[m]['mean_context']:.2f} | {mask[m]['retained']} / 384 |")
    dist=load("distribution.json")
    def margins(m):
        batch,ctx=Counter(),Counter()
        for r in dist[m]:
            batch[r["cohort"],r["active"]]+=r["wide"]
            ctx[r["context_bucket_min"]]+=r["wide"]
        return batch,ctx
    def tv(a,b):return sum(abs(a[k]-b[k]) for k in a.keys()|b.keys())/2
    base=margins("M150")
    txt.append("")
    for m in ("M96","M80"):
        ds=margins(m)
        txt.append(f"{m} 对 M150：按（cohort、实际活跃 batch）分层的宽行计数差异为 {tv(ds[0],base[0]):g}/384；按 32-token 上下文桶分层为 {tv(ds[1],base[1]):g}/384（均为计数差绝对值之和的一半）。")
    txt+=["","同 mask 稠密参考抽查（真实第一 cohort，保留被抽查步骤之前的实际 mask 历史）："]
    for r in load("checks.json"):
        txt.append(f"- {r['mode']}：{r['n']} 个 logits 位置（含 {r['wide']} 个宽位置），top-1 {r['top1']:.2%}，最大绝对误差 {r['worst_abs']['max_abs']:.5g}，最大相对 L2 {r['max_relative_l2']:.4g}，prompt KV 逐元素相同。")
    stable=[m for m in MODES[1:] if min(r["budget_ms"] for r in runs if r["mode"]==m)>0]
    txt+=["",f"五轮均超过 1.2× 的配置：{', '.join(stable) or '无'}。正预算是本 trace 整段 decode 的余量。合成 mask 不保持原 384 个位置的生成决策或原混合 KV 值，不能证明质量、在线 router 收益或在不改原位置情况下的调度可行性。"]
    (OUT/"report.md").write_text("\n".join(txt)+"\n")
    cfg=json.loads((BASE/"config.json").read_text())
    for f,h in cfg["source_hashes"].items():
        candidates=[Path(f)]
        if f=="benchmarks/free_kv_replay.py":candidates.append(Path("artifacts/b32-logits-replay/phase1_driver.py"))
        assert any(p.is_file() and sha(p)==h for p in candidates),f
    cfg.update(run="mixed-ffn-2b",modes=MODES,orders=[[MODES[(r+j)%4] for j in range(4)] for r in range(5)],
               stats=stats,inputs=load("inputs.json"),extra_source_hashes={str(p):sha(p) for p in map(Path,(
                   "benchmarks/concentrated_masks.py","benchmarks/concentrated_replay.py"))})
    dump("config.json",cfg)


if __name__=="__main__":main()
