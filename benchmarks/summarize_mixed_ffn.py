"""Rebuild shared-attention stage-two tables from commit logs and CUDA events."""
import csv
import hashlib
import json
import statistics as st
from pathlib import Path

P=Path("artifacts/mixed-ffn-v1")


def load(name):
    return json.loads((P/name).read_text())


def csvfile(name,xs):
    with (P/name).open("w",newline="") as f:
        w=csv.DictWriter(f,list(xs[0]));w.writeheader();w.writerows(xs)


def med(x):return st.median(x)


def span(x,d=1):
    return f"{med(x):.{d}f} [{min(x):.{d}f}, {max(x):.{d}f}]"


def main():
    tracefile=Path("artifacts/free-kv-v1/trace.jsonl")
    trace=[json.loads(l) for l in tracefile.read_text().splitlines()]
    groups=[trace[i:i+32] for i in range(0,len(trace),32)]
    stages=load("steps.json")
    lookup={r["request_id"]:r for r in trace}
    for s in stages:
        g=groups[s["cohort"]];t=s["step"]
        ids=s["physical_ids"]
        assert sorted(ids)==sorted(r["request_id"] for r in g if t<len(r["output_ids"]))
        assert s["wide_ids"]==[rid for rid in ids if lookup[rid]["route_for_output"][t]=="L"]
        assert s["graph"]=="FULL"
    mixed=[s for s in stages if s["mode"]=="M"]
    assert sum(s["k"] for s in mixed)==384 and sum(s["k"]>0 for s in mixed)==150
    hist=[]
    for k in sorted(set(s["k"] for s in mixed)):
        g=[s for s in mixed if s["k"]==k]
        total=lambda x:sum(s[x] for s in g)
        h=dict(k=k,steps=len(g),valid_E_rows=total("k"),executed_E_rows=total("E_rows"),extra_pad_rows=total("E_rows")-total("k"))
        for name in ("gather_ms","E_ms","scatter_ms"):h[name]=total(name)
        h["E_ms_per_step"]=h["E_ms"]/len(g)
        hist.append(h)
    csvfile("wide_batches.csv",hist)
    results=[];pairs=[]
    for rep in range(5):
        times={}
        for m in "FCWM":
            cs=load(f"time-{rep}-{m}.json");wall=sum(c["wall_s"] for c in cs)*1000
            itls=[];N=0
            assert len(cs)==len(groups)
            for c,g in zip(cs,groups):
                expected=[(r["request_id"],t,len(r["prompt_ids"])+t-1) for t in range(1,max(len(r["output_ids"]) for r in g)) for r in g if t<len(r["output_ids"])]
                assert expected==[(rid,t,pos) for rid,t,pos,stamp in c["commits"]]
                last={}
                for rid,t,pos,stamp in c["commits"]:
                    if rid in last:itls.append(stamp-last[rid])
                    last[rid]=stamp;N+=1
            assert N==4091 and len(itls)==3991
            a=sorted(itls);j=(len(a)-1)*.95;i=int(j)
            times[m]=wall
            counts=[s["graph"] for s in stages if s["mode"]==m]
            results.append(dict(repeat=rep,mode=m,N=N,wall_ms=wall,tok_s=N*1000/wall,
                speedup=times["F"]/wall,P95_ms=a[i]+(a[i+1]-a[i])*(j-i),ITL_samples=len(a),
                full_graph_steps=counts.count("FULL"),eager_steps=counts.count("NONE")))
        f,c,w,m=[times[x] for x in "FCWM"]
        pairs.append(dict(repeat=rep,M_minus_C=m-c,W_minus_F=w-f,F_minus_M=f-m,
                          F_div_1_2_minus_M=f/1.2-m,break_even_budget_from_C=f-c,
                          target_1_2_budget_from_C=f/1.2-c))
    csvfile("timings.csv",results);csvfile("pairs.csv",pairs)
    txt=["# 共享 attention，2304 + 条件 6656 FFN","",
      "V100 / Qwen2.5-1.5B-Instruct / FP16；B=32、4,091 个固定 decode 输出、五轮配对。墙钟不含预热、捕图、原生全宽 prefill。无等待队列、无在线 router。","",
      "| 模式 | decode ms | tok/s | 相对 F 加速 [五轮范围] | P95 ITL ms | FULL/eager 步数 |",
      "|---|---:|---:|---:|---:|---:|"]
    for m in "FCWM":
        g=[x for x in results if x["mode"]==m]
        v=lambda c:[x[c] for x in g]
        txt.append(f"| {m} | {med(v('wall_ms')):.1f} | {med(v('tok_s')):.1f} | {span(v('speedup'),3)} | {med(v('P95_ms')):.2f} | {g[0]['full_graph_steps']}/{g[0]['eager_steps']} |")
    txt+=["","配对差值，ms，中位数 [最小, 最大]："]
    for key in list(pairs[0])[1:]:txt.append(f"- {key}：{span([x[key] for x in pairs])}")
    txt+=["","| M 的 k | 步数 | 有效/实际 E 行 | 取行 ms | 6656 FFN ms | 加回 ms |",
          "|---:|---:|---:|---:|---:|---:|"]
    for h in hist:
        txt.append(f"| {h['k']} | {h['steps']} | {h['valid_E_rows']}/{h['executed_E_rows']} | {h['gather_ms']:.2f} | {h['E_ms']:.2f} | {h['scatter_ms']:.2f} |")
    txt+=["","设备事件在独立诊断图内记录；以上为该桶各步、28 层累计，包含事件间隔开销，不用来替代墙钟。M 按实际 k 执行；主干用原 vLLM 填充桶。W 对主干填充行也执行两段。固定动作来自 trace，计时内由物理行和输入位置取动作、生成索引并异步上传一次，28 层共用同一索引缓冲。",""]
    with (P/"checks.csv").open() as f:checks=list(csv.DictReader(f))
    assert len(checks)==6 and all(r["passed"]=="True" for r in checks)
    txt+=["| 参考 / 被测 | 对象 | 比较数 | 最大绝对误差 | 最大相对 L2 | top-1 | 最大绝对误差的请求 / 位置 / 层 |",
          "|---|---|---:|---:|---:|---:|---|"]
    for a in checks:
        txt.append(f"| {a['pair']} | {a['type']} | {a['n']} | {float(a['max_abs']):.6g} | {float(a['max_rel']):.4g} | {a['top1'] or 'N/A'} | {a['worst_request']} / {a['worst_pos']} / {a['worst_layer'] or '—'} |")
    txt+=["","短回放每组的 prompt KV 逐元素相同。诊断对每个待写 decode 槽先放 NaN，确认同一 trace 的 4,091 个实际位置、全部 28 层 K/V 均被写为有限值；各步物理宽请求 ID 与固定动作独立核对。",""]
    bs=[a["F_minus_M"] for a in pairs];ts=[a["F_div_1_2_minus_M"] for a in pairs]
    ctime=med([a["wall_ms"] for a in results if a["mode"]=="C"])
    txt.append(f"C 相对上轮 2304 的 1,165.4 ms 改变 {ctime-1165.4:+.1f} ms（跨批次参照）。")
    if min(ts)>0:txt.append("M 每轮均超过 1.2×；router 等工作必须落在本轮剩余预算内。")
    elif min(bs)>0 and max(ts)<0:txt.append("M 每轮均快于 F，但每轮均未达到 1.2×。")
    elif max(bs)<0:txt.append("M 每轮均慢于 F，更未达到 1.2×；本实现没有在线 router 的正预算。")
    else:txt.append("对应阈值在重复间未稳定分离，不能声称稳定越过。")
    txt.append("本轮未训练 router、未评估生成质量。没有 DRAM 硬件计数读数，不把理论权重字节数当作实测流量。")
    E=sum(h["E_ms"] for h in hist);gather=sum(h["gather_ms"] for h in hist);scat=sum(h["scatter_ms"] for h in hist)
    peak=max(hist,key=lambda h:h["E_ms"])
    txt.append(f"独立诊断的互补 FFN、取行、回填分别累计 {E:.1f}、{gather:.1f}、{scat:.1f} ms。k={peak['k']} 桶的互补 FFN 累计最大（{peak['E_ms']:.1f} ms）。")
    if min(bs)>0 and max(ts)<0:
        txt.insert(2,f"**结论：五轮均打平并超过 F，但未达到 1.2×。** 打平余量中位数 {med(bs):.1f} ms，1.2× 预算 {med(ts):.1f} ms。\n")
        txt.append("建议下一次固定同一 trace，只改 k=1、2 的互补 FFN 小批实现，以 1.2× 的剩余墙钟检验；本轮没有实施该优化。")
    cfg=json.loads(Path("artifacts/static-ffn-v1/config.json").read_text())
    cfg={k:cfg[k] for k in ("gpu","source","dtype","attention","trace_sha256","driver","fork_checkout","graph","sampling","block_size","tp","pp")}
    cfg.update(run="mixed-ffn-v1",native_model="original Instruct checkpoint; same stock compiled prefill and one KV pool for all modes",
        orders=["".join("FCWM"[(i+r)%4] for i in range(4)) for r in range(5)],B=32,channels=[2304,6656],layers=28,
        preemptions=0,backend="stock vLLM runner, FULL decoder CUDA graphs; native attention and non-FFN modules shared, split backbone separately compiled",
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in map(Path,("benchmarks/mixed_ffn_engine.py","benchmarks/mixed_ffn_replay.py","benchmarks/free_kv_replay.py"))})
    (P/"config.json").write_text(json.dumps(cfg,indent=2))
    (P/"report.md").write_text("\n".join(txt)+"\n")


if __name__=="__main__":main()
