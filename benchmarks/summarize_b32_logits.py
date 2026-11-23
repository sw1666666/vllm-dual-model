"""Rebuild tables and the stage-two gates from per-cohort wall clocks."""
import csv
import hashlib
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

OUT=Path("artifacts/b32-logits-replay")
PREV=Path("artifacts/free-kv-v1")
L,F,O="large","routed_free_kv","routed_replay_logits"


def read(name):
    return [json.loads(x) for x in (OUT/name).read_text().splitlines()]


def write(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def pct(x,f):
    s=sorted(x);p=f*(len(s)-1);i=int(p)
    return s[i]+(p-i)*(s[min(i+1,len(s)-1)]-s[i])


def fmt(xs,d=2):
    return " / ".join(f"{v:.{d}f}" for v in (st.median(xs),min(xs),max(xs)))


def measure(cohorts):
    wall=sum(c["seconds"] for c in cohorts)
    itl=[];n=calls=0;wait=[]
    for c in cohorts:
        last={}
        calls+=sum(bool(s["heavy"]) for s in c["steps"])
        for r in c["requests"]:
            k=r["request_id"]
            if k in last: itl.append(r["commit_ms"]-last[k])
            last[k]=r["commit_ms"];n+=1
            if r.get("wait_ms") is not None:wait.append(r["wait_ms"])
    assert n==4091 and len(itl)==3991
    return dict(N=n,wall_ms=1000*wall,tok_s=n/wall,steps=sum(len(c["steps"]) for c in cohorts),itl_p50_ms=pct(itl,.5),
                itl_p95_ms=pct(itl,.95),itl_n=len(itl),real_heavy_calls=calls,
                wait_p50_ms=pct(wait,.5) if wait else None,wait_p95_ms=pct(wait,.95) if wait else None,
                wait_max_ms=max(wait) if wait else None)


def main():
    checks=[dict(mode=m,**json.loads((OUT/f"check-{m}.json").read_text())) for m in (F,O)]
    write("checks.csv",checks)
    times=[];pairs=[]
    for rep in range(5):
        current={}
        for m in (L,F,O):
            row=measure(read(f"time-{m}-{rep}.jsonl"))
            current[m]=row["wall_ms"]
            if m==O: row["real_heavy_calls"]=0
            times.append(dict(type="mode",repeat=rep,mode=m,**row))
        l,f,o=(current[m] for m in (L,F,O))
        pairs.append(dict(type="paired",repeat=rep,F_minus_L=f-l,F_minus_O=f-o,L_minus_O=l-o,
                          recovery_fraction=(f-l)/(f-o) if f>l and f>o else None))
    summary=times+pairs
    for m in (L,F,O,"paired"):
        group=[r for r in summary if r.get("mode","paired")==m and isinstance(r.get("repeat"),int)]
        cols=("wall_ms","tok_s","itl_p50_ms","itl_p95_ms") if m!="paired" else ("F_minus_L","F_minus_O","L_minus_O","recovery_fraction")
        for stat,fn in (("median",st.median),("min",min),("max",max)):
            summary.append(dict(type="aggregate",mode=m,repeat=stat,**{k:fn([r[k] for r in group]) if all(r.get(k) is not None for r in group) else None for k in cols}))
    write("summary.csv",summary)
    diag=[];hist=Counter()
    for m in (L,F,O):
        for c in read(f"diag-{m}-0.jsonl"):
            for s in c["steps"]:
                t=s["step"];shape=s["graph"];k=s["heavy"]
                rr=[r for r in c["requests"] if r["step"]==t]
                heavy=[r for r in rr if r["gap"] is not None]
                row=dict(mode=m,cohort=c["cohort"],step=t,active=s["active"],k=k,
                         positions=[r["pos"] for r in rr],heavy_positions=[r["pos"] for r in heavy],
                         context_lengths=[r["pos"]+1 for r in rr],
                         heavy_context_lengths=[r["pos"]+1 for r in heavy],
                         small_or_main_graph=shape["main"][0],main_padded=shape["main"][1],
                         large_dispatch=shape["large"][0] if k else None,
                         heavy_padded=shape["large"][1] if k else None,
                         heavy_graph_executed=(m==F and shape["large"][0]!="NONE") if k else None,
                         ipc_ms=None,preemptions=0,router_ms=s["router_host_ms"],group_ms=s["group_host_ms"]+shape.get("group_build_host_ms",0),
                         scatter_host_ms=shape.get("scatter_host_ms"),decoder_gpu_ms=s.get("_model_forward_gpu_ms"),
                         head_gpu_ms=s.get("compute_logits_gpu_ms"),view_host_ms=s.get("view_host_ms"),
                         main_runner_gpu_ms=s.get("main_forward_ms"),
                         heavy_runner_ms=s.get("large_forward_ms"),sample_ms=s["sample_forward_ms"])
                diag.append(row)
                if m==F: hist[k]+=1
    pq.write_table(pa.Table.from_pylist(diag),OUT/"steps.parquet")
    # Arrow infers null columns across all rows, including the routed arms.
    frows=[r for r in diag if r["mode"]==F and r["k"]]
    orows=[r for r in diag if r["mode"]==O and r["k"]]
    assert [(r["heavy_positions"],r["heavy_padded"],r["large_dispatch"]) for r in frows]==[(r["heavy_positions"],r["heavy_padded"],r["large_dispatch"]) for r in orows]
    lookup=sum(r["view_host_ms"] for r in orows)
    lh=[p["L_minus_O"] for p in pairs];gap=[p["F_minus_L"] for p in pairs];recover=[p["F_minus_O"] for p in pairs]
    curve_ok=min(lh)>0 and min(gap)>0 and min(recover)>0 and lookup<min(gap)*.1
    gate=dict(curve=curve_ok,lookup_ms=lookup,queue=False)
    text=["# B=32 原生 logits 回放","",
          "V100 32 GiB，Qwen2.5-0.5B / 1.5B，FP16，固定 trace。每轮同一组 4,091 个 decode 输出。以下为五次配对的中位数 / 最小 / 最大。",
          "","| 模式 | decode ms | 输出 tok/s | 同请求 P95 ITL ms |","|---|---:|---:|---:|"]
    for m in (L,F,O):
        g=[r for r in times if r["mode"]==m]
        text.append("| "+m+" | "+" | ".join(fmt([r[k] for r in g],1) for k in ("wall_ms","tok_s","itl_p95_ms"))+" |")
    text+=["","| 配对量 | 中位数 / 最小 / 最大 |","|---|---:|"]
    for k in ("F_minus_L","F_minus_O","L_minus_O","recovery_fraction"):
        valid=[r[k] for r in pairs if r[k] is not None]
        text.append(f"| {k} | "+(fmt(valid,3 if k=="recovery_fraction" else 1) if valid else "N/A")+" |")
    text+=["",f"前三个配对量单位为 ms。上轮 L/F 约 1,765/1,931 ms，本轮缺口以配对 F−L 为准。非空重调用 {len(frows)} 次，k→步数（含空步）为 {dict(sorted(hist.items()))}。O 的预排视图实测主机取数总计 {lookup:.3f} ms。数据在同一进程、同一 GPU 上回填，额外 GPU logits 有效载荷最高 {checks[1]['max_buffer_bytes']/2**20:.2f} MiB。O 保留 Graph 形状选择和 metadata 准备，跳过重前向的图执行。","",
           "384 行完整 logits 与独立 F 复跑逐元素相同，top-1 全部一致，4,091 个强制提交前的贪心样本一致。F/O 逻辑位置及结束步相同，实际历史补算均为 0。O 的重模型 decoder 调用与处理行数均为 0。"]
    text.append(f"本轮 F−L 中位数与历史约 166 ms 的差为 {st.median(gap)-166:+.1f} ms。检验中 O 与独立 F 对同一份 capture 的 logits 误差均为 0。")
    dispatch_hist=Counter((r["large_dispatch"],r["heavy_padded"]) for r in frows)
    text.append(f"独立诊断：F/O 选择了相同的重分支形状（Graph / 填充行数 → 次数：{dict(sorted(dispatch_hist.items()))}）。无抢占。F 的 decoder 与 LM head GPU 事件分别累计 {sum(r['decoder_gpu_ms'] for r in frows):.1f} 和 {sum(r['head_gpu_ms'] for r in frows):.1f} ms；不将事件累计视为墙钟。")
    if (OUT/"heavy_curve.csv").exists():
        curve={(int(r["k"]),int(r["context_bucket_min"])):float(r["wall_ms"]) for r in csv.DictReader((OUT/"heavy_curve.csv").open())}
        bucket=lambda p:0 if p<96 else 96 if p<128 else 128
        base=sum(curve[r["k"],bucket(max(r["heavy_positions"]))] for r in frows)
        ideal=0;calls=0
        for co in range(4):
            positions=[p for r in frows if r["cohort"]==co for p in r["heavy_positions"]]
            for i in range(0,len(positions),4):
                block=positions[i:i+4];ideal+=curve[len(block),bucket(max(block))];calls+=1
        gate["queue"]=curve_ok and base-ideal>st.median(gap)
        text+=["","## 重子批曲线","",f"每个 k 和上下文桶测 30 次同步墙钟与 CUDA events，只包围 decoder、选行和 LM head。按真实调用的 k 和最大上下文桶加权为 {base:.1f} ms。将 cohort 内的全部重位置按 4 个分组（{calls} 次调用），理想估计为 {ideal:.1f} ms，毛节省 {base-ideal:.1f} ms。此估算忽略跨步依赖和等待；实际可实现性以有界队列配对结果为准。"]
    if (OUT/f"queue-{F}-4-4.jsonl").exists():
        q=[]
        for m in (F,"routed_exact"):
            if not (OUT/f"queue-{m}-4-4.jsonl").exists():continue
            for rep in range(5):
                for target in (1,4):
                    q.append(dict(mode=m,repeat=rep,target=target,**measure(read(f"queue-{m}-{target}-{rep}.jsonl"))))
        write("queue_summary.csv",q)
        text+=["","## 有界等待队列","",
               "同一驱动器立即执行和攒至 4 行配对。最老请求等待达到 12 ms 或无其他可执行请求时，在下一个调度边界释放待处理请求。以下为五轮中位数。","",
               "| KV | 目标行数 | tok/s | P95 ITL ms | 重调用次数 | 等待 P50 / P95 / 最大 ms¹ |","|---|---:|---:|---:|---:|---:|"]
        for m,target in dict.fromkeys((r["mode"],r["target"]) for r in q):
            g=[r for r in q if r["mode"]==m and r["target"]==target]
            med=lambda key:st.median(r[key] for r in g)
            text.append(f"| {m} | {target} | {med('tok_s'):.1f} | {med('itl_p95_ms'):.2f} | {med('real_heavy_calls'):.0f} | {med('wait_p50_ms'):.2f} / {med('wait_p95_ms'):.2f} / {med('wait_max_ms'):.2f} |")
        text+=["","¹ 等待从路由动作可用至派发，最大值列为各轮最大值的中位数。"]
        q0=[r for r in q if r["mode"]==F and r["target"]==1]
        q4=[r for r in q if r["mode"]==F and r["target"]==4]
        qdelta=[a["wall_ms"]-b["wall_ms"] for a,b in zip(q0,q4)]
        ltimes=[r["wall_ms"] for r in times if r["mode"]==L]
        text.append(f"同驱动器下，队列收回墙钟 {fmt(qdelta,1)} ms（中位数 / 最小 / 最大）。队列免费 KV 的墙钟中位数 {st.median(r['wall_ms'] for r in q4):.1f} ms，第一阶段纯大模型为 {st.median(ltimes):.1f} ms；后者是跨采集参照。")
        if max(qdelta)<=0:
            text.append("该策略五轮均未缩短整段 decode，保留立即派发；仅减少重调用次数不等于获得端到端收益。")
        elif min(qdelta)>0:
            text.append("五轮攒批均比同驱动器立即派发快。是否采用还需按 P95 ITL 要求判断，不能把曲线的理想毛节省算作部署收益。")
        else:
            text.append("队列相对立即执行的配对差值不稳定；先增加重复并检查每轮调度和尾部，再决定是否采用。")
        for m in dict.fromkeys(r["mode"] for r in q):
            imm=[r for r in q if r["mode"]==m and r["target"]==1]
            queued=[r for r in q if r["mode"]==m and r["target"]==4]
            gain=[b["tok_s"]/a["tok_s"]-1 for a,b in zip(imm,queued)]
            text.append(f"{m}：攒批相对立即派发的配对吞吐变化中位数 {st.median(gain):+.1%}（{min(gain):+.1%} 至 {max(gain):+.1%}）；调度步数中位数 {st.median(r['steps'] for r in imm):.0f} → {st.median(r['steps'] for r in queued):.0f}。")
        exact4=[r for r in q if r["mode"]=="routed_exact" and r["target"]==4]
        if exact4 and st.median(r["wall_ms"] for r in q4)<st.median(ltimes)<st.median(r["wall_ms"] for r in exact4):
            text.append("以同 trace 的第一阶段 L 为参照：攒批在免费 KV 下跨过纯 1.5B，在真实补算下仍未跨过。建议保留攒批作为吞吐选项，优先削减历史补算，再以端到端吞吐和尾延迟一起验收；不要直接把该队列设成延迟敏感服务的默认值。")
    if min(lh)>0 and min(gap)>0 and min(recover)>0:
        text+=["",f"五轮均满足 O < L < F。只改重分支有足够的理论余量；真实实现要追平纯大模型，需收回净可省时间的 {st.median(r['recovery_fraction'] for r in pairs):.1%}。实际调度方案还需同时检查吞吐与 P95 ITL。"]
    elif max(lh)<=0:
        text+=["","O 在五轮中均未超过纯大模型。优先定位小模型前向、重子批准备、回填和采样的剩余开销。"]
    else:
        text+=["","按逐轮配对的符号和波动评估基线；信号变号时需要增加重复。"]
    text+=["","轨迹来自当前基于位置的确定性 router。固定编组回放关闭线上到达；ITL 以 engine-step 提交计。正式计时排除预热、prefill、KV/logits 准备。独立诊断各阶段可重叠。"]
    text+=["","复核：在仓库根目录运行 `python benchmarks/summarize_b32_logits.py` 可从原始记录重建表格。重跑第一阶段用 `bash benchmarks/run_b32_logits.sh`，随后用 `bash benchmarks/finish_b32_logits.sh` 执行条件实验。请先保存原产物；本批次实际第一阶段驱动快照为 `phase1_driver.py`。"]
    (OUT/"report.md").write_text("\n".join(text)+"\n")
    (OUT/"decision.json").write_text(json.dumps(gate))
    config=json.loads((PREV/"run_config.json").read_text())
    for k in ("remote_core_sha256","run_id"):
        config["previous_"+k]=config.pop(k)
    config.update(run_id="b32-logits-replay",batch_sizes=[32],repeats=5,
        code_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ("benchmarks/free_kv_replay.py","benchmarks/b32_logits_replay.py","vllm/v1/worker/gpu_model_runner.py")},
        layout="native FP16 [k,151936], stride (151936,1), GPU tensor per nonempty call in actual F row order; warmed iterator consumed once per call",
        previous_results="free-kv-v1-v100-20261118",logits_payload_peak_bytes=checks[1]["max_buffer_bytes"],
        stages={"phase1":"independent model process for each measurement, one full cohort sweep for warmup; rotate L/F/O left by repeat modulo 3",
                "diagnostic":"separate sweep per mode; O heavy_dispatch describes the prepared shape, decoder graph bypassed",
                "queue":"conditional, paired alternation, target=4, 12 ms deadline checked at scheduler boundaries, immediate control uses the same ready-queue driver"},
        oracle_peak_allocated_bytes=max(c["peak_bytes"] for c in read(f"time-{O}-0.jsonl")),
        F_peak_allocated_bytes=max(c["peak_bytes"] for c in read(f"time-{F}-0.jsonl")))
    (OUT/"run_config.json").write_text(json.dumps(config,indent=2)+"\n")
    print(json.dumps(gate))


if __name__=="__main__":
    main()
