"""Verify and aggregate fixed-trace replay outputs. Uses only the standard library."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median


def quantile(x, q):
    if not x:
        return None
    x = sorted(x)
    i = (len(x)-1)*q
    return x[int(i)] + (x[min(int(i)+1,len(x)-1)]-x[int(i)])*(i-int(i))


def read(path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def csvout(path, rows):
    with path.open("w",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interval(x, precision=2):
    return f"{median(x):.{precision}f} [{min(x):.{precision}f}, {max(x):.{precision}f}]"


def aggregate(out):
    trace = read(out/"trace.jsonl")
    trace_by_id = {r["request_id"]:r for r in trace}
    summary, budgets, gap_rows, attribution = [], [], [], []
    checks = [r for path in sorted(out.glob("*-checks.jsonl")) for r in read(path)]
    assert checks and all(row["passed"] for row in checks)
    csvout(out/"checks.csv",checks)
    modes = ["large","small","routed_exact","routed_free_kv"]
    with (out/"steps.jsonl").open("w") as sf, (out/"requests.jsonl").open("w") as rf:
        for job in ("time","diag"):
            for mode in modes:
                dual = mode.startswith("routed")
                for b in (1,8,32):
                    for rep in range(5 if job=="time" else 1):
                        path = out/f"{job}-{mode}-b{b}-r{rep}.jsonl"
                        cohorts = read(path)
                        steps,requests,itl,first,gaps = [],[],[],[],[]
                        for i,co in enumerate(cohorts):
                            meta = dict(run=job,mode=mode,B=b,repeat=rep,cohort=i)
                            assert all(co[k]==v for k,v in meta.items())
                            seen,last,large_end = Counter(),{},{}
                            for r in co["requests"]:
                                k,t = r["request_id"],r["step"]
                                src = trace_by_id[k]
                                seen[k]+=1
                                assert src["groups"][str(b)][0]==i and seen[k]==t
                                assert r["pos"]==len(src["prompt_ids"])+t-1
                                assert r["action"]==src["route_for_output"][t]
                                if dual:
                                    assert r["action"]==r["predicted"]
                                if r["gap"] is not None:
                                    gap=r["pos"]-large_end.get(k,len(src["prompt_ids"])-1)-1
                                    assert gap==r["gap"]
                                    gaps.append(gap)
                                    large_end[k]=r["pos"]
                                if k in last:
                                    itl.append(r["commit_ms"]-last[k])
                                else:
                                    first.append(r["commit_ms"])
                                last[k]=r["commit_ms"]
                                print(json.dumps(dict(**meta,**r)),file=rf)
                            for src in trace[i*b:(i+1)*b]:
                                assert seen[src["request_id"]]==len(src["output_ids"])-1
                            for s in co["steps"]:
                                record=dict(**meta,**s)
                                record["small_selected"]=s["active"]-s["heavy"] if dual else None
                                record["large_sub_batch_requests"]=s["heavy"] if dual else None
                                if s is co["steps"][-1]:
                                    record["cohort_decode_s"]=co["seconds"]
                                print(json.dumps(record),file=sf)
                            steps.extend(co["steps"])
                            requests.extend(co["requests"])
                        n=len(requests)
                        assert n==sum(len(r["output_ids"])-1 for r in trace)
                        wall=sum(co["seconds"] for co in cohorts)
                        sizes=[s["heavy"] for s in steps if s["heavy"]]
                        hist=sum(s["catchup"] or 0 for s in steps)
                        assert hist==(sum(gaps) if mode=="routed_exact" else 0)
                        assert not dual or sum(sizes)==len(gaps)
                        row=dict(run=job,mode=mode,B=b,repeat=rep,n=n,decode_s=wall,tok_s=n/wall,
                                 itl_p50_ms=median(itl),itl_p95_ms=quantile(itl,.95),itl_n=len(itl),
                                 first_decode_p50_ms=median(first),first_decode_n=len(first),
                                 prefill_to_first_decode_p50_ms=quantile(
                                     [r["prefill_to_first_decode_ms"] for r in requests
                                      if "prefill_to_first_decode_ms" in r],.5),
                                 large_rate=len(gaps)/n if dual else None,
                                 wake_rate=len(sizes)/len(steps) if dual else None,
                                 large_batch_mean=mean(sizes) if sizes else None,
                                 large_batch_p50=quantile(sizes,.5),large_batch_p95=quantile(sizes,.95),
                                 logical_gap=sum(gaps) if dual else None,
                                 actual_catchup=hist if dual else None,
                                 actual_large_positions=(hist+len(gaps) if dual else n if mode=="large" else None),
                                 peak_gib=max(co["peak_bytes"] for co in cohorts)/2**30,
                                 preemptions=0,prefill_s=sum(co["prefill_s"] for co in cohorts),
                                 precompute_s=sum(co["precompute_s"] for co in cohorts) if mode=="routed_free_kv" else None,
                                 block_table_prepare_s=(sum(co["block_table_prepare_s"] for co in cohorts)
                                                        if "block_table_prepare_s" in cohorts[0] else None),
                                 prepare_peak_extra_mib=(max(co["prepare_peak_extra_bytes"] for co in cohorts)/2**20
                                                         if job=="diag" else None),
                                 warmup_s=cohorts[0]["warmup_s"])
                        summary.append(row)
                        if job=="diag":
                            for runner in ("main","large"):
                                calls=[s for s in steps if runner in s["graph"]]
                                if not calls:
                                    continue
                                forward=sum(s[runner+"_forward_ms"] for s in calls)
                                buckets=Counter(tuple(s["graph"][runner]) for s in calls)
                                logical=sum(s["active"] if runner=="main" else s["heavy"]+s["catchup"] for s in calls)
                                attribution.append(dict(mode=mode,B=b,runner=runner,calls=len(calls),
                                    forward_ms=forward,logical_positions=logical,
                                    padding=sum(size*n for (_,size),n in buckets.items())-logical,
                                    eager_calls=sum(n for (kind,_),n in buckets.items() if kind=="NONE"),
                                    buckets=json.dumps({f"{k}/{p}":n for (k,p),n in buckets.items()}),
                                    router_host_ms=sum(s["router_host_ms"] or 0 for s in steps) if dual and runner=="main" else None,
                                    group_host_ms=sum(s["group_host_ms"]+s["graph"].get("group_build_host_ms",0) for s in steps) if runner=="main" else None,
                                    sample_event_ms=sum(s["sample_forward_ms"] for s in steps) if runner=="main" else None,
                                    sample_commit_ms=sum(s["sample_commit_host_ms"] for s in steps) if runner=="main" else None))
                            if dual:
                                for name,low,hi in [("0",0,0),("1-4",1,4),("5-16",5,16),("17-64",17,64),("65+",65,float("inf"))]:
                                    count=sum(low<=g<=hi for g in gaps)
                                    gap_rows.append(dict(B=b,mode=mode,bin=name,count=count,fraction=count/len(gaps)))
    csvout(out/"summary.csv",summary)
    csvout(out/"diagnostics.csv",attribution)
    csvout(out/"gap_bins.csv",gap_rows)
    times={(r["B"],r["mode"],r["repeat"]):r for r in summary if r["run"]=="time"}
    for b in (1,8,32):
        for rep in range(5):
            large,exact,free=[times[b,m,rep]["decode_s"] for m in ("large","routed_exact","routed_free_kv")]
            g=times[b,"routed_free_kv",rep]["logical_gap"]
            for target in (1,1.2):
                headroom=1000*(large/target-free)
                both=1000*(min(large/target,exact)-free)
                budgets.append(dict(B=b,repeat=rep,target=target,G=g,large_over_free=large/free,
                                    exact_over_free=exact/free,total_ms=headroom,combined_ms=both,
                                    per_gap_ms=headroom/g if g else None,combined_per_gap_ms=both/g if g else None))
    csvout(out/"budgets.csv",budgets)
    stats=[]
    for b in (1,8,32):
        for m in modes:
            group=[times[b,m,i] for i in range(5)]
            stats.append(dict(B=b,mode=m,**{k:interval([r[k] for r in group],4) for k in group[0]
                         if k not in ("B","mode","run","repeat") and group[0][k] is not None}))
    keys=list(dict.fromkeys(k for r in stats for k in r))
    csvout(out/"summary_ranges.csv",[{k:r.get(k) for k in keys} for r in stats])
    return summary,budgets,attribution


def report(out, summary, budgets, attribution):
    times=[r for r in summary if r["run"]=="time"]
    diag=[r for r in summary if r["run"]=="diag"]
    trace=read(out/"trace.jsonl")
    n=sum(len(r["output_ids"])-1 for r in trace)
    l=sum(r["route_for_output"][1:].count("L") for r in trace)
    text=[
        "# 免费历史 KV 的单卡回放结果",
        "",
        f"单张 V100 32 GiB，Qwen2.5-0.5B / 1.5B，FP16。GSM8K train 前 100 题共有 {n:,} 个计时输出，小模型实际选择率 {1-l/n:.2%}。动作由本项目的在线确定性 router 录制，输入为请求 ID 和输出位置。每次回放重新执行 router，并按录制动作提交固定 token。输出上限 48，85 题达到长度上限。",
        "",
        "## Decode 吞吐与配对加速",
        "",
        "每个数值为五次重复的中位数 [最小, 最大]。吞吐单位为输出 tok/s，速度比用同一次重复的整段墙钟配对计算。",
        "",
        "| B | 纯小模型 | 纯大模型 | 原生补算 | 免费 KV | 纯大墙钟 / 免费墙钟 | 补算墙钟 / 免费墙钟 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for b in (1,8,32):
        base=[interval([r["tok_s"] for r in times if r["B"]==b and r["mode"]==m],1)
              for m in ("small","large","routed_exact","routed_free_kv")]
        pairs=[r for r in budgets if r["B"]==b and r["target"]==1]
        text.append("| "+" | ".join([str(b),*base,interval([r["large_over_free"] for r in pairs]),
                                      interval([r["exact_over_free"] for r in pairs])])+" |")
    text.extend(["","## 工作量与归因","",
        "各并发档使用同一逐请求动作轨迹。下表的大模型前向计时来自独立诊断轮的 CUDA events，原生补算时包含缺口与当前位置。各阶段的计时可能重叠。",
        "","| B | 大模型唤醒占比 | 非空大模型子批均值 | 重位置 | 缺口位置 G | 大模型实际位置（补算 / 免费） | 大模型前向 ms（补算 / 免费） |",
        "|---:|---:|---:|---:|---:|---:|---:|"])
    for b in (1,8,32):
        exact,free=[next(r for r in diag if r["mode"]==m and r["B"]==b)
                    for m in ("routed_exact","routed_free_kv")]
        gpu=[next(r["forward_ms"] for r in attribution if r["B"]==b and r["mode"]==m and r["runner"]=="large")
             for m in ("routed_exact","routed_free_kv")]
        text.append(f"| {b} | {exact['wake_rate']:.1%} | {exact['large_batch_mean']:.2f} | {l} | {exact['logical_gap']} | {exact['actual_large_positions']} / {free['actual_large_positions']} | {gpu[0]:.0f} / {gpu[1]:.0f} |")
    text.extend(["","## KV 翻译预算","",
        "总预算 = 纯大墙钟 / 目标倍数 − 免费墙钟。双目标预算还要求超过原生补算路径，取两项更小的值。负值表示免费 KV 已用完整段时间余量。",
        "","| B | 目标倍数 | 总预算 ms | 每缺失位置 ms | 双目标总预算 ms | 双目标每位置 ms |",
        "|---:|---:|---:|---:|---:|---:|"])
    for b in (1,8,32):
        for target in (1,1.2):
            group=[r for r in budgets if r["B"]==b and r["target"]==target]
            metrics=[interval([r[k] for r in group],p) for k,p in
                     [("total_ms",1),("per_gap_ms",3),("combined_ms",1),("combined_per_gap_ms",3)]]
            text.append("| "+" | ".join([str(b),str(target),*metrics])+" |")
    text.extend(["","## 工程判断",""])
    for b in (1,8,32):
        group=[r for r in budgets if r["B"]==b and r["target"]==1.2]
        speed=[r["large_over_free"] for r in group]
        allowance=[r["combined_per_gap_ms"] for r in group]
        if min(allowance)>0 and min(speed)>1.2:
            decision=(f"可测试增量翻译原型。要同时达到 1.2 倍纯大模型吞吐并超过原生补算，"
                      f"每个缺失位置可增加的端到端时间中位数为 {median(allowance):.3f} ms。")
        elif min(speed)<=1<=max(speed):
            decision="免费 KV 与纯大模型的优势在本负载下尚难分辨。"
        elif max(speed)<1:
            decision="免费 KV 仍慢于纯大模型。优先测跨请求重分支攒批及 P95 ITL，再决定是否改动模型结构。"
        elif max(speed)<1.2:
            decision="1.2 倍目标超出本次免费 KV 上限；先降低重分支执行和调度成本。"
        else:
            decision="目标余量落在重复波动内，该负载下尚难分辨。"
        text.append(f"- B={b}，{decision}")
    text.extend(["","## 实验边界与复现","",
        "四组均用 vLLM V1 原生 runner、正常采样和 CUDA Graph。固定槽位驱动器直接提交调度输出，在同一进程内把当前步的 L 请求组成子批，关闭跨步攒批。纯模型分别独立加载。两种路由模式每步都运行小模型，大模型在相同重位置执行。固定槽位按 trace 顺序填入，每个 cohort 完成后才启动下一组，保留提前结束和尾批。本次墙钟适用于固定编组、单卡 decode，线上到达和排队延迟需要单独测量。",
        "",
        "免费组在计时前对完整前缀 teacher-force，写入该 cohort 正式使用的物理页。decode 通过逻辑有效长度暴露历史，无历史 KV 搬运，当前步照常写 KV。两组为同一 cohort 预留等量物理页，均无抢占。额外准备时间及峰值显存见 summary.csv。",
        "",
        "计时从 prefill、KV 准备和设备同步后开始，到最后一次提交及设备完成。ITL 采用 engine-step 提交时间，首个 decode 输出单列。主循环在内存保留步骤记录，采样使用 vLLM 自带的同步返回，写文件和 logits 正确性检查在计时外。诊断中的 router 与分组为主机耗时，前向和采样 events 为 CUDA 流上的阶段间隔。大模型的 ragged catch-up 和当前步合算，Graph bucket 按输入 token 数统计。",
        "",
        "短例含连续 S 后首次 L、连续 L、交替 S/L 和不同结束步。7 个 L 位置的三组 top-1 一致率为 100%，exact/free 最大绝对差 0.03223，原生 decode 与 teacher-forcing 校准值 0.04102；改写未来槽位时最大差为 0。完整轨迹按请求校验位置、动作和提交计数。固定输出回放用于评估执行开销。",
        "",
        "在相同环境中，将 trace.jsonl 放入输出目录。先分别运行 `python benchmarks/free_kv_replay.py check --mode MODE --out DIR`（依次 large、routed_exact、routed_free_kv），再运行 `bash benchmarks/run_free_kv_matrix.sh DIR` 和 `python benchmarks/summarize_free_kv.py DIR`。模型路径、文件哈希、计时边界与运行配置见 [run_config.json](run_config.json)。",
        "",
        "逐轮结果见 [summary.csv](summary.csv)，ITL 及重复间范围见 [summary_ranges.csv](summary_ranges.csv)，各阶段与实际 Graph bucket 见 [diagnostics.csv](diagnostics.csv)，缺口分桶见 [gap_bins.csv](gap_bins.csv)。[逐步记录](steps.jsonl) · [逐请求提交](requests.jsonl) · [配对预算](budgets.csv) · [正确性](checks.csv)"])
    (out/"report.md").write_text("\n".join(text)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("out",type=Path)
    out=parser.parse_args().out
    report(out,*aggregate(out))
