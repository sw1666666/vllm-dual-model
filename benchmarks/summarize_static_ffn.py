"""Rebuild stage-one timing, paired budgets and six-row table from raw runs."""
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path

OUT=Path("artifacts/static-ffn-v1")
widths=(8960,4480,2304)
median=statistics.median


def read(p):
    return json.loads(p.read_text())


def write(name,data):
    with (OUT/name).open("w",newline="") as f:
        w=csv.DictWriter(f,list(data[0]))
        w.writeheader();w.writerows(data)


def span(vals, digits=1):
    return f"{median(vals):.{digits}f} [{min(vals):.{digits}f}, {max(vals):.{digits}f}]"


def compiled_graphs():
    # Verify the graph used by the real vLLM decoder, in addition to the
    # independent profiler run of the loaded MLP. Do not instrument timing.
    matrix_log=OUT/"engine.log"
    if not matrix_log.exists():matrix_log=Path("/root/autodl-tmp/logs/static-ffn-matrix.log")
    compiled_path=OUT/"compiled_graphs.json"
    if matrix_log.exists():
        compiled={}
        for line in matrix_log.read_text().splitlines():
            if "Starting to load model " in line:
                m=re.search(r"static-ffn-(\d+)",line)
                w=int(m[1]) if m else 8960
            m=re.search(r"Using cache directory: (\S+)",line)
            if not m or w in compiled:continue
            graph=Path(m[1])/"computation_graph.py"
            src=graph.read_text()
            for i in range(28):
                prefix=f"L_self_modules_layers_modules_{i}_modules_"
                for part,dimensions in (("mlp_modules_gate_up_proj",[2*w,1536]),("mlp_modules_down_proj",[1536,w]),
                                        ("self_attn_modules_qkv_proj",[2048,1536]),("self_attn_modules_o_proj",[1536,1536])):
                    v=prefix+part+"_parameters_weight_"
                    assert f'{v}: "f16{dimensions}"' in src,(w,i,v)
                    assert re.search(r'torch\._C\._nn\.linear\([^\n;]*\b'+v.lower()+r'\b',src),(w,i,v)
            compiled[w]=dict(path=str(graph),sha256=hashlib.sha256(graph.read_bytes()).hexdigest(),validated_layers=28,
                             qkv=[2048,1536],attn_out=[1536,1536],gate_up=[2*w,1536],down=[1536,w])
        assert len(compiled)==3
        compiled_path.write_text(json.dumps(compiled,indent=2))
    return read(compiled_path)


def main():
    prev=read(Path("artifacts/free-kv-v1/run_config.json"))
    trace_file=Path("artifacts/free-kv-v1/trace.jsonl")
    assert hashlib.sha256(trace_file.read_bytes()).hexdigest()==prev["trace_sha256"]
    trace=[json.loads(line) for line in trace_file.read_text().splitlines()]
    raw={}
    for r in range(5):
        for w in widths:
            for b in (1,32):
                x=read(OUT/f"time-{r}-{w}-{b}.json")
                c=x.pop("cohorts")
                assert len(c)==(100 if b==1 else 4) and x["N"]==sum(t["N"] for t in c)==4091
                assert abs(x["wall_ms"]-sum(t["seconds"]*1000 for t in c))<1e-6
                intervals=[]
                for i,t in enumerate(c):
                    group=trace[i*b:(i+1)*b];last={}
                    want=[(item["request_id"],step) for step in range(1,max(len(item["output_ids"]) for item in group))
                          for item in group if step<len(item["output_ids"])]
                    assert [(rid,step) for rid,step,stamp in t["commits"]]==want
                    for rid,step,stamp in t["commits"]:
                        if rid in last:intervals.append(stamp-last[rid])
                        last[rid]=stamp
                assert len(intervals)==x["itl_n"]==3991 and min(intervals)>=0
                s=sorted(intervals);p=.95*(len(s)-1);j=int(p)
                assert abs(s[j]+(p-j)*(s[j+1]-s[j])-x["itl_p95_ms"])<1e-8
                raw[r,w,b]=x
    checks=list(csv.DictReader((OUT/"checks.csv").open()))
    assert len(checks)==84 and all(r["passed"]=="True" for r in checks)
    diag={w:read(OUT/f"diag-{w}.json") for w in widths}
    same_graph=all(d["graph"]==diag[8960]["graph"] for d in diag.values())
    compiled=compiled_graphs()
    out=[];budgets=[]
    for (r,w,b),x in raw.items():
        full=raw[r,8960,b]["wall_ms"];wall=x["wall_ms"]
        out.append(dict(**x,paired_speedup=full/wall,paired_saved_ms=full-wall))
        if b==32 and w!=8960:
            budgets.append(dict(repeat=r,width=w,break_even_ms=full-wall,target_1_2_ms=full/1.2-wall))
    write("timings.csv",out);write("budgets.csv",budgets)
    text=["# 静态 FFN 宽度：阶段一","",
          "同一 V100、Qwen2.5-1.5B-Instruct、FP16，正常 28 层 attention；同一固定 trace，强制提交不衡量质量。墙钟、吞吐与 P95 是五次中位数，加速和节省先按轮配对。",
          "","| B | FFN 宽度 | 有效输出 | decode ms | tok/s | 相对 full 加速 [范围] | P95 ITL ms | 省下 ms |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for b in (1,32):
        for w in widths:
            xs=[x for x in out if x["width"]==w and x["B"]==b]
            vals=lambda k:[x[k] for x in xs]
            text.append(f"| {b} | {w} | 4091 | {median(vals('wall_ms')):,.1f} | {median(vals('tok_s')):,.1f} | {span(vals('paired_speedup'),3)} | {median(vals('itl_p95_ms')):.2f} | {median(vals('paired_saved_ms')):,.1f} |")
    text+=["","B=32 恢复宽通道、分组和回填的整段额外墙钟预算（ms，中位数 [五轮范围]）：","",
           "| 宽度 | 打平 | 保持 1.2× |","|---:|---:|---:|"]
    for w in widths[1:]:
        g=[x for x in budgets if x["width"]==w]
        text.append(f"| {w} | {span([x['break_even_ms'] for x in g])} | {span([x['target_1_2_ms'] for x in g])} |")
    diff=[raw[r,4480,32]["wall_ms"]-raw[r,2304,32]["wall_ms"] for r in range(5)]
    text+=["",f"B=32 下 2304 相对 4480 额外省下 {span(diff)} ms。"]
    text.append(f"诊断：各档 graph 分派是否一致：{same_graph}。B=32 检查四个 cohort 的所有解码步，B=1 检查唯一形状。无抢占（固定并独占 KV 页）；每个宽度的 28 层 KV 写入和连续四步有限 logits 已通过，实测原生 MLP 的 GEMM 操作数已存入 diag-*.json。")
    text.append("单模型在自身宽度 prefill；prefill 时间另列 timings.csv。ITL 在同一请求的 decode 提交之间取样（每格 3,991 个间隔），不含首 token。未执行阶段二、三。")
    (OUT/"report.md").write_text("\n".join(text)+"\n")
    config={k:prev[k] for k in ("gpu","driver","fork_checkout","dtype","attention","quantization","tp","pp","block_size","max_num_seqs","max_model_len","max_num_batched_tokens","single_cache_mib","sampling","graph","torch","triton","transformers","trace_sha256")}
    config.update(source=prev["large_model"],models=read(OUT/"models.json"),source_config_sha256=prev["large_config_sha256"],
        kernel_shapes={w:d["gemm_ops"] for w,d in diag.items()},compiled_graphs=compiled,graph_same=same_graph,
        measured_graph={w:d["graph"] for w,d in diag.items()},
        benchmark="static-ffn-v1",layers=28,hidden=1536,widths=widths,
        slice="each HF gate/up first w rows and down first w columns; all other checkpoint tensors unchanged",
        numerical="FP16 masked-full against exported narrow; additional same-shaped projection and FP32 algebra checks. relative_l2 is norm(diff)/norm(ref).",
        trace="artifacts/free-kv-v1/trace.jsonl",orders=[[widths[(r+j)%3] for j in range(3)] for r in range(5)],
        schedule="fixed trace cohort and slot index, B=1/32; no route, no admission/queue; one GPU model per process",
        timing="synchronize before and after contiguous cohort decode, engine-step commit timestamps, exclude prefill/load/log writes; warm B=1 on first request, B=32 all cohorts each measurement; B order alternates by round",
        preemptions=0,source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in map(Path,(
            "benchmarks/free_kv_replay.py","benchmarks/static_ffn_replay.py","benchmarks/export_static_ffn.py",
            "vllm/v1/worker/gpu_model_runner.py","vllm/model_executor/models/qwen2.py"))})
    (OUT/"config.json").write_text(json.dumps(config,indent=2)+"\n")


if __name__=="__main__":
    main()
