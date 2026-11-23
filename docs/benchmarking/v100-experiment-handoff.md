# V100 上的 vLLM 路由与条件 FFN 实验交接

交接日期：2026-11-22。已完成阶段 2B。**在 B=32 的固定回放上，把 384 个宽位置集中到 80 个 decode 步，五轮均超过原生全宽的 1.2×，但这不是已实现的在线调度收益。** M80 只保留了原始 384 个宽位置中的 31 个。下一项待验证的是：不移动任何请求的宽位置，只改变不同请求之间的执行顺序，能否保住加速和尾延迟。尚未训练 router，也未做独立质量评测。

## Qwen3 速度补测（2026-11-22）

同一张 V100 的 B=32 固定回放中，保留原始 384 个宽位置时，Qwen3-0.6B / 8B 相对各自全宽的配对加速为 **0.944× / 1.038×**。采用 M80 的合成宽位置时分别为 **0.999× / 1.250×**。8B M80 的 P95 提交间隔从 33.43 增至 35.77 ms。正常生成、双模型 lazy 路由吞吐、五轮回放及复现命令见 [Qwen3 V100 测速报告](../../artifacts/qwen3-v100/report.md)。两份 Qwen3 权重已缓存在远端 `/root/autodl-tmp/models/`；保持原宽位置的在线调度仍待验证。

## 真实 GRPO 路由（2026-11-23）

运行入口为 `python -m vllm_dual`。生成、测速、阈值设置与结果文件见[中文使用文档](grpo-router.md)。

`grpo_switch_policy_2600.pt` 已接入 Qwen3-0.6B / 8B 的 lazy 双模型生成。在 B=32 的 GSM8K 短生成上，3.52% 的输出选用 8B，吞吐 728.5 tok/s，同批纯 8B 为 638.8 tok/s。固定 128-token 合成输入则为 359.8 对 462.4 tok/s。两组均逐 token 在线决策，条件 FFN 阶段仍使用原有回放。配置入口、同动作回放开销与独立 HF 核对见 [GRPO 实验报告](../../artifacts/grpo-v100/report.md)。

## 连接服务器

从本机终端连接；2026-11-22 已只读验证 SSH、远端项目、Python 环境和 2B 报告可访问：

```bash
ssh -p 23667 root@region-46.seetacloud.com
```

登录后执行：

```bash
cd /root/autodl-tmp/vllm-dual-model
source /root/autodl-tmp/envs/vllm-dual-py312/bin/activate
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv
cat artifacts/mixed-ffn-2b/report.md
```

- 本地仓库：`/Users/wangshuai/vllm-dual-model`。远端日志：`/root/autodl-tmp/logs/`。连接使用已有 SSH 凭据；新环境若提示密码，在终端交互输入，不要把密码或私钥粘贴到对话里。
- 硬件和环境：单张 Tesla V100-PCIE-32GB；现有 Python 3.12 环境、PyTorch 2.9.0、FP16、`TRITON_ATTN`、TP=1。保留现有 SM70 构建，不要直接用官方 wheel 替换。
- 权重已在 `/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct` 和 `/root/autodl-tmp/models/Qwen2.5-0.5B-Instruct`。原 Qwen2.5 共享 attention 实验只加载 **1.5B-Instruct**，不需要下载，也不是最初方案写的 Qwen3-8B/H200。
- fork 基线为 `18df3cef8f61ce371d1958374927b48aba7eafe2`，实验还依赖工作树改动和远端构建。**不要只按 commit 重装或整仓覆盖远端。** 本地与远端的 `benchmarks/free_kv_replay.py` 有版本差异；2A/2B 实跑版本的哈希见 [2B 配置](../../artifacts/mixed-ffn-2b/config.json)，本地有[对应快照](../../artifacts/b32-logits-replay/phase1_driver.py)。

## 共同的回放口径

唯一基准输入是 [trace.jsonl](../../artifacts/free-kv-v1/trace.jsonl)，SHA-256 为 `393281ea1a8eee5613d4533be07d193fe1043cb3e8d8190f429e04df83ad4586`。它来自 GSM8K train 前 100 题，输出上限 48，85 题触及上限。原确定性 router 依据请求 ID 和输出位置选择动作，没有学习型 checkpoint。

模型正常前向、词表投影和贪心采样，再强制提交 trace 的 token。首个输出来自 prefill，不计入 decode；共 **4,091 个计时输出、3,991 个同请求相邻 decode 间隔**。B=32 固定编组为 32/32/32/4；立即执行时共 188 步。原轨迹有 384 个宽位置（9.39%），分布在 150 步，四个 cohort 分别有 121/119/126/18 个。若 `j` 是从 0 起的输出下标，decode 的 `j≥1`：`route_for_output[j]` 对应输入位置 `len(prompt_ids)+j−1`，预测 `output_ids[j]`。

下述固定回放只计各 cohort 同步后的 decode 墙钟总和，不含权重加载、prefill、预热和捕图。速度比先按同轮墙钟配对，再取五轮中位数；ITL 是引擎提交间隔，不是线上排队延迟。不同批次的全宽基线会波动，**不能跨批次相减来算预算**。

## 已完成的实验

| 实验 | 改了什么 | 结论与证据 |
|---|---|---|
| 双模型逐 token 路由 | 自定义 V1 调度器和 worker；0.5B 与 1.5B 各自持有 KV，lazy 路径按需补历史。 | 早期 B=16：纯小/纯大为 1,722.5/987.0 tok/s，大模型概率 0.2–0.8 时为 251.3–375.7 tok/s。此批**含 prefill 和路由日志**，不能与后续 decode-only 结果直接对比。[源码改动与各概率结果](dual-model-v100.md) |
| 免费历史 KV | 预先将原生大模型的正确历史 KV 放入最终 GPU 页，当前重位置仍正常执行。 | 免费 KV 相对纯大模型：B=1 为 1.81×，B=8 为 1.13×，B=32 只有 0.92×。B=32 纯大/原生补算/免费 KV 分别为 2,317/1,494/2,118 tok/s；免除 2,705 个历史补算位置仍不足以打平。[报告与翻译预算](../../artifacts/free-kv-v1/report.md) |
| 预置 logits 与双模型攒批 | 在免费 KV 路径仅跳过大模型当前步前向；再实测目标 4 行、12 ms 等待阈值的队列。 | 预置 logits 为 4,412 tok/s，说明重分支有优化空间。真实攒批把重调用 150→83 次：免费 KV 为 2,087→2,447 tok/s；恢复原生补算后仅 1,451→1,732 tok/s，P95 从 19.90→36.19 ms。**83 次是双模型结果，不是共享 attention 的调度结果。** [报告](../../artifacts/b32-logits-replay/report.md) |
| 阶段一：静态 FFN | 保留 28 层 attention、hidden=1536；逐层从原 8960 通道取前 4480 或 2304，模型用自身宽度 prefill、decode。 | B=32 三档墙钟分别为 1,802.2 / 1,371.4 / 1,165.4 ms。2304 比 4480 再省 198.1 ms（配对中位数），因此选 2304 做基础通道。[六格速度表与数值检查](../../artifacts/static-ffn-v1/report.md) |
| 阶段 2A：原始宽位置 | 四组均用原始全宽 prefill、一套 KV。F=原生全宽，C=只算 2304，W=所有行拆算 2304+6656，M=仅原始宽行恢复 6656。 | F/C/W/M 为 1,795.6 / 1,167.6 / 1,910.1 / 1,702.1 ms。M 为 1.055×，距 1.2× 差 205.2 ms。独立诊断中互补 FFN、取行、回填分别累计 521.0/42.3/35.1 ms，未测 DRAM 字节数。W/F 与 M/同 mask 稠密参考短回放 top-1 均为 21/21。[报告与误差](../../artifacts/mixed-ffn-v1/report.md) |

## 阶段 2B 的决策依据

只改离线宽标记，不改 kernel、请求编组、输出 token 或结束位置。H 表示至少有一行走宽分支的 decode 步数；余量为同轮 `T_F/1.2 − T_模式`，正数表示仍能增加的**整条轨迹墙钟**。

| 模式 | H / 非空平均宽行数 | decode ms | tok/s | 相对 F 加速（五轮范围） | P95 ITL ms | 1.2× 余量 ms（五轮范围） |
|---|---:|---:|---:|---:|---:|---:|
| F 原生全宽 | 188 / 21.76 | 1,810.2 | 2,260.0 | 1.000× | 10.49 | — |
| M150 原始 mask | 150 / 2.56 | 1,716.1 | 2,383.8 | 1.048×（1.046–1.061） | 10.76 | −214.7（−222.1 至 −197.8） |
| M96 合成 mask | 96 / 4.00 | 1,536.1 | 2,663.2 | 1.179×（1.166–1.201） | 10.74 | −27.5（−43.5 至 +0.7） |
| M80 合成 mask | 80 / 4.80 | 1,463.1 | 2,796.0 | **1.230×（1.210–1.239）** | 10.74 | **+35.6（+12.4 至 +47.9）** |

M150→M96 配对净省 180.0 ms，M96→M80 再省 59.7 ms。四组均为 188 次 FULL Graph、0 次 eager；三个混合组实际各执行 384 个互补行，无额外填充。M96/M80 与原始宽位置只重合 50/31 个；同 mask 稠密参考分别有 125/125、124/125 个 logits 位置 top-1 一致。**这些数证明集中执行有速度价值，不证明搬动宽位置后生成质量相同，也不包含在线 router 开销。**

## 代码与复核入口

- 原始位置：从 [trace](../../artifacts/free-kv-v1/trace.jsonl) 取动作，用 [锁定的逐步物理行与输入位置](../../artifacts/mixed-ffn-2b/locked_steps.json) 交叉核对。`original` 是该步原本走宽的请求 ID；[M80.json](../../artifacts/mixed-ffn-2b/M80.json) 是合成参照，**不是下一轮的路由输入**。
- 共享 attention：[混合 FFN 与 Graph](../../benchmarks/mixed_ffn_engine.py)、[2A 驱动与稠密参考](../../benchmarks/mixed_ffn_replay.py)。2B 的 [mask 生成](../../benchmarks/concentrated_masks.py)、[回放](../../benchmarks/concentrated_replay.py)、[汇总](../../benchmarks/summarize_concentrated.py) 独立于 kernel。
- 只看结果无需 GPU：打开 [2B 报告](../../artifacts/mixed-ffn-2b/report.md)、[20 次计时](../../artifacts/mixed-ffn-2b/timings.csv)、[五轮配对差值](../../artifacts/mixed-ffn-2b/pairs.csv)、[正确性检查](../../artifacts/mixed-ffn-2b/checks.json)。如需从原始提交时间戳重建报告，在项目根目录运行 `python benchmarks/summarize_concentrated.py`，它会重写汇总文件。
- **仅在需要重跑 2B 时**，在远端激活已有环境，先备份 `artifacts/mixed-ffn-2b/`（脚本使用固定目录，会覆盖同名产物）。使用已锁定的 mask，依次运行 `python -u benchmarks/concentrated_replay.py`、`python benchmarks/summarize_concentrated.py`。新调度实验使用新目录，不覆盖 2A/2B。

## 可直接粘贴到新对话

> 请先读本地仓库 `/Users/wangshuai/vllm-dual-model/docs/benchmarking/v100-experiment-handoff.md` 和 `artifacts/mixed-ffn-2b/report.md`。阶段 2B 已完成，不需要重跑。下一步要验证：保持原始 384 个 `(request_id, input_pos)` 不变，在共享 attention 的 2304+6656 实现中，只让已就绪的不同请求在 token 边界有限等待并组批；同一请求必须提交前一 token 才能继续。请先给出实验设计和验收条件，再决定是否启动 GPU。以同轮原生全宽 F、原始 mask 立即执行 M150 为对照，核对各请求的 token 顺序、混合历史 logits/KV、宽位置集合，再报告实际 H、总调度步数、等待分布、P95 ITL 和 1.2× 剩余墙钟。M80 只作性能参照，不挪动宽位置来凑 80 步；暂不训练，也不把固定 trace 当作质量数据。
