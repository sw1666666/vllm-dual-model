# Qwen3-0.6B 与 8B 在单卡 V100 上的速度

2026-11-22，Tesla V100-PCIE-32GB。B=32、每请求输入 128 和输出 32 个 token 时，0.6B / 8B 的原生生成吞吐为 **1,330.1 / 463.8 tok/s**，双模型 lazy 路由为 **359.7 tok/s**。在交接文档的 decode-only 回放上，8B 条件 FFN 保持原宽位置时快 3.8%，集中到 80 步时快 25.0%；0.6B 在 80 步时与全宽持平。

## 正常生成吞吐

固定 128 个合成输入 token，贪心生成 32 个输出 token。每个模型配置在每个 batch 大小先预热一次，再计时三批；表中是三次输出 tok/s 的算术均值。计时覆盖生成调用、prefill、decode 和 GPU 同步。启用 CUDA Graph，关闭 prefix caching、chunked prefill 和异步调度。双模型同时驻留，lazy 路由使用 seed=17、目标大模型概率 0.1、攒批阈值 4 行、等待上限 4 步，计时包含路由日志。末列只统计三批正式请求。

| batch | 0.6B tok/s | 8B tok/s | 双模型 lazy tok/s | 实际选用 8B 的输出 |
|---:|---:|---:|---:|---:|
| 1 | 183.2 | 39.6 | 101.8 | 10/96 (10.42%) |
| 8 | 835.2 | 212.9 | 278.8 | 67/768 (8.72%) |
| 16 | 1,164.7 | 336.6 | 337.1 | 140/1536 (9.11%) |
| 32 | 1,330.1 | 463.8 | 359.7 | 325/3072 (10.58%) |

双模型相对纯 8B 的吞吐随 batch 改变。B=1 快 2.57×，B=16 基本持平，B=32 慢 22.4%。B=32 时进程内 PyTorch 峰值 reserved 显存为 4.01 / 18.12 / 19.16 GiB（0.6B / 8B / 双模型）。旧脚本仅预热最大 batch，0.6B 单请求首轮曾降至 27.0 tok/s；上表使用逐 batch 预热后的结果。[生成计时与三轮范围](generation.csv) · [0.6B 原始计时](generation/small.json) · [8B 原始计时](generation/large.json) · [双模型原始计时](generation/dual-lazy.json)

## B=32 共享 attention、条件 FFN 的 decode 回放

沿用 2B 的 GSM8K 固定轨迹、32/32/32/4 分组、FP16、TP=1、TRITON_ATTN。每轮同步计时 188 个 decode 步、4,091 个输出 token，排除权重加载、原生全宽 prefill、捕图和预热。每步经过真实模型前向、词表投影和贪心采样，随后强制提交已有轨迹中的 token。以下是五轮中位数；同模型、同轮的 F 墙钟作为加速基准。P95 ITL 取自每轮 3,991 个同请求相邻 decode 提交间隔。余量 = 同轮 F 墙钟 / 1.2 − 被测墙钟。

| 模型 | 模式 | decode ms | 输出 tok/s | 配对加速（五轮范围） | P95 ITL ms | 1.2× 余量 ms |
|---|---|---:|---:|---:|---:|---:|
| 0.6B | F | 1,431.4 | 2,858.1 | 1.000× (1.000–1.000) | 9.32 | -238.6 |
| 0.6B | M150 | 1,518.6 | 2,693.9 | 0.944× (0.943–0.950) | 10.04 | -323.2 |
| 0.6B | M96 | 1,452.7 | 2,816.1 | 0.986× (0.983–0.991) | 10.00 | -261.5 |
| 0.6B | M80 | 1,433.4 | 2,854.1 | 0.999× (0.993–1.006) | 9.98 | -240.5 |
| 8B | F | 5,906.1 | 692.7 | 1.000× (1.000–1.000) | 33.43 | -984.3 |
| 8B | M150 | 5,689.0 | 719.1 | 1.038× (1.037–1.040) | 35.75 | -767.9 |
| 8B | M96 | 4,956.8 | 825.3 | 1.192× (1.189–1.194) | 35.72 | -35.1 |
| 8B | M80 | 4,724.3 | 866.0 | 1.250× (1.249–1.252) | 35.77 | +197.9 |

F 使用原生全宽。各 M 模式都在 384 个位置恢复全宽：M150 使用原始宽位置，M96 和 M80 分别将宽位置集中到 96 和 80 步，与原始集合分别重合 50 和 31 个位置。窄分支取前四分之一 FFN 通道：0.6B 为 768 + 2304，8B 为 3072 + 9216，两者都保留原生 attention、全宽 prefill 和单套 KV。生成质量和保留原位置的在线调度收益仍待验证。

每个模型的三个掩码都抽查了首组 125 个 logits 位置。稀疏分支与同掩码稠密参考的 top-1 均为 125/125，prefill KV 逐元素相同。各模式的独立核对均记录了 188 次 FULL CUDA Graph 分派，三个 M 模式均执行 384 个补充行。各模式的 P95 ITL 均列于上表，8B M80 的整段轨迹墙钟下降，但 P95 较全宽高约 2.34 ms。[0.6B 正确性](ffn-0.6B/checks.json) · [8B 正确性](ffn-8B/checks.json) · [逐轮计时](timings.csv) · [模型配置与权重分片记录](models.json)

## 复现

在交接文档所列的 V100 环境中，从仓库根目录运行。两份权重位于 `/root/autodl-tmp/models/Qwen3-{0.6B,8B}`，来自 ModelScope 的 `Qwen/Qwen3-0.6B` 和 `Qwen/Qwen3-8B`。环境为 vLLM 0.11.0.dev0+cu121（远端 SM70 构建）、PyTorch 2.9.0+cu128、Triton 3.5.0、Transformers 4.57.6。以下命令使用新目录保存结果。

```bash
set -eu
source /root/autodl-tmp/envs/vllm-dual-py312/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
models=/root/autodl-tmp/models
result=$(mktemp -d "$PWD/artifacts/qwen3-repeat-XXXXXX")
mkdir "$result/generation"
cp artifacts/server-v100/benchmark_modes.py "$result/generation/"
patch -d "$result/generation" -p1 < artifacts/qwen3-v100/generation.patch
for mode in small large dual-lazy; do
  python "$result/generation/benchmark_modes.py" --mode "$mode" \
    --small-model "$models/Qwen3-0.6B" --large-model "$models/Qwen3-8B" \
    --p-large 0.1 --batch-sizes 1 8 16 32 \
    --trace-path "$result/generation/$mode.trace.jsonl" --output-json "$result/generation/$mode.json"
done
mkdir "$result/replay"
cp benchmarks/{mixed_ffn_engine,mixed_ffn_replay,concentrated_masks,concentrated_replay}.py "$result/replay/"
cp artifacts/b32-logits-replay/phase1_driver.py "$result/replay/free_kv_replay.py"
patch -d "$result/replay" -p1 < artifacts/qwen3-v100/qwen3-ffn.patch
for size in 0.6B 8B; do
  python "$result/replay/concentrated_replay.py" --model "$models/Qwen3-$size" \
    --out "$result/ffn-$size"
done
```

在本地仓库执行 `python3 artifacts/qwen3-v100/summarize.py` 即可从已保存的四个 cohort 墙钟和提交时间戳重建 40 行计时表。脚本逐轮核对每个请求的输出位置、顺序和间隔数；GPU 回放驱动另行核对锁定的物理行、宽位置和 CUDA Graph 分派。输入校验和沿用 [2B 记录](../mixed-ffn-2b/inputs.json)。此次适配以独立副本运行。[维度适配补丁](qwen3-ffn.patch) · [原始生成测速脚本](../server-v100/benchmark_modes.py) · [2B 回放源码](../../benchmarks/concentrated_replay.py)
