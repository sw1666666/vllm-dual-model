# GRPO 逐 token 路由在 Qwen3-0.6B / 8B 上的速度

2026-11-22 至 23 日，单张 Tesla V100-PCIE-32GB，FP16、TP=1、TRITON_ATTN、CUDA Graph。使用用户提供的 `grpo_switch_policy_2600.pt`，SHA-256 为 `98707ce5a741bb9692c62a19f76a434b4d72b59e1c2453694e9c38bed5ae04de`。

**B=32 时，GSM8K 的在线 GRPO 为 728.5 tok/s，纯 8B 为 638.8 tok/s（+14.0%）。固定 128-token 输入时，在线 GRPO 为 359.8 tok/s，纯 8B 为 462.4 tok/s（−22.2%）。** 两种负载分别有 3.52% 和 17.48% 的输出选用 8B。GSM8K 的三个批次各有 32 道不同题，GRPO 分别为 607.9、773.5、804.1 tok/s，对应纯 8B 为 621.3、627.9、667.3 tok/s。

| 输入 | B | 纯 0.6B | 纯 8B | 固定概率 | 在线 GRPO | 同动作回放 | 8B 输出占比 | 在线相对回放额外墙钟 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 固定 128 token | 1 | 177.1 | 39.5 | 103.2 | 110.5 | 119.2 | 6/96 (6.25%) | +8.3% |
| 固定 128 token | 8 | 813.6 | 212.1 | 236.5 | 222.5 | 232.6 | 144/768 (18.75%) | +4.5% |
| 固定 128 token | 16 | 1,119.6 | 336.3 | 298.6 | 261.2 | 275.4 | 471/1536 (30.66%) | +5.4% |
| 固定 128 token | 32 | 1,273.5 | 462.4 | 352.5 | 359.8 | 375.3 | 537/3072 (17.48%) | +4.3% |
| GSM8K | 32 | 1,952.3 | 638.8 | 461.4 | 728.5 | 751.3 | 108/3072 (3.52%) | +3.3% |

吞吐单位为输出 tok/s。最后一列逐批计算 `在线耗时 / 回放耗时 − 1` 再取均值，包含策略特征组装、头部计算、动作读回和历史更新，也包含两次独立运行的墙钟波动。固定输入和 GSM8K 在 B=32 的三个配对范围分别为 +3.9% 至 +4.6% 和 +2.1% 至 +4.9%。[逐次计时](timings.csv) · [真实动作与输出](grpo.json) · [同动作回放](replay.json)

## 路由与计时

0.6B 每步对已提交前缀正常前向。策略将当前采样位置的 1024 维最终隐状态与 1024 维使用历史拼接，由 FP32 二分类头取 argmax。历史是此前实际选择 8B 的各输出位置对应 embedding 之和；位置表有 128 行。动作 0 使用小模型 logits，动作 1 由 8B 补算缺失的历史并提供当前 logits。第一个输出也经过策略。启动时在 CPU 核对 checkpoint 主干与 0.6B 的 FP16 权重，两者相同，运行时共享这次小模型前向。[路由实现](../../vllm_dual/grpo_router.py) · [worker](../../vllm_dual/worker.py)

两模型各自维护 KV。调度器为已调度请求预留大模型页，当前步的小模型前向结束后，worker 立即执行命中 8B 的行。此路径不跨步等待攒批；本次固定概率对照同样立即执行（p=0.1，seed=17）。回放对照保持相同分支、KV 补算、采样和日志，按请求和输出下标读取预先记录的动作。检查包含全部正式请求的每个 prompt、动作与生成 token。

固定输入沿用前次测速的 128 个合成 token，分别测 B=1/8/16/32。GSM8K 使用 train 前 96 道真实题，按原顺序组成三批 B=32，用 Qwen3 的聊天模板关闭 thinking，输入为 38–127 个 token（均值 73.4）。两种负载均正常贪心生成 32 个 token，忽略 EOS，每档先预热，随后计时三次。吞吐为三批各自输出 tok/s 的算术均值，含 prefill、decode 和同步，双模型含逐 token 路由日志；模型加载和编译在计时之外。GSM8K 仅用于测生成速度，未给答案评分。

## 调整选用 8B 的阈值

生成入口 `examples/offline_inference/dual_model_v100.py` 和测速入口 `benchmarks/benchmark_grpo_router.py` 都支持以下写法。概率来自当前 token 的二分类策略头；表中原有测速使用默认阈值。

| 参数 | 当前 token 选用 8B 的条件 |
|---|---|
| 省略，或 `--router-thresholds 0.5/0.5` | P(8B) > 0.5，平票选 0.6B |
| `--router-thresholds 0.6/0.4` 或 `--router-large-threshold 0.4` | P(8B) > 0.4 |
| `--router-thresholds 0.4/0.6` 或 `--router-large-threshold 0.6` | P(8B) > 0.6 |

双数写法的顺序为 **小模型/大模型**，两数之和须为 1。实际选用 8B 的 token 占比取决于输入和之前的路由决策。Python 配置使用 `DualModelConfig(router_large_threshold=...)`。分别测试多个阈值时，为每个阈值指定独立的 `--out` 目录，同动作回放使用对应目录和阈值。

在同一张 V100 上用 `0.6/0.4` 重跑。B=32 固定输入中，8B 从 537/3072（17.48%）增至 598/3072（19.47%），吞吐为 344.2 tok/s。GSM8K 中，8B 从 108/3072 变为 100/3072（3.26%），吞吐为 754.2 tok/s。96 个请求中有 4 个请求的轨迹发生变化，四者的首个分歧均由小模型改为大模型；后续决策受到新轨迹影响。独立参考核对了 224 个位置，动作和输出 token 均一致。[0.6/0.4 原始计时](threshold-040/grpo.json) · [独立参考](threshold-040/reference.json)

## 正确性与复现

五种负载配置共 8,544 个正式输出位置（含 GSM8K）。真实路由与回放的全部输入、动作和输出完全相同。另取七个请求，在已提交前缀上独立运行 Transformers 的 Qwen3 模型，并用位置掩码直接重算 checkpoint 策略。路由决定 224/224 相同，小模型出词 216/216、大模型出词 8/8 相同。每个正式请求的 trace 另行校验提交顺序、选中 logits 的 argmax 和大模型实际补算的历史 token。21 项单元测试通过；另有 17 项旧验收测试读取 M4–M7 的历史 JSON 文件时因文件缺失失败。[独立参考核对](reference.json) · [核对脚本](check_reference.py)

在远端项目根目录运行以下命令。单条生成可以通过 `examples/offline_inference/dual_model_v100.py` 使用 `--mode lazy --enable-dual-forward --router-checkpoint "$ckpt"`，并提供两个模型目录及 trace 路径。

```bash
source /root/autodl-tmp/envs/vllm-dual-py312/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
models=/root/autodl-tmp/models
ckpt="$models/grpo_switch_policy_2600.pt"
out=$(mktemp -d "$PWD/artifacts/grpo-repeat-XXXXXX")
for mode in small large hash grpo replay; do
  python benchmarks/benchmark_grpo_router.py "$mode" --models "$models" \
    --checkpoint "$ckpt" --out "$out"
done
python artifacts/grpo-v100/check_reference.py --models "$models" \
  --checkpoint "$ckpt" --out "$out"
```
