# 单卡 Qwen3 双模型 GRPO 路由

在实验服务器的仓库根目录运行 `python -m vllm_dual`。用 `generate` 生成文本，用 `benchmark` 测端到端吞吐。两个子命令都会加载 Qwen3-0.6B、Qwen3-8B 和训练好的 GRPO 策略，按 token 选择模型。

两个子命令都接受 `--variant`。版本控制选中 8B 后何时执行，`--router-large-threshold` 单独控制路由阈值。

| 参数 | 选中 8B 后的行为 |
|---|---|
| `--variant v1`（默认） | 在本轮立即执行 8B，然后提交 token。省略版本参数也使用 V1。 |
| `--variant v2` | 暂停该请求，其他请求可继续用 0.6B 解码。当待算位置达到 2 × 并发上限、最早的请求等到下一轮 0.6B 结束、或没有 0.6B 工作时，按 token 预算分批执行 8B。 |

## 生成文本

```bash
cd /root/autodl-tmp/vllm-dual-model
source /root/autodl-tmp/envs/vllm-dual-py312/bin/activate
python -m vllm_dual generate --prompt "What is the Moon?" --max-tokens 32
```

默认阈值为 **0.5**。下面这条命令把阈值提高到 0.65，策略给 8B 的概率必须超过 65% 才会选用它。

```bash
python -m vllm_dual generate --prompt "计算 17 × 24。" --router-large-threshold 0.65
```

单条输入按原始提示文本生成。批量输入用 `--prompts-json prompts.json` 替换 `--prompt`；文件格式为字符串数组，例如 `["计算 17 × 24。", "用一句话解释光合作用。"]`。默认每条最多生成 32 个 token，当前实验的模型上下文长度为 256 个 token，包含输入与输出。

## 设置 8B 阈值

`--router-large-threshold` 接受 **0.1 到 0.9 之间的任意数值，包含两端**，例如 `0.1`、`0.375` 或 `0.9`。每个输出位置都计算一次策略概率：当 `P(8B) > 阈值` 时选择 8B，其余情况（含相等）选择 0.6B。

| 参数 | 对当前 token 的效果 |
|---|---|
| `--router-large-threshold 0.4` | 超过 40% 即选 8B |
| 省略参数 | 超过 50% 才选 8B |
| `--router-large-threshold 0.6` | 超过 60% 才选 8B |

兼容写法 `--router-thresholds 0.6/0.4` 的顺序是 **小模型/大模型**，等价于 `--router-large-threshold 0.4`。两种写法任选其一。整批实际使用 8B 的比例取决于提示、生成的 token 和此前的路由选择。

## 在 96 GB 卡上复现大并发实验

在同一台机器上依次运行两种版本。下面的阈值来自约 15% 8B 路由实验，省略阈值时使用 0.5。

```bash
export OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1
for variant in v1 v2; do
  for batch in 128 256; do
    python -m vllm_dual benchmark --variant "$variant" --batch "$batch" --router-large-threshold 0.6475
  done
  python -m vllm_dual benchmark --variant "$variant" --batch 96 --workload gsm8k --router-large-threshold 0.6475
done
```

`--batch B` 设置活跃并发上限。默认合成负载一次提交 4B 个固定请求，每条输入 128 个 token。GSM8K 一次提交原有的 96 道题。每条固定输出 32 个 token，每组预热一次、正式计时三次，关闭详细 trace。`result.json` 中的 `median_tok_s` 是包含 prefill 和 decode 的端到端输出吞吐中位数。双模型每轮的实际 8B 路由率为 `runs[i].large_outputs / (requests × 32)`。

计时外核对路由时，用 `--batch 8 --workload check` 分别运行 V1 和 V2，比较两个运行目录的 `actions-tokens.json`。每条请求记录 32 对路由动作和 token ID。

阈值 0.6475 下，V2 在 B=128、256 合成输入上的实际 8B 占比分别为 15.65%、15.06%，在 GSM8K 上为 3.84%。大批合成输入上有少量请求随 FP16 组批形状改变生成轨迹，逐项核对见报告。[约 15% 路由率的结果和对齐记录](../../artifacts/grpo-blackwell-15pct/report.md) · [原阈值 0.5 的结果和配置](../../artifacts/grpo-blackwell-batch/report.md)

## 复现 V100 小并发基准

```bash
python -m vllm_dual benchmark --router-large-threshold 0.65
```

每档先预热，再计时三批。合成输入为每请求 128 个 token，并发分别为 1、8、16、32；GSM8K 取前 96 道训练题，分三批，并发 32。每个请求固定生成 32 个 token。终端逐批显示输出 tok/s，计时包含 prefill、decode、在线路由及路由日志。

比较阈值时，分别用不同的 `--router-large-threshold` 运行相同命令。在 `grpo.json` 中按相同 `workload` 和 `batch` 对齐三批，取 `tok_s` 的算术平均。8B 使用率为三批的 `large_outputs` 总和除以三批的实际输出 token 总数。

## 模型路径和结果文件

默认模型目录为 `/root/autodl-tmp/models`，其中包含 `Qwen3-0.6B/`、`Qwen3-8B/` 和 `grpo_switch_policy_2600.pt`。两个子命令均可用 `--models DIR` 指定模型父目录，用 `--checkpoint FILE` 单独指定策略文件。

启动时终端会打印 `Run directory`。每次运行自动在 `artifacts/grpo-runs/` 下新建独立子目录；用 `--output-root DIR` 可以改变结果的父目录。

| 子命令 | 文件 | 内容 |
|---|---|---|
| `generate` | `generation.json` | 生成文本、输出 token ID、策略文件和实际阈值 |
| `generate` | `trace.jsonl` | 每个输出位置选用的模型及提交的 token |
| `benchmark` 省略 `--batch` | `grpo.json`、`grpo.trace.jsonl` | 每批阈值、耗时、吞吐、8B 输出数及逐请求动作，包含详细 trace |
| `benchmark --batch B` | `result.json` | 测试配置、各轮吞吐、中位数、8B 工作量及实际并发 |
| `benchmark --batch B --workload check` | `actions-tokens.json`、`check.jsonl`、`result.json` | 逐请求动作和 token、详细 trace 及本次配置与计数 |

用 `python -m vllm_dual generate --help` 或 `python -m vllm_dual benchmark --help` 查看参数。已有 V100 吞吐、固定概率路由对照与阈值实测见[实验报告](../../artifacts/grpo-v100/report.md)。
