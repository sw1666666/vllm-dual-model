"""Generate text or benchmark the single-GPU Qwen3 GRPO router."""
import argparse
import os
import sys
from pathlib import Path
from uuid import uuid4

from .config import add_grpo_threshold_args


REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(prog="python -m vllm_dual", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    add_grpo_threshold_args(common)
    common.add_argument("--variant", choices=("v1", "v2"), default="v1",
                        help="v1 runs 8B immediately (default); v2 batches pending 8B work")
    common.add_argument("--models", type=Path, default=Path("/root/autodl-tmp/models"),
                        help="Parent of Qwen3-0.6B and Qwen3-8B (default: /root/autodl-tmp/models)")
    common.add_argument("--checkpoint", type=Path,
                        help="Policy checkpoint (default: grpo_switch_policy_2600.pt under --models)")
    common.add_argument("--output-root", type=Path, default=REPO / "artifacts/grpo-runs",
                        help="Parent directory for a fresh result directory per run")
    generate = commands.add_parser("generate", parents=[common], help="Generate with the online router")
    inputs = generate.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--prompt", help="One text prompt")
    inputs.add_argument("--prompts-json", type=Path, help="JSON file containing a list of text prompts")
    generate.add_argument("--max-tokens", type=int, default=32, help="Output token limit per prompt")
    benchmark = commands.add_parser("benchmark", parents=[common], help="Measure online GRPO throughput")
    benchmark.add_argument("--batch", type=int, help="Active concurrency for a fixed-request run")
    benchmark.add_argument("--workload", choices=("synthetic", "gsm8k", "check"),
                           help="Fixed-request workload (default with --batch: synthetic)")
    args = parser.parse_args()
    if args.command == "benchmark":
        if args.workload and args.batch is None:
            parser.error("--workload requires --batch")
        if args.batch is not None and args.batch < 1:
            parser.error("--batch must be positive")
        if args.workload == "gsm8k" and args.batch != 96:
            parser.error("--workload gsm8k requires --batch 96")
    checkpoint = args.checkpoint or args.models / "grpo_switch_policy_2600.pt"
    threshold = str(args.router_large_threshold)
    out = (args.output_root / f"{args.command}-{args.variant}-{threshold}-{uuid4().hex}").resolve()

    if args.command == "generate":
        out.mkdir(parents=True)
        script = REPO / "examples/offline_inference/dual_model_v100.py"
        flags = ["--small-model", args.models / "Qwen3-0.6B",
                 "--large-model", args.models / "Qwen3-8B", "--mode", "lazy",
                 "--enable-dual-forward", "--router-checkpoint", checkpoint,
                 "--max-tokens", str(args.max_tokens), "--trace-path", out / "trace.jsonl",
                 "--output-json", out / "generation.json"]
        if args.prompt is not None:
            flags.extend(["--prompt", args.prompt])
        else:
            flags.extend(["--prompts-json", args.prompts_json])
    else:
        script = REPO / "benchmarks/benchmark_grpo_router.py"
        flags = ["grpo", "--models", args.models, "--checkpoint", checkpoint, "--out", out,
                 "--questions", REPO / "artifacts/grpo-v100/questions.json"]
        if args.batch is not None:
            flags.extend(["--batch", str(args.batch), "--workload", args.workload or "synthetic"])
    if args.variant == "v2":
        flags.append("--queue-large")
    flags.extend(["--router-large-threshold", threshold])
    print(f"Run directory: {out}", flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO), env.get("PYTHONPATH")]))
    os.execve(sys.executable, [sys.executable, str(script), *map(str, flags)], env)


if __name__ == "__main__":
    main()
