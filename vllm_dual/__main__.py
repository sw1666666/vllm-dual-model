"""Generate text or benchmark the Qwen3 GRPO router on the V100."""
import argparse
import os
import sys
import tempfile
from pathlib import Path

from .config import add_grpo_threshold_args


REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(prog="python -m vllm_dual", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    add_grpo_threshold_args(common)
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
    commands.add_parser("benchmark", parents=[common], help="Measure online GRPO throughput")
    args = parser.parse_args()
    checkpoint = args.checkpoint or args.models / "grpo_switch_policy_2600.pt"
    threshold = str(args.router_large_threshold)
    args.output_root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=f"{args.command}-{threshold}-", dir=args.output_root)).resolve()

    if args.command == "generate":
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
    flags.extend(["--router-large-threshold", threshold])
    print(f"Run directory: {out}", flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO), env.get("PYTHONPATH")]))
    os.execve(sys.executable, [sys.executable, str(script), *map(str, flags)], env)


if __name__ == "__main__":
    main()
