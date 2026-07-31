#!/usr/bin/env python3
"""Run the M3 resident-weight probe through the unified entry point."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--small-model", required=True)
    parser.add_argument("--large-model", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--trace-path", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(repo_root / "examples/offline_inference/dual_model_v100.py"),
        "--small-model",
        args.small_model,
        "--large-model",
        args.large_model,
        "--mode",
        "sync",
        "--p-large",
        "0",
        "--kv-cache-memory-bytes",
        str(128 << 20),
        "--trace-path",
        str(args.trace_path),
        "--output-json",
        str(args.output_json),
    ]
    raise SystemExit(subprocess.call(command, cwd=repo_root))


if __name__ == "__main__":
    main()
