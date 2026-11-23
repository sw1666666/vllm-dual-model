#!/usr/bin/env bash
# Run on the single V100 from the repository root, in the vLLM environment.
set -eu
out=${1:?pass the directory containing trace.jsonl and validated checks}
modes=(large small routed_exact routed_free_kv)
for repeat in 0 1 2 3 4; do
    for offset in 0 1 2 3; do
        mode=${modes[$(((repeat + offset) % 4))]}
        if [[ ! -s "$out/time-$mode-b32-r$repeat.jsonl" ]]; then
            python benchmarks/free_kv_replay.py time --out "$out" --mode "$mode" --repeat "$repeat"
        fi
    done
done
for mode in "${modes[@]}"; do
    python benchmarks/free_kv_replay.py diag --out "$out" --mode "$mode"
done
