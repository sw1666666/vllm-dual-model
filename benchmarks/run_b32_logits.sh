#!/usr/bin/env bash
set -eu
out=artifacts/b32-logits-replay
mkdir -p "$out"
run() {
    echo "$(date -Is) $*" >&2
    python benchmarks/b32_logits_replay.py "$@"
}
run capture --mode routed_free_kv
run verify --mode routed_free_kv
run verify --mode routed_replay_logits
modes=(large routed_free_kv routed_replay_logits)
for r in 0 1 2 3 4; do
    for offset in 0 1 2; do
        run time --repeat "$r" --mode "${modes[$(((r + offset) % 3))]}"
    done
done
for mode in "${modes[@]}"; do run diag --mode "$mode"; done
