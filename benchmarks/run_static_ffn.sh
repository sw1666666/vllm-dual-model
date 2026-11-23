#!/usr/bin/env bash
set -eu
out=artifacts/static-ffn-v1
test -s "$out/models.json"
# Keep the runtime model/cache bindings used to audit compiled graphs.
run_replay() {
    set -o pipefail
    python benchmarks/static_ffn_replay.py "$@" 2>&1 | tee -a "$out/engine.log"
}
widths=(8960 4480 2304)
for width in "${widths[@]}"; do
    if [[ ! -s "$out/diag-$width.json" ]]; then run_replay diag "$width"; fi
done
for r in 0 1 2 3 4; do
    for o in 0 1 2; do
        w=${widths[$(((r+o)%3))]}
        if [[ ! -s "$out/time-$r-$w-1.json" || ! -s "$out/time-$r-$w-32.json" ]]; then
            run_replay time "$w" --repeat "$r"
        fi
    done
done
python benchmarks/summarize_static_ffn.py
