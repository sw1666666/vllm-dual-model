#!/usr/bin/env bash
set -eu
out=artifacts/b32-logits-replay
while [[ ! -s "$out/diag-routed_replay_logits-0.jsonl" ]]; do
    if ! pgrep -f '^bash benchmarks/run_b32_logits.sh' >/dev/null; then
        tail -15 /root/autodl-tmp/logs/b32-logits.log
        exit 1
    fi
    sleep 20
done
python benchmarks/summarize_b32_logits.py
if python -c 'import json,sys;sys.exit(not json.load(open("artifacts/b32-logits-replay/decision.json"))["curve"])'; then
    python benchmarks/b32_heavy_curve.py
    python benchmarks/summarize_b32_logits.py
    if python -c 'import json,sys;sys.exit(not json.load(open("artifacts/b32-logits-replay/decision.json"))["queue"])'; then
        queue_driver="$out/queue_driver/b32_ready_queue.py"
        if [[ ! -f "$queue_driver" ]]; then queue_driver=benchmarks/b32_ready_queue.py; fi
        PYTHONPATH="$PWD" python "$queue_driver"
        python benchmarks/summarize_b32_logits.py
    fi
fi
