"""Verify phase 2B replay results against the locked trace and compute paired metrics."""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODES = ("F", "M150", "M96", "M80")


def load(path):
    return json.loads(path.read_text())


def p95(values):
    ordered = sorted(values)
    pos = 0.95 * (len(ordered) - 1)
    low = int(pos)
    return ordered[low] + (pos - low) * (ordered[low + 1] - ordered[low])


def summarize(directory, locked, trace):
    expected = defaultdict(list)
    for step in locked:
        rows = zip(step["ids"], step["pos"])
        expected[step["cohort"]].extend((rid, step["step"], pos) for rid, pos in rows)

    audited = load(directory / "audit.json")
    timings = []
    for mode in MODES:
        steps = [row for row in audited if row["mode"] == mode]
        assert len(steps) == len(locked) == 188
        for step, ref in zip(steps, locked):
            assert step["graph"] == "FULL"
            assert (step["cohort"], step["step"], step["active"]) == (
                ref["cohort"], ref["step"], len(ref["ids"])
            )
        if mode != "F":
            assert sum(step["executed_wide"] for step in steps) == 384
            assert sum(step["k"] > 0 for step in steps) == int(mode[1:])

    for repeat in range(5):
        baseline_ms = None
        for mode in MODES:
            cohorts = load(directory / f"time-{repeat}-{mode}.json")
            assert len(cohorts) == len(expected) == 4
            wall_ms = sum(row["wall_s"] for row in cohorts) * 1000
            intervals = []
            count = 0
            for cohort, row in enumerate(cohorts):
                commits = row["commits"]
                assert {(r, t, p) for r, t, p, _ in commits} == set(expected[cohort])
                assert len(commits) == len(expected[cohort])
                count += len(commits)
                last = {}
                for rid, step, pos, stamp in commits:
                    previous, last_stamp = last.get(rid, (0, 0.0))
                    assert step == previous + 1 and pos == len(trace[rid]["prompt_ids"]) + step - 1
                    if previous:
                        intervals.append(stamp - last_stamp)
                    last[rid] = (step, stamp)
            assert count == 4091 and len(intervals) == 3991 and min(intervals) >= 0
            if mode == "F":
                baseline_ms = wall_ms
            timings.append(dict(model=directory.name.split("ffn-", 1)[-1], repeat=repeat, mode=mode,
                                decode_tokens=count, wall_ms=wall_ms, tok_s=count*1000/wall_ms,
                                speedup=baseline_ms/wall_ms, budget_ms=baseline_ms/1.2-wall_ms,
                                p95_itl_ms=p95(intervals)))
    return timings


def generation():
    directory = ROOT / "generation"
    by_request = defaultdict(list)
    with (directory / "dual-lazy.trace.jsonl").open() as stream:
        for event in map(json.loads, stream):
            by_request[event["request_id"]].append(event)
    summary = []
    for mode in ("small", "large", "dual-lazy"):
        measured_ids = set()
        groups = load(directory / f"{mode}.json")["results"]
        assert [group["batch_size"] for group in groups] == [1, 8, 16, 32]
        for group in groups:
            batch = group["batch_size"]
            rates = []
            large_count = 0
            for repeat in group["repeats"]:
                ids = repeat["request_ids"]
                assert len(set(ids)) == batch and measured_ids.isdisjoint(ids)
                measured_ids.update(ids)
                assert repeat["output_tokens"] == 32 * batch
                rates.append(32 * batch / repeat["latency_seconds"])
                if mode == "dual-lazy":
                    for rid in ids:
                        events = by_request[rid]
                        assert [event["output_index"] for event in events] == list(range(32))
                        large_count += sum(event["route"] == "large" for event in events)
            assert len(rates) == 3
            mean = statistics.mean(rates)
            assert abs(mean - group["output_tokens_per_second_mean"]) < 1e-6
            summary.append(dict(mode=mode, batch=batch, tokens=3*32*batch, mean_tok_s=mean,
                                min_tok_s=min(rates), max_tok_s=max(rates),
                                large_routes=large_count if mode == "dual-lazy" else "",
                                peak_reserved_gib=group["peak_gpu_reserved_gib_max"]))
    return summary


if __name__ == "__main__":
    artifacts = ROOT.parent
    locked = load(artifacts / "mixed-ffn-2b/locked_steps.json")
    trace = {r["request_id"]: r for r in map(json.loads, (artifacts / "free-kv-v1/trace.jsonl").read_text().splitlines())}
    timings = []
    for size in ("0.6B", "8B"):
        timings.extend(summarize(ROOT / f"ffn-{size}", locked, trace))
    with (ROOT / "timings.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, timings[0].keys())
        writer.writeheader()
        writer.writerows(timings)
    for size in ("0.6B", "8B"):
        for mode in MODES:
            runs = [r for r in timings if r["model"] == size and r["mode"] == mode]
            names = ("wall_ms", "tok_s", "speedup", "p95_itl_ms", "budget_ms")
            print(size, mode, *(round(statistics.median(r[k] for r in runs), 4) for k in names),
                  "range", *(round(f(r["speedup"] for r in runs), 4) for f in (min, max)))
    generated = generation()
    with (ROOT / "generation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, generated[0].keys())
        writer.writeheader()
        writer.writerows(generated)
