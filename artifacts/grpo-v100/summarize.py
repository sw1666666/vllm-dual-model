"""Summarize throughput and paired wall times for the V100 switch-policy run."""
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    modes = ("small", "large", "hash", "grpo", "replay")
    runs = {mode: json.loads((ROOT / f"{mode}.json").read_text()) for mode in modes}
    rows = []
    for index, active in enumerate(runs["grpo"]):
        batch = active["batch"]
        ref = runs["replay"][index]
        assert active["requests"] == ref["requests"]
        row = {key: active[key] for key in ("workload", "batch", "repeat")}
        for mode in modes:
            run = runs[mode][index]
            assert [r["prompt"] for r in active["requests"]] == [r["prompt"] for r in run["requests"]]
            assert (run["workload"], run["batch"], run["repeat"]) == tuple(row[k] for k in ("workload", "batch", "repeat"))
            assert all(len(r["tokens"]) == 32 for r in run["requests"])
            row[mode + "_tok_s"] = batch * 32 / run["seconds"]
        row.update(large_outputs=active["large_outputs"], output_tokens=batch*32,
                   replay_delta_ms=(active["seconds"] - ref["seconds"]) * 1000,
                   replay_extra_percent=(active["seconds"] / ref["seconds"] - 1) * 100)
        rows.append(row)
    with (ROOT / "timings.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for start in range(0, len(rows), 3):
        chunk = rows[start:start + 3]
        print(chunk[0]["workload"], chunk[0]["batch"],
              *[round(statistics.mean(r[m + "_tok_s"] for r in chunk), 2) for m in modes],
              "large", sum(r["large_outputs"] for r in chunk),
              "extra %", [round(r["replay_extra_percent"], 2) for r in chunk])


if __name__ == "__main__":
    main()
