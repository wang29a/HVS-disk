#!/usr/bin/env python3
import argparse
import csv
import json
import os
import statistics

KINDS = ("baseline", "full", "merge", "no-alpha")
LS = (20, 40, 60, 80, 100, 120, 160, 200, 300, 400)
METRICS = ("qps", "latency_us", "p999_us", "hops", "comparisons", "topology_ios",
           "rerank_ios", "cache_hits", "recall")


def read_log(path, expected):
    text = open(path).read()
    rows = []
    for line in text.splitlines():
        f = line.split()
        if len(f) >= 17 and f[0].isdigit() and int(f[0]) in expected:
            row = dict(L=int(f[0]))
            for key, index in zip(METRICS, (2, 3, 4, 5, 6, 7, 8, 9, -1)):
                row[key] = float(f[index])
            rows.append(row)
    if [r["L"] for r in rows] != list(expected) or "Exit status: 0" not in text:
        raise ValueError("incomplete or failed search log: " + path)
    if any(not 0 <= r["recall"] <= 100 or r["qps"] <= 0 or r["comparisons"] <= 0 for r in rows):
        raise ValueError("invalid search metrics: " + path)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--check-smoke", action="store_true")
    a = p.parse_args()
    if a.check_smoke:
        rows = {k: read_log(a.root + "/logs/smoke-" + k + ".log", (40, 100)) for k in KINDS}
        for k, curve in rows.items():
            if curve[0]["recall"] < 30 or curve[1]["recall"] < curve[0]["recall"]:
                raise ValueError("unexpected recall in " + k)
        for i in range(2):
            if abs(rows["full"][i]["recall"] - rows["merge"][i]["recall"]) > 2:
                raise ValueError("Full/Merge smoke recall differs by >2 percentage points")
        with open(a.root + "/analysis/smoke-summary.json", "w") as f:
            json.dump(rows, f, indent=2, sort_keys=True)
        print("four-scheme smoke PASS")
        return
    curves = {}
    fields = ("threads", "scheme", "L", "repeats") + METRICS + tuple(k + "_stddev" for k in METRICS)
    with open(a.root + "/analysis/search-curve-summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for t in (32, 64):
            for kind in KINDS:
                reps = [read_log(a.root + "/logs/threads-{}/rep{}-{}.log".format(t, rep, kind), LS)
                        for rep in (1, 2, 3)]
                curve = []
                for i, search_l in enumerate(LS):
                    row = dict(threads=t, scheme=kind, L=search_l, repeats=3)
                    for metric in METRICS:
                        values = [rep[i][metric] for rep in reps]
                        row[metric] = statistics.mean(values)
                        row[metric + "_stddev"] = statistics.stdev(values)
                    writer.writerow(row)
                    curve.append(row)
                curves[t, kind] = curve
    with open(a.root + "/analysis/matched-recall-summary.csv", "w", newline="") as f:
        fields = ("threads", "scheme", "target_recall", "L") + METRICS[:-1]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (t, kind), curve in curves.items():
            curve = sorted(curve, key=lambda r: r["recall"])
            for target in (80, 85, 90, 92, 95):
                if not curve[0]["recall"] <= target <= curve[-1]["recall"]:
                    continue
                for left, right in zip(curve, curve[1:]):
                    if left["recall"] <= target <= right["recall"]:
                        span = right["recall"] - left["recall"]
                        ratio = 0 if not span else (target - left["recall"]) / span
                        row = dict(threads=t, scheme=kind, target_recall=target)
                        for key in ("L",) + METRICS[:-1]:
                            row[key] = left[key] + ratio * (right[key] - left[key])
                        writer.writerow(row)
                        break
    print("PASS: 24 formal logs, 240 measured L points; summaries written")


if __name__ == "__main__":
    main()
