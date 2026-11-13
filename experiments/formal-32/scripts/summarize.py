#!/usr/bin/env python3
import csv
import glob
import json
import os
import re
from collections import defaultdict

archive = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
search_root = os.path.join(archive, "raw", "search")
datasets = ("cc3m", "openimage", "howto100m")
kinds = ("baseline", "full", "merge", "no-alpha")
metrics = ("qps", "latency_us", "p999_us", "hops", "comparisons",
           "topology_ios", "rerank_ios", "cache_hits", "recall")
rows = defaultdict(lambda: defaultdict(list))

for dataset in datasets:
    for kind in kinds:
        pattern = os.path.join(search_root, dataset, "rep*-{}.log".format(kind))
        for path in glob.glob(pattern):
            with open(path, errors="replace") as handle:
                for line in handle:
                    fields = line.split()
                    if len(fields) < 17 or not re.fullmatch(r"\d+", fields[0]):
                        continue
                    search_l = int(fields[0])
                    if search_l not in (20, 40, 60, 80, 100, 120, 160, 200, 300, 400):
                        continue
                    values = {
                        "qps": float(fields[2]), "latency_us": float(fields[3]),
                        "p999_us": float(fields[4]), "hops": float(fields[5]),
                        "comparisons": float(fields[6]), "topology_ios": float(fields[7]),
                        "rerank_ios": float(fields[8]), "cache_hits": float(fields[9]),
                        "recall": float(fields[-1]),
                    }
                    rows[(dataset, kind)][search_l].append(values)

curves = {}
with open(os.path.join(archive, "search-curve-summary.csv"), "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=("dataset", "scheme", "L", "repeats") + metrics)
    writer.writeheader()
    for dataset in datasets:
        for kind in kinds:
            curve = []
            for search_l in sorted(rows[(dataset, kind)]):
                samples = rows[(dataset, kind)][search_l]
                avg = {metric: sum(x[metric] for x in samples) / len(samples) for metric in metrics}
                avg.update({"dataset": dataset, "scheme": kind, "L": search_l,
                            "repeats": len(samples)})
                writer.writerow(avg)
                curve.append(avg)
            curves[(dataset, kind)] = curve


def interpolate(curve, target):
    ordered = sorted(curve, key=lambda row: row["recall"])
    if not ordered or target < ordered[0]["recall"] or target > ordered[-1]["recall"]:
        return None
    for left, right in zip(ordered, ordered[1:]):
        if left["recall"] <= target <= right["recall"]:
            span = right["recall"] - left["recall"]
            ratio = 0 if span == 0 else (target - left["recall"]) / span
            return {key: left[key] + ratio * (right[key] - left[key])
                    for key in ("L",) + metrics[:-1]}


with open(os.path.join(archive, "matched-recall-summary.csv"), "w", newline="") as handle:
    fields = ("dataset", "target_recall", "scheme") + ("L",) + metrics[:-1]
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for dataset in datasets:
        for target in (80.0, 85.0, 90.0, 92.0, 95.0):
            values = [(kind, interpolate(curves[(dataset, kind)], target)) for kind in kinds]
            if any(value is None for _, value in values):
                continue
            for kind, value in values:
                value.update({"dataset": dataset, "target_recall": target, "scheme": kind})
                writer.writerow(value)

with open(os.path.join(archive, "characterization-summary.csv"), "w", newline="") as handle:
    fields = ("dataset", "nodes", "edges", "full_slots", "full_nodes_per_page",
              "full_slot_utilization", "full_filter_probability", "merge_filter_probability",
              "merge_false_positive_probability", "merge_false_negative_probability",
              "full_unique_range_tuple_signatures")
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for dataset in datasets:
        with open(os.path.join(archive, "raw", "characterization", dataset + ".json")) as source:
            data = json.load(source)
        writer.writerow({
            "dataset": dataset, "nodes": data["nodes"], "edges": data["edges"],
            "full_slots": data["full"]["slots"],
            "full_nodes_per_page": data["full"]["nodes_per_page"],
            "full_slot_utilization": data["full"]["slot_utilization"],
            "full_filter_probability": data["full"]["query_weighted_filter_probability"],
            "merge_filter_probability": data["merge"]["query_weighted_filter_probability"],
            "merge_false_positive_probability":
                data["merge_vs_full"]["false_positive_active_probability"],
            "merge_false_negative_probability":
                data["merge_vs_full"]["false_negative_active_probability"],
            "full_unique_range_tuple_signatures": data["full_unique_range_tuple_signatures"],
        })

with open(os.path.join(archive, "gated-format-summary.csv"), "w", newline="") as handle:
    fields = ("dataset", "threshold", "nodes", "a0_nodes", "a0_fraction", "a2_nodes",
              "merge_topology_bytes", "gated_topology_bytes", "node_map_bytes",
              "gated_total_bytes", "topology_reduction_vs_merge",
              "total_reduction_vs_merge", "added_active_edge_probability",
              "degree_or_neighbor_mismatches")
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for dataset in datasets:
        with open(os.path.join(archive, "raw", "characterization", dataset + ".json")) as source:
            characterization = json.load(source)
        gated_by_threshold = {
            round(row["filter_threshold"], 2): row
            for row in characterization["benefit_gated_a0_a2"]
        }
        merge_bytes = ((characterization["nodes"]
                        + characterization["merge"]["nodes_per_page"] - 1)
                       // characterization["merge"]["nodes_per_page"] * 8192)
        for threshold in (0.20, 0.30):
            path = os.path.join(archive, "raw", "gated", dataset,
                                "gated-{:.2f}.manifest.json".format(threshold))
            with open(path) as source:
                manifest = json.load(source)
            total = manifest["topology_bytes"] + manifest["map_bytes"]
            offline = gated_by_threshold[threshold]
            writer.writerow({
                "dataset": dataset, "threshold": threshold, "nodes": manifest["nodes"],
                "a0_nodes": manifest["a0"]["nodes"],
                "a0_fraction": manifest["a0"]["nodes"] / manifest["nodes"],
                "a2_nodes": manifest["a2"]["nodes"],
                "merge_topology_bytes": merge_bytes,
                "gated_topology_bytes": manifest["topology_bytes"],
                "node_map_bytes": manifest["map_bytes"], "gated_total_bytes": total,
                "topology_reduction_vs_merge": 1.0 - manifest["topology_bytes"] / merge_bytes,
                "total_reduction_vs_merge": 1.0 - total / merge_bytes,
                "added_active_edge_probability": offline["added_active_edge_probability"],
                "degree_or_neighbor_mismatches": manifest["degree_or_neighbor_mismatches"],
            })
