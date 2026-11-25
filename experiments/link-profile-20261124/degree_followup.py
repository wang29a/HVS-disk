"""Repeat the original legacy and K=8 shards with the same degree cap."""

import argparse
from itertools import count
import os
from pathlib import Path

from build import read, sha
from run import obtain, save, status
from timing_guard import guarded_call, quiet_window


def compare(root, original, lock):
    cpus = set(range(16, 24))
    if os.sched_getaffinity(0) != cpus:
        raise RuntimeError("fair Link timing uses physical CPUs 16–23")
    identity = read(original / "build/identity.json")["binaries"]["on"]
    binary = Path(identity["binary"])
    if sha(binary) != identity["sha256"]:
        raise RuntimeError("frozen Link binary changed")
    choices = read(original / "analysis/inputs.json")["shards"]
    for tag in ("k8", "legacy"):
        folder = original / "data" / tag
        for file, digest in choices[tag]["material_sha256"].items():
            if sha(folder / file) != digest:
                raise RuntimeError("frozen shard material changed")
    comparison = {}
    for tag in ("k8", "legacy"):
        folder = original / "data" / tag
        item = choices[tag]
        label = tag + "-L200-C750-R26"
        for attempt in count(1):
            slot = "%s-attempt-%03d" % (label, attempt)
            runs = root / "runs" / slot
            audit = runs / "environment.json"
            if audit.exists():
                environment = read(audit)
                if not environment["valid"]:
                    continue
                result = read(runs / "result.json")
                if sha(runs / "float-graph.bin") != result["graph_sha256"]:
                    raise RuntimeError("existing Link output hash changed")
            elif runs.exists():
                # An orphaned run has no verified SMT record.
                continue
            else:
                with quiet_window(root, slot, lock, cpus) as preflight:
                    result, environment = guarded_call(
                        cpus, lambda: obtain(root, slot, binary, folder / (tag + "-0.bin"),
                                             folder / (tag + "-1.bin"), item["nodes"], 200, 26, 750),
                        preflight)
                save(audit, environment)
                if not environment["valid"]:
                    status(root, "RETRY_SMT_INTERFERENCE", item=slot, peak=environment["max_sibling_busy"])
                    continue
            comparison[tag] = dict(shard=item["name"], seconds=result["link_s"], nodes=item["nodes"],
                                   us_per_point=result["link_s"] * 1e6 / (item["nodes"] - 1),
                                   run=slot, environment=environment)
            save(root / "analysis/fair-link.json", comparison)
            break
    baseline = next(row for row in read(original / "scan-resume/analysis/results.json")
                    if row["shard"] == choices["k8"]["name"] and (row["L"], row["C"], row["R"]) == (200, 750, 40))
    comparison["k8_original_R40"] = dict(seconds=baseline["link_s"], nodes=baseline["nodes"],
                                         us_per_point=baseline["link_s"] * 1e6 / (baseline["nodes"] - 1))
    save(root / "analysis/fair-link.json", comparison)
    status(root, "FAIR_LINK_COMPLETE", report="analysis/fair-link.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "original", "lock"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    (args.root / "analysis").mkdir(parents=True, exist_ok=True)
    try:
        compare(args.root, args.original, args.lock)
    except (Exception, KeyboardInterrupt) as error:
        status(args.root, "FAIR_LINK_STOPPED", reason=str(error))
        raise
