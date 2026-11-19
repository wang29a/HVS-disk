#!/usr/bin/env python3
"""Render E1/E2 paper plots from read-only archived JSON summaries."""

import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


WEIGHTS = (0.1, 0.3, 0.5, 0.7, 0.9)
MODES = ("exact", "pq")
STYLE = {
    "exact": dict(color="#0072B2", marker="o"),
    "pq": dict(color="#D55E00", marker="s"),
    "rebuild": dict(color="#343434", marker="D"),
}


def read(path):
    with path.open(encoding="utf-8") as data:
        return json.load(data)


def add(series, key, value):
    series.setdefault(key, []).append(value)


def report_expansions(rows, points):
    """Select measured 95% ID recall on the exact-navigation report cohort."""
    for row in rows:
        if row["medium"] == "dram" and row["cohort"] == "report":
            if row["n"] != points or row["navigation"] != "exact":
                raise ValueError("Unexpected population or navigation in DRAM report cohort")
            matched = row["matched"]
            if matched["covered"]:
                yield row["alpha"], matched["expanded"]


def collect_e1(root):
    plan = read(root / "variants-complete.json")
    batch_size = read(root / "identity.json")["insertions_per_batch"]
    expansion, mean_degree, full_degree, throughput = {}, {}, {}, {}
    checkpoints = set()
    for row in plan:
        mode = row["mode"]
        if mode not in MODES:
            continue
        variant = root / "variants" / row["name"]
        for cp in range(row["checkpoints"]):
            folder = variant / f"checkpoint-{cp}"
            structure = read(folder / "structure.json")
            inserted = batch_size * cp
            if structure["points"] != 1_800_000 + inserted:
                raise ValueError(f"Unexpected E1 population in {folder}")
            checkpoints.add(inserted)
            add(mean_degree, (mode, inserted), structure["average_degree"])
            add(full_degree, (mode, inserted), structure["full_degree_fraction"])
            for alpha, value in report_expansions(read(folder / "matched.json"), structure["points"]):
                add(expansion, (mode, inserted, alpha), value)
            if cp:
                stats = read(folder / "update-summary.json")
                if stats["insertions"] != batch_size:
                    raise ValueError(f"Unexpected E1 batch size in {folder}")
                add(throughput, (mode, inserted), stats["insert_qps"])

    # The archived R1 is a single 2M rebuild, comparable at the E1 endpoint.
    for alpha, value in report_expansions(read(root / "reference-R1/matched.json"), 2_000_000):
        add(expansion, ("rebuild", 200_000, alpha), value)
    return sorted(checkpoints), expansion, mean_degree, full_degree, throughput


def collect_e2(root):
    trajectories = read(root / "E2/trajectories.json")
    expansion, mean_degree, full_degree = {}, {}, {}
    seen = set()
    for run in trajectories:
        key = run["mode"], run["repetition"]
        if key in seen:
            raise ValueError(f"Duplicate E2 trajectory {key}")
        seen.add(key)
        mode = key[0]
        if mode not in MODES or not run["merge"]["completed"]:
            continue
        for stage, when in enumerate(("before", "after")):
            structure = run[when]
            if structure["live"] != 1_800_000:
                raise ValueError(f"Unexpected E2 live population in {key}")
            add(mean_degree, (mode, stage), structure["mean_live_source_degree"])
            add(full_degree, (mode, stage), structure["full_degree_fraction"])
            # E2 retains 2M physical IDs after deleting 200k live objects.
            for alpha, value in report_expansions(run[when + "_measurements"], structure["initial"]):
                add(expansion, (mode, stage, alpha), value)
    for alpha, value in report_expansions(read(root / "R2/matched.json"), 1_800_000):
        add(expansion, ("rebuild", 2, alpha), value)
    structure = read(root / "R2/structure.json")
    add(mean_degree, ("rebuild", 2), structure["mean_live_source_degree"])
    add(full_degree, ("rebuild", 2), structure["full_degree_fraction"])
    return expansion, mean_degree, full_degree, {
        mode: sum(m == mode for m, _ in seen) for mode in MODES
    }


def curve(ax, source, points, mode, *suffix):
    measured = [(p, source[(mode, p, *suffix)]) for p in points
                if (mode, p, *suffix) in source]
    if not measured:
        return
    x, samples = zip(*measured)
    y = [mean(values) for values in samples]
    lower = [avg - min(values) for avg, values in zip(y, samples)]
    upper = [max(values) - avg for avg, values in zip(y, samples)]
    ax.errorbar(x, y, yerr=[lower, upper], linestyle="-" if len(x) > 1 else "None",
                linewidth=1.2, markersize=3.8, capsize=2, **STYLE[mode])


def legend(ax, reference=None, **options):
    labels = {"exact": "Exact", "pq": "PQ", "rebuild": reference}
    handles = [Line2D([0], [0], label=label, markersize=4,
                      linestyle="None" if mode == "rebuild" else "-", **STYLE[mode])
               for mode, label in labels.items() if label]
    ax.legend(handles=handles, frameon=False, **options)


def save(fig, output, name):
    for ext in ("pdf", "png"):
        fig.savefig(output / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def quality_plot(data, points, tick_labels, reference, xlabel, output, name, note=None):
    fig, axes = plt.subplots(2, 3, figsize=(6.9, 3.9), layout="constrained")
    for alpha, ax in zip(WEIGHTS, axes.flat):
        for mode in (*MODES, "rebuild"):
            curve(ax, data, points, mode, alpha)
        ax.set_title(rf"$\alpha = {alpha:.1f}$")
        ax.set_xticks(points, tick_labels)
    key = axes.flat[-1]
    key.set_axis_off()
    legend(key, reference, loc="center")
    fig.supxlabel(xlabel)
    fig.supylabel("Expanded nodes/query at 95% ID recall")
    if note:
        fig.suptitle(note, fontsize=9)
    save(fig, output, name)


def degree_plot(degree, full, points, labels, xlabel, output, name, reference=None):
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.55), layout="constrained")
    for ax, data, title in zip(axes, (degree, full), ("Mean out-degree", "Nodes at degree limit")):
        for mode in (*MODES, "rebuild"):
            curve(ax, data, points, mode)
        ax.set_ylabel(title)
        ax.set_xticks(points, labels)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    legend(axes[0], reference, loc="best")
    fig.supxlabel(xlabel)
    save(fig, output, name)


def draw(e1, e2, output):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": "#e5e5e5", "grid.linewidth": 0.55,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    points, expansions, degree, full, throughput = collect_e1(e1)
    labels = [str(p // 1000) for p in points]
    quality_plot(expansions, points, labels, "R1 rebuild (2M only)",
                 "Inserted objects (thousands)", output, "e1-expansions")
    degree_plot(degree, full, points, labels, "Inserted objects (thousands)",
                output, "e1-degree")
    fig, ax = plt.subplots(figsize=(3.45, 2.55), layout="constrained")
    for mode in MODES:
        curve(ax, throughput, points[1:], mode)
    ax.set_xticks(points[1:], labels[1:])
    ax.set_xlabel("Cumulative inserted objects (thousands)")
    ax.set_ylabel("Batch throughput (objects/s)")
    legend(ax, loc="best")
    save(fig, output, "e1-insert-throughput")

    expansions, degree, full, counts = collect_e2(e2)
    stages = (0, 1, 2)
    labels = ("Marked", "Merged", "R2")
    note = f"Archived E2 runs: exact n={counts['exact']}, PQ n={counts['pq']}"
    quality_plot(expansions, stages, labels, "R2 rebuild (1.8M live)",
                 "Deletion and consolidation stage", output, "e2-expansions", note)
    degree_plot(degree, full, stages, labels, "Deletion and consolidation stage",
                output, "e2-degree", "R2 rebuild")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e1-archive", required=True, type=Path)
    parser.add_argument("--e2-archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    archives = (args.e1_archive.resolve(), args.e2_archive.resolve())
    output = args.output.resolve()
    if any(output == root or root in output.parents for root in archives):
        parser.error("--output must be outside both read-only input archives")
    output.mkdir(parents=True, exist_ok=True)
    draw(*archives, output)


if __name__ == "__main__":
    main()
