"""Create an isolated metric-consistent variant; never overwrite old binaries.

The low-level L2 kernel continues returning squared L2. Only dual graph
construction calls in the copied translation units take its square root.
Single-space Index construction intentionally retains its original metric.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def rewrite_calls(text, metric):
    needle = '_distance->compare('
    positions = []
    cursor = 0
    while True:
        start = text.find(needle, cursor)
        if start < 0:
            break
        line = text[text.rfind('\n', 0, start) + 1:start]
        cursor = start + len(needle)
        if '//' in line:
            continue
        depth = 1
        end = cursor
        while depth:
            if text[end] == '(':
                depth += 1
            elif text[end] == ')':
                depth -= 1
            end += 1
        positions.append((start, end))
        cursor = end
    for start, end in reversed(positions):
        text = text[:start] + 'hvs_metric_distance(' + text[start:end] + ', ' + metric + ')' + text[end:]
    return '#include "metric_contract.h"\n' + text, len(positions)


def main():
    out = ROOT / 'variant'
    out.mkdir(exist_ok=True)
    records = {}
    for filename, metric, expected in [('hybrid_index.cpp', '_dist_metric', 14),
                                        ('aux_utils.cpp', 'm', 10)]:
        source = ROOT / 'sources' / filename
        original = source.read_text(encoding='utf-8')
        patched, count = rewrite_calls(original, metric)
        # The original loop checks R only between Pareto layers and can
        # overrun R inside a layer (fixture produced degree 43 at R=40).
        # Enforce the requested budget during selection, not in the writer.
        start = patched.index('prune_neighbors_no_opt(std::vector<NeighborH>')
        loop = 'for (size_t i = 0; i < candidate.size(); i++)'
        position = patched.index(loop, start)
        patched = patched[:position] + patched[position:].replace(
            loop, 'for (size_t i = 0; i < candidate.size() && picked.size() < range; i++)', 1)
        # Fail closed if input code changes rather than silently missing a path.
        if count != expected:
            raise RuntimeError('{}: expected {} comparisons, found {}'.format(filename, expected, count))
        (out / filename).write_text(patched, encoding='utf-8')
        records[filename] = dict(comparisons=count, degree_cap_enforced=True,
            original_sha256=hashlib.sha256(original.encode()).hexdigest(),
            variant_sha256=hashlib.sha256(patched.encode()).hexdigest())
    (out / 'metric_contract.h').write_text('''#pragma once
#include "utils.h"
#include <algorithm>
#include <cmath>
inline float hvs_metric_distance(float squared, pipeann::Metric metric) {
  return metric == pipeann::Metric::L2 ? std::sqrt(std::max(0.0f, squared)) : squared;
}
''')
    (ROOT / 'analysis' / 'metric-variant.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
