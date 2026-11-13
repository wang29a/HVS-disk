"""Validate NUMA policy using observations inside the measurement window.

An exited but unreaped Linux process may retain /proc/PID/status while its
numa_maps is already empty. Empty mappings are missing evidence, not evidence
that a prior live process lost its memory policy.
"""
import os
from pathlib import Path
import time

VERSION = 'numa-window-v2'


def parse_maps(text):
    anonymous = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        counts = [int(f.split('=', 1)[1]) for f in fields if f.startswith('anon=')]
        if counts and counts[0] > 0:
            anonymous.append((counts[0], fields[1]))
    if not anonymous:
        return None
    largest = max(anonymous)
    return dict(largest_anon_pages=largest[0], largest_anon_policy=largest[1],
                verified=largest[1] == 'bind:3',
                conflicting_policies=sorted(set(policy for _, policy in anonymous
                    if policy.startswith(('bind:', 'interleave:')) and policy != 'bind:3')))


def observe(pid):
    root = Path('/proc')/str(pid)
    try:
        # Exclude nice/ionice/numactl startup, where exec has not yet installed
        # the final CPU/memory policy and executable mappings.
        if Path(os.readlink(str(root/'exe'))).name != 'metric-search':
            return None
        lines = (root/'status').read_text().splitlines()
        status = dict(line.split(':', 1) for line in lines if ':' in line)
        if status.get('State', '').strip().startswith('Z'):
            return None
        policy = parse_maps((root/'numa_maps').read_text())
        if policy is None:
            return None
        return dict(time=time.time(), cpu_mask=status.get('Cpus_allowed_list', '').strip(),
                    cpuset_mems=status.get('Mems_allowed_list', '').strip(), **policy)
    except (OSError, ValueError):
        return None


def validate(observations, start, end):
    live = [v for v in observations if v is not None and start <= v['time'] <= end]
    reasons = []
    if len(live) < 3:
        reasons.append('insufficient_live_numa_samples_in_measurement_window')
    if any(v['cpu_mask'] != '72-75' for v in live):
        reasons.append('wrong_cpu_affinity')
    if any(not v['verified'] or v['conflicting_policies'] for v in live):
        reasons.append('wrong_live_memory_policy')
    return dict(version=VERSION, verified=not reasons, rejection_reasons=reasons,
                live_window_samples=live)
