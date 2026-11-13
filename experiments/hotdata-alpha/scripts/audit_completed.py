#!/usr/bin/env python3
"""Check completion and align pidstat CPU supply with measured intervals."""
import datetime
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent.parent


def main():
    files = sorted((ROOT / 'results').glob('*.json'))
    runs = [json.loads(p.read_text()) for p in files]
    assert len(runs) == 36
    assert len({(r['backend'], r['scheme'], r['threads'], r['repetition']) for r in runs}) == 36
    points = [p for r in runs for p in r['points']]
    assert len(points) == 324
    assert all(p['seconds'] >= 10 and p['samples'] % 1000 == 0 and
               p['unique_queries'] == 1000 for p in points)
    assert max(p['max_rss_kib'] for p in points) < 64 << 20
    build = json.loads((ROOT / 'analysis/build.json').read_text())
    for name, key in [('hot_search.cpp', 'driver_sha256'), ('hot_reader.h', 'reader_sha256')]:
        assert hashlib.sha256((ROOT / 'scripts' / name).read_bytes()).hexdigest() == build[key]
    assert hashlib.sha256((ROOT / 'bin/hot-search').read_bytes()).hexdigest() == build['binary_sha256']
    expected_recall = {}
    intervals = []
    for path, r in zip(files, runs):
        observations = []
        for line in (ROOT / 'logs' / (path.stem + '.pidstat')).read_text().splitlines():
            a = line.split()
            if len(a) >= 16 and a[0].isdigit() and a[-1] == 'hot-search':
                observations.append(dict(epoch=int(a[0]), cpu=float(a[6]), major_faults=float(a[9])))
        for p in r['points']:
            key = (r['scheme'], p['L'])
            if key in expected_recall:
                assert expected_recall[key] == p['recall']
            expected_recall[key] = p['recall']
            if r['backend'] == 'all-hot':
                assert p['ssd_topology_requests'] == p['ssd_coordinate_requests'] == 0
            samples = [s for s in observations
                       if p['start_unix'] + 1 <= s['epoch'] <= p['start_unix'] + p['seconds']]
            assert samples
            mean = statistics.mean(s['cpu'] for s in samples)
            intervals.append(dict(task=path.stem, backend=r['backend'], threads=r['threads'],
                scheme=r['scheme'], repetition=r['repetition'], L=p['L'], qps=p['qps'],
                mean_cpu_percent=mean, cpu_fraction_of_requested=mean/(100*r['threads']),
                major_faults_per_sec_mean=statistics.mean(s['major_faults'] for s in samples),
                low_cpu_supply_in_all_hot=r['backend']=='all-hot' and mean < 80*r['threads']))
    zone = datetime.timezone(datetime.timedelta(hours=8))
    start = min(r['started_unix'] for r in runs)
    end = max(r['finished_unix'] for r in runs)
    report = dict(completed_runs=36, measured_points=324, all_checks_pass=True,
        min_seconds=min(p['seconds'] for p in points), max_seconds=max(p['seconds'] for p in points),
        max_rss_gib=max(p['max_rss_kib'] for p in points)/2**20,
        formal_started_at=datetime.datetime.fromtimestamp(start, zone).isoformat(),
        formal_finished_at=datetime.datetime.fromtimestamp(end, zone).isoformat(),
        formal_elapsed_hours=(end-start)/3600,
        all_hot_points=sum(p['backend']=='all-hot' for p in intervals),
        all_hot_low_cpu_supply_points=sum(p['low_cpu_supply_in_all_hot'] for p in intervals),
        interpretation='CPU supply below 80% of requested cores is a diagnostic flag, not proof of a specific competing process. No timed points are removed.',
        intervals=intervals)
    (ROOT / 'analysis/audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print({k:v for k,v in report.items() if k != 'intervals'})


if __name__ == '__main__':
    main()
