"""Run the archived Laion workloads with the same Full SSD pages and filter on/off."""
import fcntl
from pathlib import Path
import signal
import struct
import traceback

import factorial as f

ROOT = f.ROOT
BASE = ROOT / 'laion-fixed'
base, common, monitor = f.base, f.common, f.monitor


def set_root(root):
    base.ROOT = common.ROOT = monitor.ROOT = f.ROOT = root
    for name in ('analysis', 'logs', 'results'):
        (root / name).mkdir(exist_ok=True)


def run():
    manifest = base.read(base.EXP / 'concurrency-alpha/analysis/manifest.json')
    workloads = sorted((row for row in manifest['configurations'] if row['dataset'] == 'laion'),
                       key=lambda row: row['fixed_alpha'])
    if [row['fixed_alpha'] for row in workloads] != [.1, .3, .5, .7, .9]:
        raise RuntimeError('expected five archived fixed-alpha workloads')
    config = next(c for c in base.read(base.FIRST / 'analysis/inputs.json') if c['name'] == 'laion')
    identity = base.read(base.PREV / 'analysis/laion-source-identity.json')['hashes']
    for field, expected in identity.items():
        if base.digest(config[field]) != expected:
            raise RuntimeError('Laion index input changed: ' + field)
    n, d1, d2, degree, slots, per_page = struct.unpack_from('<5IQ', Path(config['meta']).read_bytes())
    files = [config['topo'], config['coord']]
    if not slots or [Path(p).stat().st_size for p in files] != [
            (n+cap-1)//cap*f.PAGE for cap in (per_page, f.PAGE//(4*(d1+d2)))]:
        raise RuntimeError('unexpected archived page layout')
    folder = BASE / 'full-original-order'
    folder.mkdir(exist_ok=True)
    for group, source in zip((1, 6), files):
        target = folder / 'group-{}.bin'.format(group)
        if not target.is_symlink():
            target.symlink_to(source)
        if target.resolve() != Path(source).resolve():
            raise RuntimeError('archive link target changed')
    order = BASE / 'identity-order.bin'
    with order.open('wb') as stream:
        stream.write(struct.pack('<Q', n))
        f.np.arange(n, dtype='<u4').tofile(stream)
    config.update(order=str(order))
    base.save(BASE/'analysis/protocol.json', dict(dataset='Laion', order='original ID', dimensions=[d1,d2],
              full_interval_slots=slots, degree=degree, cache_bytes=0, threads=4, repeats=3, numa=3,
              index=config, fixed_workloads=workloads, source_hashes=identity))
    summary = []
    for row in workloads:
        set_root(BASE)
        base.status('working_on_fixed_alpha', alpha=row['fixed_alpha'])
        w = BASE / row['condition']
        w.mkdir(exist_ok=True)
        set_root(w)
        for path, expected in row['input_sha256'].items():
            if base.digest(path) != expected:
                raise RuntimeError('archived query/GT input changed: '+path)
        c = dict(config, **{k: row[k] for k in ('query1','query2','alpha','gt')})
        specs = {v:(c,folder,v=='B',None,0,False,0) for v in ('A','B')}
        monitor.CPUS = common.quiet_cores()
        screens = {v: f.run_case(spec,[20,40,60,80,100,120,160,200,240,320,400,600,800],
                                 v+'-screen',True,screening=True) for v,spec in specs.items()}
        coverage = {v:[t for t in f.TARGETS if base.bracket(points,t)] for v,points in screens.items()}
        targets = [t for t in f.TARGETS if all(t in x for x in coverage.values())]
        base.save(w/'analysis/coverage.json', dict(alpha=row['fixed_alpha'], by_variant=coverage,
                  matched_targets=targets, max_recall={v:max(p['recall'] for p in pts)
                  for v,pts in screens.items()}))
        records = common.benchmark(specs,screens,targets) if targets else []
        summary.append(dict(alpha=row['fixed_alpha'], matched_targets=targets,
                            accepted=sum(r['accepted'] for r in records)))
        base.save(BASE/'analysis/progress.json',summary)
    set_root(BASE)
    base.status('complete' if all(r['accepted']==6 for r in summary) else 'complete_with_partial_coverage',
                workloads=summary)


def main():
    BASE.mkdir(exist_ok=True)
    set_root(BASE)
    base.status('waiting_for_openimages')
    with (ROOT/'queue.lock').open('r') as own, (base.PREV/'queue.lock').open('r') as shared:
        fcntl.flock(own,fcntl.LOCK_EX)
        fcntl.flock(shared,fcntl.LOCK_EX)
        prior = base.read(ROOT/'status.json')['status']
        if prior not in ('complete','complete_with_flagged_points'):
            base.status('blocked_by_openimages', predecessor=prior)
            return
        base.status('verifying_archived_inputs')
        common.run_search = f.run_case
        run()


if __name__ == '__main__':
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
        signal.signal(sig,base.stop)
    try:
        main()
    except Exception as exc:
        base.ROOT = BASE
        base.status('interrupted' if isinstance(exc,InterruptedError) else 'failed',error=str(exc))
        traceback.print_exc()
        raise SystemExit(1)
