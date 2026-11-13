"""Retry only unaccepted Laion repetitions after the submitted workload sweep."""
import fcntl
import traceback
import laion as l

ROOT, BASE = l.ROOT, l.BASE
base, common, monitor = l.base, l.common, l.monitor


def main():
    marker = ROOT/'laion-repair-status.json'
    base.save(marker,dict(status='waiting_for_serial_lock'))
    with (ROOT/'queue.lock').open('r') as own, (base.PREV/'queue.lock').open('r') as shared:
        fcntl.flock(own,fcntl.LOCK_EX)
        fcntl.flock(shared,fcntl.LOCK_EX)
        predecessor = base.read(BASE/'status.json')['status']
        if predecessor not in ('complete','complete_with_partial_coverage'):
            raise RuntimeError('Laion sweep did not finish: '+predecessor)
        source = base.read(BASE/'analysis/protocol.json')['index']
        workloads = base.read(base.EXP/'concurrency-alpha/analysis/manifest.json')['configurations']
        progress = base.read(BASE/'analysis/progress.json')
        for entry in progress:
            if entry['accepted']==6:
                continue
            row = next(w for w in workloads if w['dataset']=='laion' and w['fixed_alpha']==entry['alpha'])
            root = BASE/row['condition']
            l.set_root(root)
            for path,expected in row['input_sha256'].items():
                if base.digest(path)!=expected:
                    raise RuntimeError('Laion input changed: '+path)
            config = dict(source,**{k:row[k] for k in ('query1','query2','alpha','gt')})
            records_path = root/'analysis/measurements.json'
            records = base.read(records_path)
            monitor.CPUS = common.quiet_cores()
            for variant in ('A','B'):
                screens = [p for p in base.read(root/'results'/(variant+'-screen.json')) if not p['cache_evictions']]
                levels = common.levels_for(screens,entry['matched_targets'])
                expected = {p['L']:p for p in screens}
                refined = root/'results'/(variant+'-refine.json')
                if refined.exists():
                    expected.update({p['L']:p for p in base.read(refined)})
                if not all(level in expected for level in levels):
                    raise RuntimeError('missing archived refinement')
                for repeat in (1,2,3):
                    old = [r for r in records if r['variant']==variant and r['repeat']==repeat]
                    if any(r['accepted'] for r in old):
                        continue
                    for attempt in range(len(old)+1,len(old)+4):
                        label = '{}-r{}-a{}'.format(variant,repeat,attempt)
                        base.save(marker,dict(status='timing',alpha=entry['alpha'],task=label))
                        monitor.gate(label)
                        spec=(config,BASE/'full-original-order',variant=='B',None,0,False,0)
                        points=l.f.run_case(spec,levels,label)
                        base.compare_logical(points,expected,label)
                        quality=[monitor.telemetry(label,p) for p in points]
                        accepted=all(t['quiet'] for t in quality)
                        records.append(dict(variant=variant,repeat=repeat,attempt=attempt,
                                            accepted=accepted,points=points,telemetry=quality))
                        base.save(records_path,records)
                        common.summarize(records,entry['matched_targets'])
                        if accepted:
                            break
            entry['accepted']=sum(r['accepted'] for r in records)
            base.save(BASE/'analysis/progress.json',progress)
        complete=all(r['accepted']==6 for r in progress) and len(progress)==5
        status='complete' if complete else 'complete_with_partial_coverage'
        base.save(BASE/'status.json',dict(status=status,workloads=progress,updated_at=base.now()))
        base.save(marker,dict(status=status,workloads=progress))


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        base.save(ROOT/'laion-repair-status.json',dict(status='failed',error=str(error)))
        traceback.print_exc()
        raise SystemExit(1)
