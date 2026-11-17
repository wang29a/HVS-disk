"""Fresh updated graphs after fixing entry semantics; retain old evidence."""
import fcntl
import json
import shutil
import run_g0 as c
import run_g2 as g


def main():
    old=c.ROOT/'runtime/g2-attempt2';old_out=g.OUT
    c.RUN=c.ROOT/'runtime/g2-entry-parity';c.RUN.mkdir(exist_ok=True)
    g.OUT=old_out.parent/'g2-entry-parity';g.DEPTHS=[10,12,14]+g.DEPTHS
    lock=(g.E/'budget-reorder-20261103/queue.lock').open('a+')
    g.status('waiting_global_lock');fcntl.flock(lock,fcntl.LOCK_EX)
    try:
        assert json.loads((c.ROOT/'runtime/g1-entry-parity/status.json').read_text())['stage']=='G1_PASSED_BOTH_DISTANCES','corrected G1 prerequisite failed'
        if g.OUT.exists():raise RuntimeError('new entry-parity fixtures already exist')
        retained=json.loads((old/'resume-identity.json').read_text())['retained_artifacts']
        retained={p:h for p,h in retained.items() if g.Path(p).relative_to(old_out).parts[0] in ('full','initial')
                  or g.Path(p).name.startswith(('source-ids','base-','initial-','gt-','alpha-','centers','xy.'))}
        for path,digest in retained.items():assert c.sha(path)==digest,'immutable baseline changed '+path
        g.OUT.mkdir();(c.RUN/'artifacts').symlink_to(g.OUT,target_is_directory=True)
        for p in old_out.iterdir():
            if p.is_file() and (p.name.startswith(('source-ids','base-','initial-','gt-','alpha-','centers','xy.'))) or p.name in ('full','initial'):
                (g.OUT/p.name).symlink_to(p,target_is_directory=p.is_dir())
        c.save('baseline-reuse-identity.json',dict(reason='fix minimizing entry-policy bug to verified maximizing builder policy',
            retained_artifacts=retained,baseline='unchanged B full and initial graphs; same subset and GT',
            new_mutable_directory=str(g.OUT),previous_invalid_update_results=str(old)))
        src=c.ROOT/'implementation'
        flags=json.loads((c.ROOT/'runtime/g1-entry-parity/build-identity.json').read_text())['flags']
        files=[src/v for v in ('update_store.cpp','deg_prune.cpp','dual_update.cpp','g2_insert.cpp')]
        g.run(['g++-11']+flags+files+[c.BUILD/'src/libpipeann.a','-L/mnt/nvme2/wz/disk/local/openblas/lib','-laio','-ljemalloc','-lopenblas','-pthread','-o',c.RUN/'g2_insert'],'compile-g2',600)
        c.save('build-identity.json',dict(flags=flags,sources={str(p):c.sha(p) for p in files+[src/'entry_frontier.h',src/'dual_update.h',src/'update_store.h']},binary=c.sha(c.RUN/'g2_insert')))
        # Baseline measured with exactly the same verified static executor and
        # unchanged files, so reuse all three frozen report rounds verbatim.
        full=json.loads((old/'full-matched.json').read_text());c.save('full-matched.json',full)
        c.save('baseline-query-identity.json',dict(results=c.sha(old/'full-matched.json'),source=str(old),static_executor=c.sha(g.MG/'bin/multigraph')))
        comparisons=[]
        for mode in ('pq','exact'):
            fixture=g.fixture(mode);g.run([c.RUN/'g2_insert',fixture,mode],'insert-'+mode,timeout=14400)
            rows=g.evaluate(mode,fixture/'eval.meta',fixture/'eval.topo');c.save(mode+'-matched.json',rows)
            for row,base in zip(rows,full):
                increase=row['matched']['expanded']/base['matched']['expanded']-1
                comparisons.append(dict(mode=mode,alpha=row['alpha'],cohort=row['cohort'],repeat=row['repeat'],baseline=base['matched'],incremental=row['matched'],increase=increase,passed=increase<=.15))
            c.save('comparisons.json',comparisons)
            if any(not r['passed'] for r in comparisons if r['cohort']=='report'):raise RuntimeError('G2 expansion increase exceeds15%; see comparisons.json')
        g.status('G2_PASSED_BOTH_DISTANCES',G4='PASSED_G1_AND_G2_BOTH',G3='NOT_RUN')
    except Exception as e:
        g.status('STOPPED_AT_G2',reason=str(e),dependent_experiments='not started');raise


if __name__=='__main__':main()
