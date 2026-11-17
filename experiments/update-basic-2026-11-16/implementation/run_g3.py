"""G3 only after both G2 modes pass. Two full 600-second sanitizer runs."""
import fcntl
import json
import os
import shutil
import time
import run_g0 as c
import run_g1
import run_g2 as g


def main():
    g2_run=c.ROOT/'runtime'/os.environ.get('HVS_UPDATE_G2_RUN','g2-entry-parity')
    root=c.ROOT/'runtime'/os.environ.get('HVS_UPDATE_G3_RUN','g3-entry-parity');root.mkdir(exist_ok=True);c.RUN=root
    lock=(g.E/'budget-reorder-20261103/queue.lock').open('a+')
    g.status('waiting_global_lock');fcntl.flock(lock,fcntl.LOCK_EX)
    try:
        gate=json.loads((g2_run/'status.json').read_text())
        assert gate['stage']=='G2_PASSED_BOTH_DISTANCES','G2 both-mode prerequisite not satisfied'
        identity=json.loads((g2_run/'build-identity.json').read_text())
        for path,digest in identity['sources'].items():assert c.sha(path)==digest
        src=c.ROOT/'implementation';io=c.SOURCE/'src/utils/linux_aligned_file_reader.cpp'
        g.run(['clang-format','-style=LLVM','-i',src/'g3_concurrent.cpp',src/'topology_interleave.cpp'],'format-g3')
        base=[v for v in identity['flags'] if v not in ('-O3','-O2','-static-libstdc++','-static-libgcc')]
        flags=base+['-O1','-g','-fno-omit-frame-pointer','-fno-optimize-sibling-calls']
        link=[c.BUILD/'src/libpipeann.a','-L/mnt/nvme2/wz/disk/local/openblas/lib','-laio','-ljemalloc','-lopenblas','-pthread']
        inputs=[src/'update_store.cpp',src/'deg_prune.cpp',src/'dual_update.cpp',io]
        results=[]
        for sanitizer,flag in [('asan','address'),('tsan','thread')]:
            c.RUN=root/sanitizer;c.RUN.mkdir(exist_ok=False)
            for tool,main_source in [('store_smoke','store_smoke.cpp'),('topology_interleave','topology_interleave.cpp'),('g3_concurrent','g3_concurrent.cpp')]:
                units=([src/'update_store.cpp',io] if tool!='g3_concurrent' else inputs)+[src/main_source]
                command=['g++-11']+flags+['-fsanitize='+flag]+units+link+['-o',c.RUN/tool]
                g.run(command,'compile-'+tool,timeout=900)
                c.save(tool+'-build-identity.json',dict(flags=flags+['-fsanitize='+flag],command=list(map(str,command)),
                    sources={str(p):c.sha(p) for p in units},binary=c.sha(c.RUN/tool),upstream_library=c.sha(c.BUILD/'src/libpipeann.a'),
                    coverage='new executor, pruning, test driver, header-instantiated PQ and upstream Linux reader instrumented; remaining prebuilt upstream support and external libraries uninstrumented'))
            # Directed same-page read/write interleaving under each sanitizer.
            smoke=c.RUN/'store';smoke.mkdir()
            options=dict(ASAN_OPTIONS='halt_on_error=1:abort_on_error=1:detect_leaks=1',TSAN_OPTIONS='halt_on_error=1:exitcode=66')
            g.run([c.RUN/'store_smoke',smoke],'directed-page-interleaving',extra=options)
            topology=c.RUN/'topology-case';topology.mkdir()
            g.run([c.RUN/'topology_interleave',topology],'directed-topology-record',extra=options)
            g.run([c.RUN/'g3_concurrent','--oracle-self-test'],'live-interval-oracle-check',extra=options)
            fixture=run_g1.fixture('pq')
            for cohort in ('tune','report'):shutil.copyfile(g.MG/'data'/(cohort+'-ids.txt'),fixture/(cohort+'-ids.txt'))
            g.run([c.RUN/'g3_concurrent',fixture,g.AB/'data-10k/base_img_emb.fvecs',g.AB/'data-10k/base_text_emb.fvecs',
                g.DATA/'query_img_emb.fvecs',g.DATA/'query_text_emb.fvecs',600],'ten-minute-concurrent',timeout=7200,extra=options)
            result=json.loads((fixture/'concurrent-result.json').read_text());assert result['passed']
            results.append(dict(sanitizer=sanitizer,result=result))
            (root/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        c.RUN=root;g.status('G3_PASSED_ASAN_AND_TSAN',duration_each=600,query_threads=4,update_threads=2,
            recall='conservative interval lower bound, overall and every rolling 500 queries per alpha >=93%',
            G1_G2_G4='PASSED_BOTH_DISTANCES',E1='NOT_STARTED_USER_CHECKPOINT')
    except Exception as e:
        c.RUN=root;g.status('STOPPED_AT_G3',reason=str(e),dependent_experiments='not started');raise


if __name__=='__main__':main()
