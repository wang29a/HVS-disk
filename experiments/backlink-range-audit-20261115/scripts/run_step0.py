"""Gate0a against immutable formal library; do not launch anything past failure."""
import signal,time,hashlib,json
import common as c

def main():
    for name in ('logs','commands','step0a','bin','validation'):(c.RUN/name).mkdir(parents=True,exist_ok=True)
    frozen=c.load(c.ROOT/'FREEZE.json');data=(c.ROOT/'README.md').read_bytes()
    if hashlib.sha256(data[:frozen['bytes']]).hexdigest()!=frozen['sha256']:raise RuntimeError('protocol prefix changed')
    lock=c.lock();c.capacity(2*2**30,'backlink audit compile and method gate; no full graph output yet')
    proof=c.load(c.U/'stage2-v2/validation/semantic-proof.json')['binaries']['unified'];lib=c.U/'stage2-v2/bin/libpipeann-unified.a';source=c.U/'stage2-v2/variant/unified/hybrid_index.cpp'
    if c.sha(lib)!=proof['library_sha256'] or c.sha(source)!=proof['source_sha256']:raise RuntimeError('formal library/source identity mismatch')
    cfg=c.load(c.E/'metric-consistent-20261111/analysis/build.json');flags=cfg['flags']+['-I'+str(c.U/'scripts'),'-fno-fast-math','-ffp-contract=off'];cpp=c.ROOT/'scripts/audit_idempotence.cpp';binary=c.RUN/'bin/audit-idempotence'
    c.run(['clang-format','-style=LLVM','-i',cpp],'format-method-gate',cpus='80,81',timeout=60)
    c.run(['g++-11']+flags+[cpp,lib]+cfg['link']+['-o',binary],'compile-method-gate',cpus='80,81',timeout=600,extra=dict(OMP_NUM_THREADS='2',OMP_THREAD_LIMIT='2'))
    c.save(c.RUN/'validation/build.json',dict(status='COMPILED',formal_library_sha256=c.sha(lib),formal_source_sha256=c.sha(source),test_sha256=c.sha(cpp),binary_sha256=c.sha(binary),flags=flags,link=cfg['link'],assertions='explicit checks; NDEBUG preserved',protocol_prefix=frozen))
    try:c.run([binary,c.RUN/'step0a'],'run-method-gate',cpus='80,81',timeout=900,extra=dict(OMP_NUM_THREADS='2',OMP_THREAD_LIMIT='2'))
    except RuntimeError:
        path=c.RUN/'step0a/result.json'
        if not path.exists() or c.load(path).get('status')!='FAIL':raise
        c.save(c.RUN/'step0.json',dict(status='STOP_GATE0A_FAILED',step0a=c.load(path),step0b='not_run',step1='not_run',step2='not_run',step3='not_authorised',reason='formal pruner not idempotent; repruning differences cannot yet be attributed solely to copied reverse ranges'))
        c.status('stopped_at_method_gate',reason='step0a failed as frozen protocol requires',result=str(c.RUN/'step0.json'));return
    c.save(c.RUN/'step0.json',dict(status='GATE0A_PASS_PROVENANCE_PENDING',step0a=c.load(c.RUN/'step0a/result.json'),step0b='pending_implementation_and_build',step1='not_run',step2='not_run',step3='not_authorised'))
    c.status('method_gate_passed_provenance_next',next='step0b required before graph audit')
if __name__=='__main__':
    signal.signal(signal.SIGTERM,c.stop);signal.signal(signal.SIGINT,c.stop)
    try:main()
    except Exception as e:c.stop();c.status('failed',error=str(e));raise
