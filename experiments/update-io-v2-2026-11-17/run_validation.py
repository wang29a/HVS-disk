"""New branch only: bitwise parity, G1, directed interleaving and ASan/TSan G3.
Never invokes E1 or its coordinator. All heavy work holds the shared queue lock.
"""
from pathlib import Path
import json,sys,shutil,fcntl,time,os
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'update-perf-audit-2026-11-17'))
import run_suite as s
OUT=ROOT/'runtime'/os.environ.get('HVS_OPT_VALIDATION_RUN','validation');SRC=ROOT/'implementation'
REFERENCE=ROOT/'runtime'/os.environ['HVS_OPT_REFERENCE_RUN'] if os.environ.get('HVS_OPT_REFERENCE_RUN') else None

def fixture(name,parent):
 old=s.c.RUN;s.c.RUN=parent;parent.mkdir(parents=True,exist_ok=True)
 try:r=s.run_g1.fixture(name)
 finally:s.c.RUN=old
 for which,axis in [('img','x'),('text','y')]:
  (r/('query-'+axis+'.bin')).symlink_to(s.DATA/('query_'+which+'_emb.fvecs'))
 for cohort in ('report','tune'):shutil.copyfile(s.MG/(cohort+'-ids.txt'),r/(cohort+'-ids.txt'))
 return r

def main():
 OUT.mkdir(exist_ok=False);s.OUT=OUT
 lock=(s.E/'budget-reorder-20261103/queue.lock').open('a+');s.status('WAITING_GLOBAL_LOCK');fcntl.flock(lock,fcntl.LOCK_EX)
 try:
  assert json.loads((ROOT/'runtime/baseline/status.json').read_text())['stage']=='BASELINE_COMPLETE'
  assert json.loads(Path('/mnt/pmem0/hvs-update-basic-2026-11-16/e1-degree-protected-v2/status.json').read_text())['stage']=='PAUSED_AFTER_EXACT_R1_USER_REQUEST'
  gate=json.loads((s.OLD/'runtime/g1-in-degree-protection-v2/build-identity.json').read_text());oldsrc=s.OLD/'implementation'
  for name,digest in gate['sources'].items():assert s.sha(oldsrc/name)==digest
  flags=['-I'+str(SRC)]+gate['flags'];link=[s.c.BUILD/'src/libpipeann.a','-L/mnt/nvme2/wz/disk/local/openblas/lib','-laio','-ljemalloc','-lopenblas','-pthread']
  units=[SRC/n for n in ('update_store.cpp','deg_prune.cpp','dual_update.cpp')]
  touched=['update_store.cpp','update_store.h','dual_update.cpp','dual_update.h','in_degree_ledger.h','atomic_degree_check.cpp','optimistic_store_check.cpp','g3_concurrent.cpp']
  s.run(['clang-format','-style=LLVM','-i']+[SRC/n for n in touched]+[ROOT/'static_query.cpp'],'format-new-sources')
  s.run(['g++-11']+flags+units+[SRC/'update_check.cpp']+link+['-o',OUT/'update_check'],'compile-optimized-g1',timeout=900)
  if not REFERENCE:s.run(['g++-11']+gate['flags']+[oldsrc/n for n in ('update_store.cpp','deg_prune.cpp','dual_update.cpp','update_check.cpp')]+link+['-o',OUT/'reference_check'],'compile-reference-g1',timeout=900)
  else:
   assert json.loads((REFERENCE/'build-identity.json').read_text())['library']==s.sha(s.c.BUILD/'src/libpipeann.a')
   s.save(OUT/'reference-reuse.json',dict(source=str(REFERENCE),reason='unchanged frozen reference, inputs and library; only optimized batch lock scope changed'))
  for label,src,fs in [('reference',oldsrc,gate['flags']),('optimized',SRC,flags)]:
   if REFERENCE and label=='reference':continue
   s.run(['g++-11']+fs+[src/n for n in ('update_store.cpp','deg_prune.cpp','dual_update.cpp')]+[ROOT.parent/'update-perf-audit-2026-11-17/profile_driver.cpp']+link+['-o',OUT/(label+'-insert')],'compile-'+label+'-insert',timeout=900)
  s.run(['g++-11']+flags+[SRC/'atomic_degree_check.cpp','-pthread','-o',OUT/'atomic-check'],'compile-atomic')
  s.run([OUT/'atomic-check'],'atomic-directed')
  for test in ('optimistic_store_check','topology_interleave'):
   s.run(['g++-11']+flags+[SRC/'update_store.cpp',SRC/(test+'.cpp')]+link+['-o',OUT/test],'compile-'+test)
   d=OUT/(test+'-fixture');d.mkdir();s.run([OUT/test,d],test)
  parity=[]
  for mode in ('pq','exact'):
   digests=[]
   for kind in ('reference','optimized'):
    if REFERENCE and kind=='reference':
     digests.append(next(x['versions'][0] for x in json.loads((REFERENCE/'parity100.json').read_text()) if x['mode']==mode));continue
    r=fixture(kind+'-'+mode,OUT/'parity100');out=OUT/('parity100-'+kind+'-'+mode);out.mkdir()
    s.run([OUT/(kind+'-insert'),r,mode,1,0,'e1',100,out,1],'parity100',out)
    digests.append(dict(graph=s.sha(r/'graph.pages'),coordinates=s.sha(r/'coordinates.pages'),snapshot=s.sha(out/'normal-exit.snapshot')))
   assert digests[0]==digests[1],mode+' 100-insert bitwise parity failed'
   parity.append(dict(mode=mode,count=100,versions=digests))
  s.save(OUT/'parity100.json',parity)
  s.status('PARITY100_PASSED_RUNNING_G1')
  full=[]
  for mode in ('pq','exact'):
   digests=[]
   for kind,binary in [('reference','reference_check'),('optimized','update_check')]:
    if REFERENCE and kind=='reference':
     digests.append(next(x['versions'][0] for x in json.loads((REFERENCE/'g1-parity.json').read_text()) if x['mode']==mode));continue
    r=fixture(kind+'-'+mode,OUT/'g1')
    s.run([OUT/binary,r,mode,1000],'G1-'+kind+'-'+mode,timeout=3600)
    assert 'G1_PASS operations=1000' in (OUT/('G1-'+kind+'-'+mode+'.log')).read_text()
    digests.append(dict(graph=s.sha(r/'graph.pages'),coordinates=s.sha(r/'coordinates.pages'),snapshot=s.sha(r/'normal-exit.snapshot'),operations=s.sha(r/'operations.jsonl')))
   assert digests[0]==digests[1],mode+' full mixed-operation bitwise parity failed'
   full.append(dict(mode=mode,operations=1000,status='BITWISE_IDENTICAL',versions=digests))
  s.save(OUT/'g1-parity.json',full)
  s.save(OUT/'build-identity.json',dict(flags=flags,sources={p.name:s.sha(p) for p in SRC.iterdir() if p.suffix in ('.h','.cpp')},library=s.sha(s.c.BUILD/'src/libpipeann.a'),binary=s.sha(OUT/'update_check')))
  results=[]
  for name,sanitizer in [('asan','address'),('tsan','thread')]:
   d=OUT/name;d.mkdir()
   fs=[v for v in flags if v not in ('-O3','-O2','-static-libstdc++','-static-libgcc')]+['-O1','-g','-fno-omit-frame-pointer','-fsanitize='+sanitizer]
   support=s.c.SOURCE/'src/utils/linux_aligned_file_reader.cpp'
   s.run(['g++-11']+fs+units+[support,SRC/'g3_concurrent.cpp']+link+['-o',d/'concurrent'],'compile-'+name,timeout=1200)
   env=dict(ASAN_OPTIONS='halt_on_error=1:abort_on_error=1:detect_leaks=1',TSAN_OPTIONS='halt_on_error=1:exitcode=66')
   for test,cpp in [('atomic','atomic_degree_check.cpp'),('optimistic','optimistic_store_check.cpp'),('interleave','topology_interleave.cpp')]:
    source=[SRC/cpp] if test=='atomic' else [SRC/'update_store.cpp',support,SRC/cpp]
    s.run(['g++-11']+fs+source+link+['-o',d/test],'compile-'+name+'-'+test,timeout=900)
    fd=d/(test+'-fixture');fd.mkdir();args=[d/test] if test=='atomic' else [d/test,fd]
    s.run(args,'directed-'+name+'-'+test,extra=env)
   for mode in ('pq','exact'):
    r=fixture(mode,d)
    s.run([d/'concurrent',r,s.E/'backlink-ab-20261115/runtime/data-10k/base_img_emb.fvecs',s.E/'backlink-ab-20261115/runtime/data-10k/base_text_emb.fvecs',s.DATA/'query_img_emb.fvecs',s.DATA/'query_text_emb.fvecs',600,mode],name+'-'+mode+'-600s',extra=env,timeout=3600)
    result=json.loads((r/'concurrent-result.json').read_text());assert result['passed'] and result['update_operations']==1000
    results.append(dict(sanitizer=name,mode=mode,result=result,binary_sha256=s.sha(d/'concurrent')));s.save(OUT/'sanitizers.json',results)
  s.status('OPTIMIZED_G1_G3_PASSED_E1_NOT_RUN',sanitizers=4,awaiting_user_confirmation=True)
 except Exception as error:s.status('OPTIMIZATION_VALIDATION_FAILED',reason=str(error));raise
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':main()
