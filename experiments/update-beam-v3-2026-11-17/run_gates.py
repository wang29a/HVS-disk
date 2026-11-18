"""G1 + fresh G2 incremental graphs in both modes + four sanitizer G3 runs.
Old E1 is never started. Unchanged G2 full graph / GT are identity-checked.
"""
from pathlib import Path
import sys,json,fcntl,shutil,re,os
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'update-perf-audit-2026-11-17'))
import run_suite as s
import run_g2 as g
OUT=ROOT/'runtime'/os.environ.get('HVS_GATES_RUN','gates');SRC=ROOT/'implementation'
PREFLIGHT=ROOT/'runtime'/os.environ.get('HVS_PREFLIGHT_RUN','preflight-v2')
def fixture(name,parent):
 parent.mkdir(parents=True,exist_ok=True);old=s.c.RUN;s.c.RUN=parent
 try:r=s.run_g1.fixture(name)
 finally:s.c.RUN=old
 for key,axis in [('img','x'),('text','y')]: (r/('query-'+axis+'.bin')).symlink_to(s.DATA/('query_'+key+'_emb.fvecs'))
 for cohort in ('report','tune'):shutil.copyfile(s.MG/(cohort+'-ids.txt'),r/(cohort+'-ids.txt'))
 return r

def check_sources(identity):
 for name,digest in identity['sources'].items():assert s.sha(SRC/name)==digest,name+' changed after compilation'
 for name,digest in identity['binaries'].items():assert s.sha(PREFLIGHT/name)==digest,name+' binary changed'
 assert s.sha(s.c.BUILD/'src/libpipeann.a')==identity['library']
 for name in ('deg_prune.cpp','deg_prune.h','pruning_semantics.h'):assert (SRC/name).read_bytes()==(ROOT.parent/'update-io-v2-2026-11-17/implementation'/name).read_bytes()

def zero_indegree(meta,topo):
 import struct,numpy as np
 with Path(meta).open('rb') as f:n=struct.unpack('<I',f.read(4))[0]
 raw=Path(topo).read_bytes();ind=np.zeros(n,dtype=np.uint32)
 for node in range(n):
  off=(node//49)*8192+(node%49)*164;degree=struct.unpack_from('<I',raw,off)[0];assert degree<=40
  for v in struct.unpack_from('<'+'I'*degree,raw,off+4):assert v<n and v!=node;ind[v]+=1
 return set(map(int,np.flatnonzero(ind==0)))

def g2():
 target=OUT/'g2';target.mkdir();g.OUT=target
 oldrun=s.OLD/'runtime/g2-entry-parity';olddata=Path('/mnt/nvme2/wz/hvs-update-basic-2026-11-16/g2-entry-parity')
 retained=json.loads((oldrun/'baseline-reuse-identity.json').read_text())['retained_artifacts']
 for p,digest in retained.items():assert s.sha(Path(p))==digest,'G2 immutable input changed '+p
 query=json.loads((oldrun/'baseline-query-identity.json').read_text());assert s.sha(g.MG/'bin/multigraph')==query['static_executor']
 for p in olddata.iterdir():
  if (p.is_file() and p.name.startswith(('source-ids','base-','initial-','gt-','alpha-','centers','xy.'))) or p.name in ('full','initial'):(target/p.name).symlink_to(p,target_is_directory=p.is_dir())
 s.save(OUT/'g2-baseline-identity.json',dict(retained=retained,full_results=s.sha(oldrun/'full-matched.json'),static_executor=query['static_executor'],reuse='unchanged full100k B, initial90k B, subset, coordinates and GT; fresh incremental graphs; quality counts deterministic'))
 full=json.loads((oldrun/'full-matched.json').read_text());s.save(OUT/'g2-full-matched.json',full)
 g.DEPTHS=[10,12,14]+g.DEPTHS
 old=s.c.RUN;s.c.RUN=OUT/'g2-evaluation';s.c.RUN.mkdir()
 def evaluate_run(args,label,timeout=7200,extra=None):s.run(args,label,folder=s.c.RUN,timeout=timeout,extra=extra)
 g.run=evaluate_run
 results=[]
 try:
  for mode in ('exact','pq'):
   root=g.fixture(mode);initial_zero=zero_indegree(root/'index.meta',root/'graph.pages')
   s.run([PREFLIGHT/'g2',root,mode],'G2-insert-'+mode,timeout=14400)
   final_zero=zero_indegree(root/'eval.meta',root/'eval.topo');assert final_zero<=initial_zero,'G2 introduced zero indegree'
   rows=g.evaluate(mode,root/'eval.meta',root/'eval.topo');s.save(OUT/('g2-'+mode+'-matched.json'),rows)
   for row,base in zip(rows,full):
    assert (row['alpha'],row['cohort'],row['repeat'])==(base['alpha'],base['cohort'],base['repeat'])
    inc=row['matched']['expanded']/base['matched']['expanded']-1
    results.append(dict(mode=mode,alpha=row['alpha'],cohort=row['cohort'],repeat=row['repeat'],baseline=base['matched'],incremental=row['matched'],increase=inc,passed=inc<=.15))
   s.save(OUT/'g2-comparisons.json',results)
   s.save(OUT/('g2-'+mode+'-identity.json'),dict(graph=s.sha(root/'eval.topo'),metadata=s.sha(root/'eval.meta'),initial_zero_indegree=len(initial_zero),final_zero_indegree=len(final_zero),insert_result=json.loads((root/'insert-result.json').read_text())))
 finally:s.c.RUN=old
 assert len(results)==24 and all(x['passed'] for x in results if x['cohort']=='report'),'G2 quality increase exceeds15%'

def main():
 OUT.mkdir(exist_ok=False);s.OUT=OUT
 lock=(s.E/'budget-reorder-20261103/queue.lock').open('a+');s.status('WAITING_GLOBAL_LOCK');fcntl.flock(lock,fcntl.LOCK_EX)
 try:
  assert json.loads((PREFLIGHT/'status.json').read_text())['stage']=='PREFLIGHT_STATIC_PARITY_PASSED_E1_NOT_RUN'
  identity=json.loads((PREFLIGHT/'identity.json').read_text());check_sources(identity)
  flags=identity['flags'];link=[s.c.BUILD/'src/libpipeann.a','-L/mnt/nvme2/wz/disk/local/openblas/lib','-laio','-ljemalloc','-lopenblas','-pthread'];units=[SRC/n for n in ('update_store.cpp','deg_prune.cpp','dual_update.cpp')]
  s.save(OUT/'build-identity.json',identity)
  for mode in ('exact','pq'):
   r=fixture(mode,OUT/'g1');s.run([PREFLIGHT/'check',r,mode,1000],'G1-'+mode,timeout=3600)
   assert 'G1_PASS operations=1000' in (OUT/('G1-'+mode+'.log')).read_text()
  s.save(OUT/'g1.json',dict(passed=True,modes=['exact','pq'],operations_each=1000,initial=10000,bitwise_gate=False))
  g2();s.status('G1_G2_BOTH_PASSED_RUNNING_G3')
  check_sources(identity);results=[]
  for name,sanitizer in [('asan','address'),('tsan','thread')]:
   d=OUT/name;d.mkdir()
   fs=[v for v in flags if not re.match(r'^-O(?:[0-9sg]|fast)$',v) and v not in ('-static-libstdc++','-static-libgcc')]+['-O1','-g','-fno-omit-frame-pointer','-fsanitize='+sanitizer]
   support=s.c.SOURCE/'src/utils/linux_aligned_file_reader.cpp'
   s.run(['g++-11']+fs+units+[support,SRC/'g3_concurrent.cpp']+link+['-o',d/'concurrent'],'compile-'+name,timeout=1200)
   env=dict(ASAN_OPTIONS='halt_on_error=1:abort_on_error=1:detect_leaks=1',TSAN_OPTIONS='halt_on_error=1:exitcode=66')
   for test,cpp in [('atomic','atomic_degree_check.cpp'),('optimistic','optimistic_store_check.cpp'),('interleave','topology_interleave.cpp')]:
    source=[SRC/cpp] if test=='atomic' else [SRC/'update_store.cpp',support,SRC/cpp]
    s.run(['g++-11']+fs+source+link+['-o',d/test],'compile-'+name+'-'+test,timeout=900)
    fd=d/(test+'-fixture');fd.mkdir();args=[d/test] if test=='atomic' else [d/test,fd]
    s.run(args,'directed-'+name+'-'+test,extra=env)
   for mode in ('exact','pq'):
    r=fixture(mode,d)
    s.run([d/'concurrent',r,s.E/'backlink-ab-20261115/runtime/data-10k/base_img_emb.fvecs',s.E/'backlink-ab-20261115/runtime/data-10k/base_text_emb.fvecs',s.DATA/'query_img_emb.fvecs',s.DATA/'query_text_emb.fvecs',600,mode],name+'-'+mode+'-600s',extra=env,timeout=5400)
    result=json.loads((r/'concurrent-result.json').read_text());assert result['passed'] and result['update_operations']==1000
    ops=[json.loads(x) for x in (r/'concurrent-operations.jsonl').read_text().splitlines()]
    results.append(dict(sanitizer=name,mode=mode,result=result,updates_within_query_window=sum(x['seconds']<=600 for x in ops),binary=s.sha(d/'concurrent')));s.save(OUT/'sanitizers.json',results)
  check_sources(identity);s.status('G1_G2_G3_BOTH_PASSED_E1_NOT_RUN',awaiting_user_confirmation=True)
 except Exception as error:s.status('GATES_FAILED_E1_NOT_RUN',reason=str(error));raise
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':main()
