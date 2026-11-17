"""Small post-gate verification only: 1000-point microbench, never E1."""
from pathlib import Path
import fcntl,json,sys,shutil,re
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'update-perf-audit-2026-11-17'))
import run_suite as s
OUT=ROOT/'runtime/micro'
def main():
 OUT.mkdir(exist_ok=False);s.OUT=OUT
 lock=(s.E/'budget-reorder-20261103/queue.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX)
 try:
  validation=ROOT/'runtime/validation-v2'
  assert json.loads((validation/'status.json').read_text())['stage']=='OPTIMIZED_G1_G3_PASSED_E1_NOT_RUN'
  assert json.loads(Path('/mnt/pmem0/hvs-update-basic-2026-11-16/e1-degree-protected-v2/status.json').read_text())['stage']=='PAUSED_AFTER_EXACT_R1_USER_REQUEST'
  assert shutil.disk_usage(str(ROOT)).free>=16*2**30
  (OUT/'pq-prefix').symlink_to(ROOT.parent/'update-perf-audit-2026-11-17/runtime/pq-prefix',target_is_directory=True)
  rows=[]
  for rep in range(1,4):
   for mode in ('pq','exact'):
    name=mode+'-r'+str(rep);root=s.fixture(name);out=OUT/name;out.mkdir()
    s.run([validation/'optimized-insert',root,mode,4,0,'lean',1000,out,1],'micro1000',out)
    shutil.copyfile(root/'graph.pages',out/'graph.pages')
    rows.append(dict(mode=mode,rep=rep,result=json.loads((out/'result.json').read_text()),graph_sha256=s.sha(out/'graph.pages'),snapshot_sha256=s.sha(out/'normal-exit.snapshot')));s.save(OUT/'measurements.json',rows)
    assert json.loads((root/'owner.json').read_text())['owner']==str(s.ROOT);shutil.rmtree(str(root))
  source=ROOT/'static_query.cpp';text=source.read_text(encoding='utf-8')
  text,n=re.subn(r'for\s*\(\s*unsigned L\s*:\s*\{\s*18u,\s*24u,\s*48u,\s*96u,\s*192u,\s*384u\s*\}\s*\)','for (unsigned L : {96u})',text)
  assert n==1
  generated=OUT/'static-fixed.cpp';generated.write_text(text,encoding='utf-8')
  gate=json.loads((s.OLD/'runtime/g1-in-degree-protection-v2/build-identity.json').read_text());src=ROOT/'implementation'
  flags=['-I'+str(src)]+gate['flags']+['-DHVS_ORIGINAL_QUERY_SOURCE="'+str(s.E/'multigraph-20261113/scripts/multigraph.cpp')+'"']
  link=[s.c.BUILD/'src/libpipeann.a','-L/mnt/nvme2/wz/disk/local/openblas/lib','-laio','-ljemalloc','-lopenblas','-pthread']
  s.run(['g++-11']+flags+[src/n for n in ('update_store.cpp','deg_prune.cpp','dual_update.cpp')]+[generated]+link+['-o',OUT/'static-optimized'],'compile-static-optimized')
  static=ROOT/'runtime/baseline/hot/static-1800k'
  s.run([OUT/'static-optimized',static,s.COLD/'gt',OUT/'static-query.jsonl'],'static-optimized',timeout=1800)
  old=[json.loads(x) for x in (ROOT/'runtime/baseline/static-query.jsonl').read_text().splitlines()]
  now=[json.loads(x) for x in (OUT/'static-query.jsonl').read_text().splitlines()]
  assert len(now)==30
  for row in now:
   reference=next(x for x in old if x['path']==row['path'] and x['alpha']==row['alpha'] and x['L']==row['L'] and x['rep']==row['rep'])
   assert row['recall']==reference['recall'] and row['expanded']==reference['expanded'],'static search behavior changed'
  s.save(OUT/'identity.json',dict(binary=s.sha(validation/'optimized-insert'),static_binary=s.sha(OUT/'static-optimized'),static_source=s.sha(generated),source=s.sha(source),static_recall_and_expansion_equal=True,E1_run=False))
  s.status('MICRO_COMPLETE_E1_NOT_RUN',awaiting_user_confirmation=True)
 except Exception as error:s.status('MICRO_FAILED',reason=str(error));raise
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':main()
