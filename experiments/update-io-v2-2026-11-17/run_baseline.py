"""Only frozen-executor diagnostics; E1 remains paused and is never launched."""
from pathlib import Path
import json,sys,shutil,fcntl
ROOT=Path(__file__).resolve().parent
E=ROOT.parent
sys.path.insert(0,str(E/'update-perf-audit-2026-11-17'))
import run_suite as s
from prepare_profile import prepare
OUT=ROOT/'runtime/baseline'

def main():
 OUT.mkdir(exist_ok=False)
 s.OUT=OUT
 lock=(E/'budget-reorder-20261103/queue.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX)
 try:
  assert json.loads(Path('/mnt/pmem0/hvs-update-basic-2026-11-16/e1-degree-protected-v2/status.json').read_text())['stage']=='PAUSED_AFTER_EXACT_R1_USER_REQUEST'
  (OUT/'pq-prefix').symlink_to(E/'update-perf-audit-2026-11-17/runtime/pq-prefix',target_is_directory=True)
  gate=json.loads((s.OLD/'runtime/g1-in-degree-protection-v2/build-identity.json').read_text())
  src=OUT/'held-source';prepare(s.OLD/'implementation',src,gate)
  p=src/'update_store.h';v=p.read_text();v=v.replace('uint64_t ledger_wait_ns = 0, ledger_hold_ns = 0;','uint64_t ledger_wait_ns = 0, ledger_hold_ns = 0;\n  uint64_t held_topology_reads = 0, held_coordinate_reads = 0;');p.write_text(v)
  p=src/'dual_update.cpp';v=p.read_text();v=v.replace('uint64_t hold_start = 0;','uint64_t hold_start = 0;\n  uint64_t reads_before[2] = {};');v=v.replace('hold_start = perf_now();','hold_start = perf_now();\n          reads_before[0] = stats.read_pages[0]; reads_before[1] = stats.read_pages[1];');v=v.replace('  if (ledger.owns_lock()) stats.ledger_hold_ns += perf_now() - hold_start;','  if (ledger.owns_lock()) { stats.ledger_hold_ns += perf_now() - hold_start;\n    stats.held_topology_reads += stats.read_pages[0]-reads_before[0];\n    stats.held_coordinate_reads += stats.read_pages[1]-reads_before[1]; }');p.write_text(v)
  driver=OUT/'profile_driver.cpp';v=(E/'update-perf-audit-2026-11-17/profile_driver.cpp').read_text();v=v.replace('  FIELD(ledger_hold_ns);','  FIELD(ledger_hold_ns);\n  FIELD(held_topology_reads); FIELD(held_coordinate_reads);');driver.write_text(v)
  link=[s.c.BUILD/'src/libpipeann.a','-L/mnt/nvme2/wz/disk/local/openblas/lib','-laio','-ljemalloc','-lopenblas','-pthread']
  flags=['-I'+str(src)]+gate['flags']+['-DHVS_PROFILE_PHASES']
  binary=OUT/'held_driver';s.run(['g++-11']+flags+[src/n for n in ('update_store.cpp','deg_prune.cpp','dual_update.cpp')]+[driver]+link+['-o',binary],'compile-held')
  rows=[]
  for mode in ('pq','exact'):
   fixture=s.fixture('held-'+mode);out=OUT/('held-'+mode);out.mkdir()
   s.run([binary,fixture,mode,4,4,'e1',1000,out,0],'insert1000',out)
   a=[json.loads(v) for v in (out/'inserts.jsonl').read_text().splitlines()];assert len(a)==1000
   rows.append(dict(mode=mode,insertions=1000,mean_held_topology_reads=sum(v['held_topology_reads'] for v in a)/1000.,mean_held_coordinate_reads=sum(v['held_coordinate_reads'] for v in a)/1000.,mean_hold_ms=sum(v['ledger_hold_ns'] for v in a)/1e9,mean_wait_ms=sum(v['ledger_wait_ns'] for v in a)/1e9))
   shutil.rmtree(str(fixture))
  s.save(OUT/'held-results.json',rows)
  static=s.fixture('static-1800k');prefix=OUT/'pq-prefix'
  (static/'manifest.txt').write_text('HVS_MULTI_V1 1800000 512 512 1\n%s\n%s/\nu %s %s - 0\n'%(static/'coordinates.pages',prefix,static/'index.meta',static/'graph.pages'))
  binary=OUT/'static-query';source=E/'multigraph-20261113/scripts/multigraph.cpp'
  flags=['-I'+str(s.OLD/'implementation')]+gate['flags']+['-DHVS_ORIGINAL_QUERY_SOURCE="'+str(source)+'"']
  s.run(['g++-11']+flags+[s.OLD/'implementation'/n for n in ('update_store.cpp','deg_prune.cpp','dual_update.cpp')]+[ROOT/'static_query.cpp']+link+['-o',binary],'compile-static')
  s.save(OUT/'identity.json',dict(original_query_source=s.sha(source),frozen_executor={n:s.sha(s.OLD/'implementation'/n) for n in ('update_store.cpp','dual_update.cpp')},held_binary=s.sha(OUT/'held_driver'),static_binary=s.sha(binary),fixture=str(static),topology=s.sha(static/'graph.pages'),original_parameters='DEG_off, PQ navigation, beam4, SIMD exact',executor_parameters='node-wise PQ traversal, double-accumulator exact; full rerank L on both'))
  s.run([binary,static,s.COLD/'gt',OUT/'static-query.jsonl'],'static-query',timeout=3600)
  s.status('BASELINE_COMPLETE')
 except Exception as error:s.status('BASELINE_FAILED',reason=str(error));raise
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':main()
