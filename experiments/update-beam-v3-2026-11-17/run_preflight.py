"""Compile first and compare actual original Engine on the same static graph."""
from pathlib import Path
import sys,json,fcntl,os
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'update-perf-audit-2026-11-17'))
import run_suite as s
OUT=ROOT/'runtime'/os.environ.get('HVS_PREFLIGHT_RUN','preflight-v2');SRC=ROOT/'implementation'
def main():
 OUT.mkdir(parents=True,exist_ok=False);s.OUT=OUT
 lock=(s.E/'budget-reorder-20261103/queue.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX)
 try:
  assert json.loads(Path('/mnt/pmem0/hvs-update-basic-2026-11-16/e1-degree-protected-v2/status.json').read_text())['stage']=='PAUSED_AFTER_EXACT_R1_USER_REQUEST'
  gate=json.loads((ROOT.parent/'update-io-v2-2026-11-17/runtime/validation-v2/build-identity.json').read_text())
  flags=['-I'+str(SRC)]+gate['flags'];link=[s.c.BUILD/'src/libpipeann.a','-L/mnt/nvme2/wz/disk/local/openblas/lib','-laio','-ljemalloc','-lopenblas','-pthread']
  s.run(['clang-format','-style=LLVM','-i']+[p for p in SRC.glob('*.cpp') if p.name!='deg_prune.cpp']+[SRC/n for n in ('dual_update.h','update_store.h','beam_search.h','profile_scope.h')]+[ROOT/'profile_driver.cpp',ROOT/'static_query.cpp'],'format-sources')
  units=[SRC/n for n in ('update_store.cpp','deg_prune.cpp','dual_update.cpp')]
  for name,driver,extra in [('check',SRC/'update_check.cpp',[]),('insert',ROOT/'profile_driver.cpp',[]),('profile',ROOT/'profile_driver.cpp',['-DHVS_PROFILE_PHASES']),('g2',SRC/'g2_insert.cpp',[]),('static',ROOT/'static_query.cpp',['-DHVS_ORIGINAL_QUERY_SOURCE="'+str(s.E/'multigraph-20261113/scripts/multigraph.cpp')+'"'])]:
   s.run(['g++-11']+flags+extra+units+[driver]+link+['-o',OUT/name],'compile-'+name,timeout=900)
  s.save(OUT/'identity.json',dict(base_commit='c61836b95be711daf552d243b93f59bb57aacbfe',flags=flags,sources={p.name:s.sha(p) for p in SRC.iterdir() if p.suffix in ('.cpp','.h')},library=s.sha(s.c.BUILD/'src/libpipeann.a'),original_query=s.sha(s.E/'multigraph-20261113/scripts/multigraph.cpp'),binaries={p.name:s.sha(p) for p in (OUT/n for n in ('check','insert','profile','g2','static'))}))
  s.run([OUT/'static',ROOT.parent/'update-io-v2-2026-11-17/runtime/baseline/hot/static-1800k',s.COLD/'gt',OUT/'static-query.jsonl'],'static-parity',timeout=1800)
  s.status('PREFLIGHT_STATIC_PARITY_PASSED_E1_NOT_RUN')
 except Exception as error:s.status('PREFLIGHT_FAILED',reason=str(error));raise
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':main()
