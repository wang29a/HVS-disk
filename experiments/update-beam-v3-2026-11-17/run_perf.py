"""Fresh same-device throughput, 1000-insert phases and native OdinANN reference."""
from pathlib import Path
import sys,json,fcntl,shutil,tarfile,hashlib
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'update-perf-audit-2026-11-17'))
import run_suite as s
from run_gates import fixture,check_sources,PREFLIGHT
OUT=ROOT/'runtime/perf';BIN=PREFLIGHT
def main():
 OUT.mkdir(exist_ok=False);s.OUT=OUT
 lock=(s.E/'budget-reorder-20261103/queue.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX)
 try:
  assert json.loads((ROOT/'runtime/gates/status.json').read_text())['stage']=='G1_G2_G3_BOTH_PASSED_E1_NOT_RUN'
  identity=json.loads((BIN/'identity.json').read_text());check_sources(identity)
  assert json.loads(Path('/mnt/pmem0/hvs-update-basic-2026-11-16/e1-degree-protected-v2/status.json').read_text())['stage']=='PAUSED_AFTER_EXACT_R1_USER_REQUEST'
  assert shutil.disk_usage(str(ROOT)).free>=14*2**30,'need14GiB including largest native shadow or dual-space fixture'
  (OUT/'pq-prefix').symlink_to(s.ROOT/'runtime/pq-prefix',target_is_directory=True)
  controls=[]
  for mode in ('exact','pq'):
   versions=[]
   for kind in ('insert','profile'):
    r=fixture(mode+'-'+kind,OUT/'controls');out=OUT/('control-'+mode+'-'+kind);out.mkdir()
    s.run([BIN/kind,r,mode,1,0,'e1',100,out,1],'observer-control',out)
    versions.append(dict(graph=s.sha(r/'graph.pages'),coordinates=s.sha(r/'coordinates.pages'),snapshot=s.sha(out/'normal-exit.snapshot')))
   assert versions[0]==versions[1],'phase observers changed graph/state '+mode
   controls.append(dict(mode=mode,insertions=100,bitwise_equal=True,versions=versions))
  s.save(OUT/'observer-controls.json',controls)
  rows=[]
  for rep in (1,2,3):
   for mode in (('exact','pq') if rep%2 else ('pq','exact')):
    for kind in ('insert','profile'):
     name=kind+'-'+mode+'-r'+str(rep);root=s.fixture(name);out=OUT/name;out.mkdir()
     s.run([BIN/kind,root,mode,4,0,'e1' if kind=='profile' else 'lean',1000,out,1 if kind=='insert' else 0],'insert1000',out)
     if kind=='insert':shutil.copyfile(root/'graph.pages',out/'graph.pages')
     row=dict(kind=kind,mode=mode,rep=rep,result=json.loads((out/'result.json').read_text()),graph_sha256=s.sha(root/'graph.pages'))
     if kind=='insert':row['snapshot_sha256']=s.sha(out/'normal-exit.snapshot')
     else:
      samples=[json.loads(x) for x in (out/'inserts.jsonl').read_text().splitlines()];assert len(samples)==1000
      for sample in samples:assert sum(sample['phase_ns'])==sample['insert_total_ns']
      row['phases']=dict(mean_phase_ms=[sum(x['phase_ns'][i] for x in samples)/1e9 for i in range(7)],mean_full_reprune_ms=sum(x['reprune_ns'] for x in samples)/1e9,mean_page_lock_wait_ms=sum(x['page_lock_wait_ns'] for x in samples)/1e9,mean_io_ms=sum(sum(x['read_io_ns_'+str(f)]+x['write_io_ns_'+str(f)] for f in (0,1)) for x in samples)/1e9,mean_read_batches=sum(x['read_batches'] for x in samples)/1000.,mean_write_batches=sum(x['write_batches'] for x in samples)/1000.,mean_coordinate_materialized_MiB=sum(x['coordinate_materialized_bytes'] for x in samples)/1000./2**20,mean_coordinate_cache_hits=sum(x['coordinate_cache_hits'] for x in samples)/1000.)
     rows.append(row);s.save(OUT/'measurements.json',rows)
     assert json.loads((root/'owner.json').read_text())['owner']==str(s.ROOT);shutil.rmtree(str(root))
  native=s.ROOT/'runtime/odin-driver';native_proof=json.loads((s.ROOT/'runtime/upstream-identity.json').read_text())
  assert s.sha(native)==native_proof['binary_sha256']
  assert s.sha(s.ROOT/'odin_driver.cpp')==native_proof['harness_sha256']
  for path,digest in native_proof['source_files'].items():assert s.sha(s.UPSTREAM/path)==digest
  archive=s.ROOT/'odinann-upstream-ae-0.1.1.tar'
  with tarfile.open(str(archive)) as tar:
   assert tar.pax_headers.get('comment')==s.UPSTREAM_SHA
   files={}
   for member in tar.getmembers():
    if not member.isfile():continue
    rel=Path(member.name);assert not rel.is_absolute() and '..' not in rel.parts
    digest=hashlib.sha256(tar.extractfile(member).read()).hexdigest()
    assert s.sha(s.UPSTREAM/rel)==digest,str(rel)
    files[str(rel)]=digest
   archive_proof=dict(archive_sha256=s.sha(archive),pax=tar.pax_headers,files=files,identical=True)
  s.save(OUT/'native-archive-proof.json',archive_proof)
  assert archive_proof['identical'] and archive_proof['pax']['comment']==s.UPSTREAM_SHA
  for path,digest in archive_proof['files'].items():assert s.sha(s.UPSTREAM/path)==digest
  old=s.ROOT/'runtime/odin-index';proof=json.loads((old/'identity.json').read_text())
  new=OUT/'odin-reference';new.mkdir()
  for name,digest in proof['files'].items():
   if name.startswith('base'):
    assert s.sha(old/name)==digest;(new/name).symlink_to(old/name)
  s.save(OUT/'upstream-identity.json',dict(source=native_proof,archive=archive_proof,index=proof,working_prefix=str(new/'base'),rebuild=False,remeasured=True))
  s.odin_measurements(native,new/'base')
  for name,digest in proof['files'].items():
   if name.startswith('base'):assert s.sha(old/name)==digest,'native initial index changed '+name
  check_sources(identity);s.save(OUT/'identity.json',dict(initial=1800000,insertions=1000,update_threads=4,query_threads=0,NUMA=3,CPU='72-75',repetitions=3,binaries=identity['binaries'],static_results=str(BIN/'static-query.jsonl'),static_results_sha256=s.sha(BIN/'static-query.jsonl'),E1_run=False,default_reprune='exact'))
  s.status('PERF_COMPLETE_E1_NOT_RUN',awaiting_user_confirmation=True)
 except Exception as error:s.status('PERF_FAILED_E1_NOT_RUN',reason=str(error));raise
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':main()
