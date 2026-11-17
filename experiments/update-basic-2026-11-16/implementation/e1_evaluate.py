"""Common frozen executor for E1 DRAM and SSD, annotated ID-recall ceilings."""
from pathlib import Path
import json
import struct
import numpy as np
import run_g0 as c
from gt_unreachable import matrix
from quality_metrics import matched_work
from e1_metrics import id_ceiling

WEIGHTS=[.1,.3,.5,.7,.9]
DEPTHS=[24,32,64,96,128,192,256,384,512,768,1024,1536,2048]


def save(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')


def snapshot_codes(snapshot,out,expected_codes,online_codes=None,online_first=1800000):
    """Parse actual committed PQ codes, preserve internal/tag mapping for GT."""
    snapshot=Path(snapshot);out=Path(out)
    with snapshot.open('rb') as f:
        assert f.read(8)==b'HVSUPD1\0'
        initial,n,capacity,d1,d2,c1,c2=struct.unpack('<7I',f.read(28))
    words=(capacity+63)//64
    dtype=np.dtype([('tag','<u8'),('merged','u1'),('x','u1',(c1,)),('y','u1',(c2,))])
    records=np.memmap(str(snapshot),dtype=dtype,mode='r',offset=36+16*words,shape=(n,))
    tags=np.asarray(records['tag'],dtype='<u4');assert len(np.unique(tags))==n and tags.max()==n-1
    assert np.all(records['merged']==0)
    audit={}
    for key,space in [('x','emb'),('y','loc')]:
        codes=np.ascontiguousarray(records[key]);reference=matrix(expected_codes/(space+'_pq_compressed.bin'),'u1')
        frozen=np.asarray(reference[tags]);different=codes!=frozen
        rr,cc=np.where(different)
        audit[space]=dict(different_from_bulk_bytes=int(len(rr)),different_from_bulk_nodes=int(len(np.unique(rr))),
            details=[dict(internal_id=int(i),tag=int(tags[i]),chunk=int(j),actual=int(codes[i,j]),bulk=int(frozen[i,j])) for i,j in zip(rr[:100],cc[:100])])
        if online_codes is None:
            assert not len(rr),space+' updated PQ codes differ from frozen codebook encoding'
        else:
            old=tags<online_first;new=~old
            assert np.array_equal(codes[old],frozen[old]),space+' initial PQ codes changed'
            scalar=matrix(Path(online_codes)/(space+'_pq_compressed.bin'),'u1')
            assert np.all(tags[new]-online_first<len(scalar))
            assert np.array_equal(codes[new],np.asarray(scalar[tags[new]-online_first])),space+' inserted PQ codes differ from independently reencoded upstream scalar method'
            audit[space]['all_inserted_equal_online_scalar']=True
            audit[space]['initial_codes_unchanged']=True
        with (out/(space+'_pq_compressed.bin')).open('wb') as f:f.write(struct.pack('<II',*codes.shape));codes.tofile(f)
    save(out/'pq-code-audit.json',dict(policy='initial codes must equal frozen bulk; inserted codes must equal upstream scalar encoder; actual codes are measured and never replaced',
        scalar_reference=str(online_codes),spaces=audit))
    inverse=np.empty(n,dtype='<u4');inverse[tags]=np.arange(n,dtype='<u4')
    return tags,inverse


def collect(out,n,meta,topology,coordinates,pq_prefix,gt_directory,unreachable_tags,tags,run,executor,data,split,report_repeats=3):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    _,inverse=(None,None)
    if tags is not None:
        inverse=np.empty(n,dtype='<u4');inverse[tags]=np.arange(n,dtype='<u4')
    manifest=out/'manifest.txt'
    manifest.write_text('HVS_MULTI_V1 %d 512 512 1\n%s\n%s/\nu %s %s - 0\n'%(n,coordinates,pq_prefix,meta,topology))
    with Path(meta).open('rb') as f:f.seek(28);entry_count=struct.unpack('<I',f.read(4))[0]
    depths=sorted(set([max(10,entry_count)]+[v for v in DEPTHS if v>=entry_count]))
    results=[]
    for a in WEIGHTS:
        canonical=matrix(gt_directory/('gt-%d-%s.bin'%(n,a)))
        truth=out/('gt-internal-'+str(a)+'.bin')
        gt=np.asarray(canonical,dtype='<u4') if inverse is None else np.asarray(inverse[canonical],dtype='<u4')
        with truth.open('wb') as f:f.write(struct.pack('<II',*gt.shape));gt.tofile(f)
        annotations={}
        for cohort in ('tune','report'):
            ids=[int(v) for v in (split/(cohort+'-ids.txt')).read_text().split()]
            annotations[cohort]=id_ceiling(canonical,unreachable_tags,ids)
        save(out/('ceiling-'+str(a)+'.json'),annotations)
        if any(v['id_recall_ceiling']<.95 for v in annotations.values()):
            raise RuntimeError('95% target exceeds corresponding GT ID ceiling at n=%d alpha=%s'%(n,a))
        for medium,nav in [('dram','exact'),('ssd','pq')]:
            rows=[]
            def measure(cohort,rep,budgets,segment):
                label='%s-%s-%s-%d-segment%d'%(medium,a,cohort,rep,segment)
                jobs=out/('jobs-'+label+'.txt');p=out/(label+'-raw.json')
                jobs.write_text(''.join('L%d DEG_off %d %d 0 %s\n'%(L,L,10 if nav=='exact' else min(L,512),a) for L in budgets))
                run([executor,manifest,data/'query_img_emb.fvecs',data/'query_text_emb.fvecs',gt_directory/('alpha-'+str(a)+'.bin'),truth,
                     split/(cohort+'-ids.txt'),p,medium,4,0,nav,jobs,'-',1],label,timeout=14400,out=out)
                return json.loads(p.read_text())
            def match_points(raw):
                return matched_work([dict(L=v['L'],recall_id=v['recall'],expanded=sum(g['expanded'] for g in v['graphs'])) for v in raw])
            tune_budget=depths[:4];tune_raw=measure('tune',0,tune_budget,0);segment=1
            while not match_points(tune_raw)['covered'] and len(tune_budget)<len(depths):
                L=depths[len(tune_budget)];tune_budget.append(L);tune_raw.extend(measure('tune',0,[L],segment));segment+=1
            if not match_points(tune_raw)['covered']:raise RuntimeError('tune95% measured bracket absent '+str(out))
            report_budget=depths[:min(len(depths),len(tune_budget)+1)]
            save(out/('sampling-policy-'+medium+'-'+str(a)+'.json'),dict(axis=depths,tune=tune_budget,report_initial=report_budget,
                rule='tune first four valid depths, extend upward only to bracket95; report uses these plus one higher depth; if a held-out report curve still lacks coverage, append the next frozen-axis depth; no extrapolation or report-based optimum selection',
                warmup='none; one fixed500-query batch per point; DRAM preload outside query timing; SSD O_DIRECT and per-query private buffers'))
            for cohort,repeats in [('tune',1),('report',report_repeats)]:
                for rep in range(repeats):
                    label='%s-%s-%s-%d'%(medium,a,cohort,rep);p=out/(label+'-raw.json')
                    raw=list(tune_raw) if cohort=='tune' else measure(cohort,rep,report_budget,0)
                    used=list(tune_budget if cohort=='tune' else report_budget);segment=1
                    while not match_points(raw)['covered'] and len(used)<len(depths):
                        L=depths[len(used)];used.append(L);raw.extend(measure(cohort,rep,[L],segment));segment+=1
                    raw.sort(key=lambda v:v['L']);save(p,raw)
                    save(out/(label+'-annotated.json'),[dict(v,GT_theoretical_limit=annotations[cohort]) for v in raw])
                    match=match_points(raw)
                    if not match['covered']:raise RuntimeError('95% measured bracket absent '+str(p))
                    low,high=match['labels'];left=next(v for v in raw if v['L']==low);right=next(v for v in raw if v['L']==high);t=match['interpolation_weight']
                    interpolated={key:left[key]+t*(right[key]-left[key]) for key in ['latency_us','qps','coordinate_pages','coordinate_submitted_pages','reranked']}
                    # Throughput at a virtual matched point follows latency at
                    # fixed concurrency; retain direct QPS interpolation separately.
                    match.update(interpolated=interpolated,search_depth=low+t*(high-low),raw_p95_us={str(low):left['p95_us'],str(high):right['p95_us']})
                    rows.append(dict(alpha=a,medium=medium,navigation=nav,cohort=cohort,repeat=rep,n=n,matched=match,GT_theoretical_limit=annotations[cohort]))
                    save(out/'matched-partial.json',rows)
            results.extend(rows)
    save(out/'matched.json',results)
    return results
