"""node3 read-only extraction: no distance computation, no graph scan.

Copy only 256 held-out query rows, existing GT rows and their base-vector rows.
The actual statistics run on the local workstation.
"""
import hashlib
import json
from pathlib import Path
import random
import struct
import sys

REMOTE=Path('/mnt/nvme3/wz/hvs-disk-build-20261027/experiments')
ALPHAS=[.1,.3,.5,.7,.9]


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def shape(path):
    with Path(path).open('rb') as f:n,d=struct.unpack('<II',f.read(8))
    if Path(path).stat().st_size!=8+n*d*4:raise RuntimeError('matrix size')
    return n,d


def rows(source,indices,dest):
    n,d=shape(source)
    with Path(source).open('rb') as f,Path(dest).open('wb') as out:
        out.write(struct.pack('<II',len(indices),d))
        for i in indices:
            if not 0<=i<n:raise RuntimeError('row ID')
            f.seek(8+i*d*4);data=f.read(d*4)
            if len(data)!=d*4:raise RuntimeError('row truncated')
            out.write(data)


def main(target):
    target=Path(target);target.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((REMOTE/'concurrency-alpha/analysis/manifest.json').read_text())['configurations']
    sources=[('laion',Path('/mnt/nvme2/wz/disk/dataset/Laion'),REMOTE/'laion-six-cell-20261111/workloads/alpha-0.5/analysis/tuning-split-v2.json'),
             ('openimage',Path('/mnt/nvme3/wz/dataset/Openimage_disk'),REMOTE/'metric-consistent-20261111/analysis/tuning-split-v2.json')]
    out=[]
    for dataset,data,split in sources:
        folder=target/dataset;folder.mkdir(exist_ok=True)
        report_ids=json.loads(split.read_text())['cohorts']['report']['query_ids']
        chosen=sorted(random.Random(20261112).sample(report_ids,256))
        work=[x for x in manifest if x['dataset']==dataset and (dataset!='laion' or x['fixed_alpha'] in ALPHAS)]
        neighbors=set();cases=[]
        for w in work:
            for key in ('alpha','gt'):
                expected=w['input_sha256'].get(w[key])
                if expected and digest(w[key])!=expected:raise RuntimeError('GT/alpha identity changed')
            gtname='gt-'+w['condition']+'.bin';aname='alpha-'+w['condition']+'.bin'
            rows(w['gt'],chosen,folder/gtname);rows(w['alpha'],chosen,folder/aname)
            with (folder/gtname).open('rb') as f:
                n,d=struct.unpack('<II',f.read(8));ids=struct.unpack('<'+'I'*(n*d),f.read())
            neighbors.update(ids)
            cases.append(dict(condition=w['condition'],fixed_alpha=w['fixed_alpha'],gt=gtname,alpha=aname,
                              gt_source=w['gt'],gt_sha256=digest(w['gt'])))
        base_ids=sorted(neighbors)
        for key in ('img','text'):
            rows(data/('query_'+key+'_emb.fvecs'),chosen,folder/('query_'+key+'.bin'))
            rows(data/('base_'+key+'_emb.fvecs'),base_ids,folder/('base_'+key+'.bin'))
        (folder/'base_ids.bin').write_bytes(struct.pack('<II',len(base_ids),1)+struct.pack('<'+'I'*len(base_ids),*base_ids))
        item=dict(dataset=dataset,source_directory=str(data),query_ids=chosen,cohort='256 of original 500 held-out report queries',
            sample_seed=20261112,unique_base_ids=len(base_ids),cases=cases,
            files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in folder.iterdir() if p.is_file()})
        (folder/'manifest.json').write_text(json.dumps(item,indent=2)+'\n');out.append(item)
    (target/'manifest.json').write_text(json.dumps(dict(datasets=out,compute='extraction only; no distances computed on node3'),indent=2)+'\n')
    print(json.dumps(dict(output=str(target),bytes=sum(p.stat().st_size for p in target.rglob('*') if p.is_file()),
                         datasets=[dict(name=x['dataset'],queries=len(x['query_ids']),unique_base_ids=x['unique_base_ids']) for x in out])))


if __name__=='__main__':main(sys.argv[1])
