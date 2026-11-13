"""Relocate offline float checkpoints and a static archive; verify every byte.

Canonical topology, coordinates, PQ and all benchmark physical files stay put.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent.parent
OLD=ROOT.parent/'metric-consistent-20261111'
DEST=Path('/mnt/nvme2/wz/hvs-metric-consistent-20261111/source-checkpoints')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def fsync_dir(path):
    fd=os.open(str(path),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    records=[]
    items=[(OLD/'scratch'/arm/'float-graph.bin',DEST/(arm+'-float-graph.bin')) for arm in ('dual','single')]
    # Already linked into immutable query executables; not read at query time.
    items.append((OLD/'bin/libpipeann-metric.a',DEST/'libpipeann-metric.a'))
    for source,target in items:
        if source.is_symlink():
            if source.resolve()!=target.resolve():raise RuntimeError('unexpected existing link')
            records.append(dict(source=str(source),target=str(target),bytes=target.stat().st_size,sha256=digest(target),already_relocated=True))
            continue
        n=source.stat().st_size;sha=digest(source)
        if shutil.disk_usage(str(DEST)).free<n+(1<<30):raise RuntimeError('nvme2 reserve insufficient')
        temp=target.with_suffix('.copying')
        with source.open('rb') as inp,temp.open('wb') as out:
            shutil.copyfileobj(inp,out,8<<20);out.flush();os.fsync(out.fileno())
        if temp.stat().st_size!=n or digest(temp)!=sha:raise RuntimeError('copied checkpoint differs')
        os.replace(str(temp),str(target));fsync_dir(DEST)
        link=source.with_suffix('.verified-link')
        if link.exists() or link.is_symlink():raise RuntimeError('stale staging link')
        link.symlink_to(target)
        os.replace(str(link),str(source));fsync_dir(source.parent)
        if source.stat().st_size!=n or digest(source)!=sha:raise RuntimeError('relocated path mismatch')
        records.append(dict(source=str(source),target=str(target),bytes=n,sha256=sha,already_relocated=False))
    report=dict(status='PASS',records=records,reason='free nvme3 temporary-layout space without changing query I/O placement',
                nvme3_free_bytes=shutil.disk_usage(str(ROOT)).free,
                unchanged='canonical SSD topology, coordinates, PQ, historical timings and their paths')
    (ROOT/'analysis/checkpoint-relocation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
