"""Assemble and read-back check native TXY from the same immutable float graph."""
from pathlib import Path
import json,struct,sys
import numpy as np
import common as c
sys.path.insert(0,str(c.E/'paper-priorities-20261114/scripts'))
from no_sidecar import matrix
OUT=c.RUN/'native'
def write(name,source,data,pq,entries=None):
    root=OUT/'indexes'/name;root.mkdir(parents=True,exist_ok=True);stamp=root/'identity.json'
    entries=entries or source
    identity=dict(source=c.sha(source/'float-graph.bin'),entry=c.sha(entries/'full.meta'),data=[c.sha(data/n) for n in ('base_img_emb.fvecs','base_text_emb.fvecs')],packer=c.sha(c.MG_RUN/'bin/pack'))
    if stamp.exists():
        old=c.load(stamp)
        if old['identity']!=identity or old['native_sha256']!=c.sha(root/'single_index'):raise RuntimeError('native index identity changed')
        return root
    meta=bytearray((source/'full.meta').read_bytes());n,d1,d2,R,M=struct.unpack_from('<5I',meta)
    if R!=40 or not M:raise RuntimeError('expected Full native layout')
    width=4+R*(4+2*M);record=(d1+d2)*4+width;size=((record+8191)//8192)*8192
    if size!=8192:raise RuntimeError('this protocol expects one native record per page')
    c.capacity((n*8192+n*R*2*M+2**30),'corrected native coupled index '+name)
    log=c.run([c.MG_RUN/'bin/pack',source/'full.meta',source/'float-graph.bin',root/'compact.meta',root/'compact.topo',root/'ranges.bin',entries/'full.meta'],'native-writer-pack-'+name,cpus='80,81',timeout=1200)
    m=bytearray((root/'compact.meta').read_bytes());struct.pack_into('<I',m,16,M);struct.pack_into('<Q',m,20,0)
    x,y=(matrix(data/n) for n in ('base_img_emb.fvecs','base_text_emb.fvecs'));pages=(n+48)//49
    topology=np.memmap(str(root/'compact.topo'),dtype='u1',mode='r');words=np.ndarray((pages,49,41),dtype='<u4',buffer=topology,strides=(8192,164,4)).reshape(-1,41)[:n]
    ranges=np.memmap(str(root/'ranges.bin'),dtype='u1',mode='r',shape=(n,40,2*M))
    target=root/'single_index';part=root/'single_index.partial';start=c.time.time()
    with part.open('wb') as f:
        f.write(m)
        for first in range(0,n,1024):
            end=min(n,first+1024);b=end-first;block=np.zeros((b,8192),dtype='u1');xy=np.concatenate((x[first:end],y[first:end]),axis=1).astype('<f4')
            block[:,:xy.shape[1]*4]=xy.view('u1').reshape(b,-1)
            offset=(d1+d2)*4;block[:,offset:offset+4]=words[first:end,0].copy().view('u1').reshape(b,4)
            edges=np.zeros((b,40,4+2*M),dtype='u1');edges[:,:,:4]=words[first:end,1:].copy().view('u1').reshape(b,40,4);edges[:,:,4:]=ranges[first:end]
            block[:,offset+4:offset+width]=edges.reshape(b,-1);block.tofile(f)
        f.flush();c.os.fsync(f.fileno())
    part.replace(target)
    # Independent full read-back compares coordinates, IDs, fixed range bytes,
    # metadata and page padding, not only a file length/header.
    image=np.memmap(str(target),dtype='u1',mode='r',offset=8192,shape=(n,8192))
    for first in range(0,n,1024):
        end=min(n,first+1024);b=end-first;xy=np.concatenate((x[first:end],y[first:end]),axis=1).astype('<f4');off=(d1+d2)*4
        if not np.array_equal(image[first:end,:off],xy.view('u1').reshape(b,-1)):raise RuntimeError('native coordinates changed')
        if not np.array_equal(image[first:end,off:off+4],words[first:end,0].copy().view('u1').reshape(b,4)):raise RuntimeError('native degrees changed')
        edge=np.array(image[first:end,off+4:off+width]).reshape(b,40,4+2*M)
        if not np.array_equal(edge[:,:,:4],words[first:end,1:].copy().view('u1').reshape(b,40,4)) or not np.array_equal(edge[:,:,4:],ranges[first:end]):raise RuntimeError('native edge/range mismatch')
        if np.any(image[first:end,off+width:]):raise RuntimeError('native nonzero padding')
    (root/'_disk.index').write_bytes(m)
    for space in ('emb','loc'):
        for suffix in ('pq_compressed.bin','pq_pivots.bin'):
            src=pq/(space+'_'+suffix);dest=root/src.name
            if not dest.exists():dest.symlink_to(src)
            elif dest.resolve()!=src.resolve():raise RuntimeError('native PQ link changed')
    c.save(stamp,dict(identity=identity,status='PASS_FULL_READBACK',native_sha256=c.sha(target),nodes=n,dims=[d1,d2],R=R,M=M,record_bytes=record,aligned_record_bytes=size,index_bytes=target.stat().st_size,wall_seconds=c.time.time()-start))
    return root
