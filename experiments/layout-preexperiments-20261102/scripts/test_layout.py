#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import numpy as np
from run_all import ROOT,ENV,run,save,digest

def main():
    folder=ROOT/'scratch/synthetic';folder.mkdir(exist_ok=True)
    n,d1,d2,R,M=2503,1024,768,40,8
    width=4+R*(4+2*M);npp=8192//width
    meta=bytearray(8192);struct.pack_into('<IIIIIQI',meta,0,n,d1,d2,R,M,npp,1);struct.pack_into('<I',meta,32,0)
    (folder/'meta').write_bytes(meta)
    topo=np.zeros(((n+npp-1)//npp,8192),dtype=np.uint8)
    coord=np.zeros((n,8192),dtype=np.uint8)
    for i in range(n):
        rec=bytearray(width);degree=0 if i%17==0 else R;struct.pack_into('<I',rec,0,degree)
        for j in range(degree):
            struct.pack_into('<I',rec,4+j*(4+2*M),(i+j+1)%n)
            off=8+j*(4+2*M);rec[off:off+2]=bytes([0,100])
            if j==0:rec[off:off+4]=bytes([10,20,50,70])
        topo[i//npp,(i%npp)*width:(i%npp+1)*width]=np.frombuffer(rec,dtype=np.uint8)
        vec=(np.arange(d1+d2,dtype=np.float32)+i)/1000
        coord[i,:4*(d1+d2)]=vec.view(np.uint8)
    topo.tofile(str(folder/'topo'));coord.tofile(str(folder/'coord'))
    source=[folder/'meta',folder/'topo',folder/'coord'];before=[digest(p) for p in source]
    run(['clang-format','-i','-style=LLVM',str(ROOT/'scripts/test_layout_io.cpp')],'format-layout-test')
    run(['g++-11','-std=c++17','-O2','-fopenmp',str(ROOT/'scripts/test_layout_io.cpp'),'-laio','-o',str(ROOT/'bin/test-layout-io')],'build-layout-test')
    for layout in ['TXY','TX_Y','TY_X','T_XY','T_X_Y']:
        target=folder/layout;target.mkdir(exist_ok=True)
        base=[str(ROOT/'bin/layout-engine'),'build']+[str(p) for p in source]+[str(target),layout,str(folder/'build.json')]
        run(base,'unit-'+layout+'-build')
        run([str(ROOT/'bin/test-layout-io')]+[str(p) for p in source]+[str(target),layout],'unit-'+layout+'-io')
        for mask in [1,2,4,6]:
            for pattern in ['uniform','hot']:
                result=folder/'patch.json';cmd=[str(ROOT/'bin/layout-engine'),'patch']+[str(p) for p in source]+[str(target),layout,str(result),'7','129',str(mask),pattern]
                run(cmd,'unit-'+layout+'-patch-'+str(mask)+'-'+pattern)
                report=json.loads(result.read_text())
                if report['restored_page_mismatches'] or report['modified_page_mismatches']:raise RuntimeError('patch failed')
        shutil.rmtree(str(target))
    if [digest(p) for p in source]!=before:raise RuntimeError('unit originals modified')
    save(ROOT/'analysis/layout-unit-tests.json',dict(status='PASS',nodes=n,dimensions=[d1,d2],full_range_slots=M,
        layouts=5,concurrent_read_threads=4,patch_cases=40,partial_pages=True,LRU_eviction=True,
        io_batching_over_128=True,exact_field_identity=True,original_snapshots_restored=True,
        synthetic_storage_test_only=True,new_dataset_GT=False))
    shutil.rmtree(str(folder))

if __name__=='__main__':main()
