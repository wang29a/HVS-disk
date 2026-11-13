#!/usr/bin/env python3
import json
from pathlib import Path
import shutil
import struct
import subprocess
import numpy as np
from run_all import ROOT,ENV,run,save,digest
from partition_order import convert

def main():
    folder=ROOT/'scratch/synthetic'
    if folder.exists():shutil.rmtree(str(folder))
    folder.mkdir()
    n,d1,d2,R,M=2503,1024,768,40,8
    width=4+R*(4+2*M);npp=8192//width
    meta=bytearray(8192);struct.pack_into('<IIIIIQI',meta,0,n,d1,d2,R,M,npp,1);struct.pack_into('<I',meta,32,0)
    (folder/'meta').write_bytes(meta)
    topo=np.zeros(((n+npp-1)//npp,8192),dtype=np.uint8);coord=np.zeros((n,8192),dtype=np.uint8)
    for i in range(n):
        rec=bytearray(width);degree=0 if i%17==0 else R;struct.pack_into('<I',rec,0,degree)
        for j in range(degree):
            struct.pack_into('<I',rec,4+j*(4+2*M),(i+j+1)%n)
            off=8+j*(4+2*M)
            intervals=[(0,10),(20,30),(40,50),(70,100)] if j==0 else [(0,100)]
            if j==1:intervals=[(20,20)]
            if j==2:intervals=[(100,100)]
            if j==3:intervals=[]
            for k,(a,b) in enumerate(intervals):rec[off+2*k:off+2*k+2]=bytes([a,b])
        topo[i//npp,(i%npp)*width:(i%npp+1)*width]=np.frombuffer(rec,dtype=np.uint8)
        coord[i,:4*(d1+d2)]=((np.arange(d1+d2,dtype=np.float32)+i)/1000).view(np.uint8)
    topo.tofile(str(folder/'topo'));coord.tofile(str(folder/'coord'))
    order=folder/'order';order.write_bytes(struct.pack('<Q',n)+np.arange(n-1,-1,-1,dtype='<u4').tobytes())
    source=[folder/'meta',folder/'topo',folder/'coord'];before=[digest(p) for p in source]
    run(['g++-11','-std=c++17','-O2','-fopenmp',ROOT/'scripts/test_layout_io.cpp','-laio','-o',ROOT/'bin/test-layout-io'],'build-layout-test')
    cases=0;io_cases=0
    for budget in [8,2,0]:
        chosen=source
        if budget!=8:
            chosen=[folder/('meta'+str(budget)),folder/('topo'+str(budget)),source[2]]
            run([ROOT/'bin/convert-budget']+source+chosen[:2]+[budget,folder/'conversion.json'],'unit-convert-'+str(budget))
        for reordered in [False,True]:
            env=dict(ENV)
            if reordered:env['LAYOUT_ORDER']=str(order)
            for layout in ['TXY','TX_Y','TY_X','T_XY','T_X_Y']:
                label='unit-b{}-o{}-{}'.format(budget,int(reordered),layout)
                target=folder/'layout';target.mkdir()
                common=[ROOT/'bin/layout-engine','build']+chosen+[target,layout,folder/'build.json']
                run(common,label+'-build',env)
                run([ROOT/'bin/test-layout-io']+chosen+[target,layout],label+'-io',env);io_cases+=1
                for mask in [1,2,4,6]:
                    for hot in ['uniform','hot']:
                        reads=[]
                        for coalesce in [0,1]:
                            e=dict(env,LAYOUT_PATCH_COALESCE=str(coalesce),LAYOUT_PATCH_BATCH='128')
                            result=folder/'patch.json'
                            run([ROOT/'bin/layout-engine','patch']+chosen+[target,layout,result,7,129,mask,hot],label+'-patch',e)
                            v=json.loads(result.read_text());assert v['modified_page_mismatches']==v['restored_page_mismatches']==0
                            assert v['batch_count']==2 and v['fdatasync_batch_operations']==128
                            reads.append(v['submitted_read_bytes']);cases+=1
                        assert reads[1]<=reads[0]
                shutil.rmtree(str(target))
    bad=folder/'bad-order';bad.write_bytes(struct.pack('<Q',n)+np.zeros(n,dtype='<u4').tobytes())
    (folder/'layout').mkdir()
    fail=subprocess.run([str(x) for x in [ROOT/'bin/layout-engine','build']+source+[folder/'layout','TXY',folder/'bad.json']],env=dict(ENV,LAYOUT_ORDER=str(bad)),stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    assert fail.returncode!=0
    part=folder/'partition'
    part.write_bytes(struct.pack('<QQQ',3,2,5)+struct.pack('<III',2,0,1)+struct.pack('<IIII',3,2,3,4)+struct.pack('<IIIII',0,0,1,1,1))
    convert(part,folder/'part-order',folder/'part-report.json')
    assert json.loads((folder/'part-report.json').read_text())['internal_holes_filled']==1
    assert [digest(p) for p in source]==before
    save(ROOT/'analysis/layout-unit-tests.json',dict(status='PASS',nodes=n,dimensions=[d1,d2],
        layouts=5,budgets=[8,2,0],orders=['identity','reverse'],io_cases=io_cases,patch_cases=cases,
        empty_degree=True,empty_ranges=True,alpha_endpoints=True,point_intervals=True,
        partial_pages=True,LRU_eviction=True,io_batch_over_128=True,patch_batch_tail=True,
        bad_permutation_rejected=True,partition_holes_validated=True,new_GT=False))
    shutil.rmtree(str(folder))

if __name__=='__main__':main()
