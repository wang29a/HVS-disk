"""Strict raw float-graph validation and independent node/interval comparison."""
import json,struct,math,gzip
from pathlib import Path

def read(path):
    data=Path(path).read_bytes();pos=0
    def get(fmt):
        nonlocal pos
        size=struct.calcsize(fmt)
        if pos+size>len(data):raise RuntimeError('truncated float graph')
        values=struct.unpack_from(fmt,data,pos);pos+=size;return values
    n,d1,d2=get('<QII');graph=[];issues=[];edges=0;hist={}
    for u in range(n):
        nn=get('<I')[0];row=[];seen=set();hist[nn]=hist.get(nn,0)+1
        if nn>40:issues.append(dict(node=u,type='degree',degree=nn))
        for k in range(nn):
            nb,nr=get('<II');ranges=[get('<ff') for _ in range(nr)];edges+=1
            if nb>=n or nb==u or nb in seen:issues.append(dict(node=u,neighbor=nb,type='ID/self/duplicate'))
            seen.add(nb)
            if not nr:issues.append(dict(node=u,neighbor=nb,type='empty_interval_list'))
            prev=-1.
            for lo,hi in ranges:
                if not math.isfinite(lo) or not math.isfinite(hi) or lo<0 or hi>1 or lo>=hi or lo<=prev:issues.append(dict(node=u,neighbor=nb,type='range_bounds_order_merge',lo=lo,hi=hi))
                prev=hi
            row.append(dict(id=nb,ranges=ranges))
        graph.append(row)
    if pos!=len(data):raise RuntimeError('trailing float graph bytes')
    result=dict(nodes=n,dims=[d1,d2],edges=edges,mean_degree=edges/n,degree_histogram=hist,issues=issues,status='PASS' if not issues else 'FAIL')
    return graph,result

def compare(first,second,output):
    if len(first)!=len(second):raise RuntimeError('graph node count differs')
    same=0;interval_same=0;different=0;examples=[];added=removed=0
    with gzip.open(str(output),'wt') as out:
        for u,(a,b) in enumerate(zip(first,second)):
            x={e['id']:e['ranges'] for e in a};y={e['id']:e['ranges'] for e in b};sets_equal=set(x)==set(y);same+=sets_equal
            common=set(x)&set(y);changed=[v for v in common if x[v]!=y[v]];interval_same+=sets_equal and not changed
            if not sets_equal or changed:
                row=dict(node=u,only_B=[e for e in a if e['id'] not in y],only_author=[e for e in b if e['id'] not in x],changed_ranges=[dict(id=v,B=x[v],author=y[v]) for v in sorted(changed)])
                out.write(json.dumps(row)+'\n');different+=1;added+=len(set(x)-set(y));removed+=len(set(y)-set(x))
                if len(examples)<20:examples.append(row)
    return dict(nodes=len(first),neighbor_sets_equal=same,neighbor_sets_equal_fraction=same/len(first),neighbor_and_intervals_equal=interval_same,different_nodes=different,B_only_edges=added,author_only_edges=removed,examples=examples,all_differences=str(output),explanation_status='requires first-divergence/float/tie evidence; not automatically waived')
