"""Deterministic training-only work frontiers and matched-recall reporting."""
import math

LEVELS=[40,80,120,160,240,400,800,1600]
BUDGETS=[10,20,40,80,160,320]
TARGETS=[.50,.70,.80,.85,.88,.90,.92,.94,.95,.96,.97,.98,.99,1.]
METHODS=['DEG_off','DEG_on','Or_ideal','F','M_equal','M_weighted']
RULES=['pages','compute','weighted_pages']

def allocations(k,weight):
    x=max(1,min(2*k-1,int(math.floor(2*k*weight+.5))))
    return x,2*k-x
def jobs(method,alpha,calibrated):
    out=[]
    for L in LEVELS:
        for k in BUDGETS:
            variants=[('equal',k,k)] if method=='M_equal' else [('alpha',)+allocations(k,alpha),('contribution',)+allocations(k,calibrated)] if method=='M_weighted' else [('single',k,0)]
            seen=set()
            for name,x,y in variants:
                if max(x,y)>L or (x,y) in seen:continue
                seen.add((x,y));out.append(dict(label=method+'-'+name+'-L'+str(L)+'-k'+str(k),method=method,L=L,kx=x,ky=y,weight=alpha,allocation=name))
    return out
def work(point):
    t=sum(g['topology_pages'] for g in point['graphs']);c=point['coordinate_pages']
    pq=sum(g['pq_space_evals'] for g in point['graphs'])
    # A PQ evaluation is 64 table additions; an exact space is 512 coordinates.
    compute=pq*64+point['reranked']*2*512
    return dict(pages=t+c,compute=compute,weighted_pages=24.8*t+5.2*c,topology_pages=t,coordinate_pages=c,pq=pq)
def frontiers(points):
    result={}
    for rule in RULES:
        chosen=[]
        for target in TARGETS:
            covering=[p for p in points if p['recall']>=target]
            if not covering:continue
            best=min(covering,key=lambda p:(work(p)[rule],work(p)['pages'],p['L'],p['kx']+p['ky'],p['label']))
            if best['label'] not in [p['label'] for p in chosen]:chosen.append(best)
        # Freeze order using training recall. Timing cannot influence this order.
        result[rule]=[p['label'] for p in sorted(chosen,key=lambda p:(p['recall'],work(p)[rule],p['label']))]
    return result
def crossing(points,target):
    for p in points:
        if p['recall']==target:return p,p,0.
    for a,b in zip(points,points[1:]):
        if a['recall']<target<b['recall']:return a,b,(target-a['recall'])/(b['recall']-a['recall'])
    return None
def matched(points,target):
    pair=crossing(points,target)
    if pair is None:return dict(covered=False,recall_range=[min((p['recall'] for p in points),default=None),max((p['recall'] for p in points),default=None)])
    a,b,w=pair;out=dict(covered=True,labels=[a['label'],b['label']],recall_bracket=[a['recall'],b['recall']],interpolation_weight=w)
    for key in ('latency_us','qps','reranked','coordinate_pages','coordinate_submitted_pages','coordinate_wait_us','candidates_before_union','candidates_after_union'):
        out[key]=(1-w)*a[key]+w*b[key]
    for key in ('expanded','pq_space_evals','topology_pages','topology_submitted_pages','topology_wait_us'):
        out[key]=(1-w)*sum(g[key] for g in a['graphs'])+w*sum(g[key] for g in b['graphs'])
    out['raw_p95_us']={p['label']:p['p95_us'] for p in (a,b)}
    return out
