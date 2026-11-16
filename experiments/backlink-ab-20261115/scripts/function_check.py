"""Exact serialization comparisons; no tolerance quietly replaces the gate."""
import gzip,json,struct
from pathlib import Path
import common as c
F=c.RUN/'function-v2'
FIX={1:'strict_degree_cap',2:'merge_last_interval',4:'zero_slope_and_rhs'}

def take(f,n):
    b=f.read(n)
    if len(b)!=n:raise RuntimeError('truncated function evidence')
    return b
def unpack(f,fmt):return struct.unpack(fmt,take(f,struct.calcsize(fmt)))
def edges(f):
    count=unpack(f,'<I')[0];raw=struct.pack('<I',count);values=[]
    if count>1000:raise RuntimeError('output degree bound')
    for _ in range(count):
        h=take(f,20);i,e,g,layer,n=struct.unpack('<IffII',h);r=take(f,n*8);raw+=h+r;values.append(dict(id=i,e=e,g=g,layer=layer,ranges=[struct.unpack_from('<ff',r,k*8) for k in range(n)],blob=h+r))
    return raw,values
def trace(path):
    with Path(path).open('rb') as f:
        if take(f,8)!=b'HVSREV2\0':raise RuntimeError('trace magic')
        while f.peek(1):
            ordinal,target,new,m=unpack(f,'<QIII');pool=[unpack(f,'<Iff') for _ in range(m)];f.seek(m*m*8,1);raw,value=edges(f)
            yield dict(ordinal=ordinal,target=target,new=new,pool=pool,raw=raw,edges=value)
def outputs(path):
    with Path(path).open('rb') as f:
        if take(f,8)!=b'HVSOUT2\0':raise RuntimeError('output magic')
        result=[]
        while f.peek(1):
            ordinal=unpack(f,'<Q')[0];raw,v=edges(f);result.append((ordinal,raw,v))
        return result
def ulp(a,b):
    if a==b:return 0
    if a<0 or b<0:return 2**32
    return abs(struct.unpack('<I',struct.pack('<f',a))[0]-struct.unpack('<I',struct.pack('<f',b))[0])
def explain(a,b,pool):
    if len(a)==len(b):
        max_ulp=0;possible=True
        for x,y in zip(a,b):
            if (x['id'],x['e'],x['g'],x['layer'],len(x['ranges']))!=(y['id'],y['e'],y['g'],y['layer'],len(y['ranges'])):possible=False;break
            for r,s in zip(x['ranges'],y['ranges']):max_ulp=max(max_ulp,ulp(r[0],s[0]),ulp(r[1],s[1]))
        if possible and max_ulp<=8:return dict(kind='endpoint_last_bits',max_ulp=max_ulp)
    tied=len({(e,g) for _,e,g in pool})<len(pool)
    if tied and sorted(x['blob'] for x in a)==sorted(x['blob'] for x in b):return dict(kind='tie_output_order_only',input_has_exact_two_distance_ties=True)
    return dict(kind='unexplained')
def clean(v):return [{k:x for k,x in row.items() if k!='blob'} for row in v]

def main():
    records=list(trace(F/'capture.bin'));b=outputs(F/'results/B.bin');author={m:outputs(F/'results'/('author-'+str(m)+'.bin')) for m in range(8)}
    if len(records)!=10000 or len(b)!=10000 or any(len(v)!=10000 for v in author.values()):raise RuntimeError('must compare10000 captured inputs, not a shorter prefix')
    reproduced=0;equal={m:0 for m in range(8)};influence={m:0 for m in (1,2,4)};minimal_counts={};unexplained=[];residuals=[];original_diff=0
    details=F/'results/per-input-attribution.jsonl.gz'
    with gzip.open(details,'wt',encoding='utf8') as f:
      for i,r in enumerate(records):
        ordinal,raw,bb=b[i]
        if ordinal!=r['ordinal'] or any(v[i][0]!=ordinal for v in author.values()):raise RuntimeError('record order mismatch')
        reproduced+=raw==r['raw']
        same=[]
        for m in range(8):
            if author[m][i][1]==raw:equal[m]+=1;same.append(m)
        original_diff+=author[0][i][1]!=raw
        for m in influence:influence[m]+=author[m][i][1]!=author[0][i][1]
        minimum=[m for m in same if not any(n!=m and n&m==n for n in same)]
        key='|'.join(str(x) for x in minimum) if minimum else 'none';minimal_counts[key]=minimal_counts.get(key,0)+1
        row=dict(record=i,ordinal=ordinal,target=r['target'],new=r['new'],pool_size=len(r['pool']),B_reproduces_capture=raw==r['raw'],matching_fix_masks=same,minimal_fix_masks=minimum)
        if 7 not in same:
            why=explain(bb,author[7][i][2],r['pool']);row['corrected_difference']=why
            diff=dict(row,B=clean(bb),author_fixed=clean(author[7][i][2]),input_pool=r['pool']);residuals.append(diff)
            if why['kind']=='unexplained':unexplained.append(i)
        f.write(json.dumps(row,allow_nan=False)+'\n')
    c.save(F/'validation/corrected-residuals.json',residuals)
    passed=reproduced==10000 and equal[7]>=9990 and not unexplained
    result=dict(status='PASS' if passed else 'STOP_AT_FUNCTION_GATE',cases=len(records),B_reproduces_actual_capture=reproduced,
        exact_matches={str(k):dict(count=v,fraction=v/10000) for k,v in equal.items()},original_diff_inputs=original_diff,
        single_fix_changes_from_original={FIX[m]:v for m,v in influence.items()},minimal_fix_set_counts=minimal_counts,fix_bits=FIX,
        corrected_residual_count=len(residuals),unexplained_input_ids=unexplained,remaining_explanations_file=str(F/'validation/corrected-residuals.json'),per_input_attribution=str(details),
        definition='complete ordered output bitwise; fixes overlap and interact; shared captured raw pair distances, not re-evaluated SIMD distances')
    c.save(F/'validation/function-gate.json',result)
    if not passed:raise RuntimeError('function gate failed; see '+str(F/'validation/function-gate.json'))
    return result
if __name__=='__main__':main()
