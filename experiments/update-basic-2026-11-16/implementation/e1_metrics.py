"""E1 checkpoint GT ceilings and trend definitions; ID recall stays primary."""


def id_ceiling(gt_rows, unreachable, query_ids):
    query_ids=list(query_ids)
    if not query_ids:raise ValueError('empty cohort')
    unreachable=set(map(int,unreachable));missing=0;affected=0
    for q in query_ids:
        row=list(map(int,gt_rows[q][:10]))
        if len(row)!=10 or len(set(row))!=10:raise ValueError('GT must contain 10 distinct IDs')
        count=sum(v in unreachable for v in row);missing+=count;affected+=bool(count)
    denominator=10*len(query_ids)
    return dict(missing_gt_entries=missing,gt_entries=denominator,affected_queries=affected,
                id_recall_ceiling=1-missing/denominator,theoretical_ceiling_loss_pp=100*missing/denominator)


def trend(points,reference_expanded):
    if reference_expanded<=0:raise ValueError('invalid reference')
    points=sorted(points,key=lambda p:p['inserted'])
    if len({p['inserted'] for p in points})!=len(points):raise ValueError('duplicate checkpoints')
    values=[p['expanded']/reference_expanded-1 for p in points]
    deltas=[b-a for a,b in zip(values,values[1:])]
    return dict(inserted=[p['inserted'] for p in points],expanded=[p['expanded'] for p in points],
                increase_vs_R1=values,step_change_pp=[100*d for d in deltas],
                strictly_increasing_all_checkpoints=len(deltas)>=3 and all(d>0 for d in deltas),
                nondecreasing_all_checkpoints=len(deltas)>=3 and all(d>=0 for d in deltas) and any(d>0 for d in deltas),
                net_change_pp=100*(values[-1]-values[0]) if values else None,
                scope='fixed 2M full B denominator; each checkpoint uses its own live-set GT; prefix size changes, so this is descriptive, not an isolated causal effect')
