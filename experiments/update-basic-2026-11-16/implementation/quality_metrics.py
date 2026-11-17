"""Recall definitions kept separate; none silently replaces an existing gate."""


def ceilings(gt_ids, unreachable_ids, strict_total, strict_unreachable,
             boundary_total, boundary_unreachable, k=10):
    """Attainable hit fractions when only reachable distinct IDs may return.

    strict/boundary use distances below/equal to the exhaustive kth distance.
    The threshold score is deliberately distinguished from rank-slot recall:
    extra boundary ties cannot replace missing strictly closer points in the
    latter. Exact equality is the default; a tolerance is a separate analysis.
    """
    if len(gt_ids) < k or len(set(gt_ids[:k])) != k:
        raise ValueError('GT must contain K distinct IDs')
    if not (0 <= strict_unreachable <= strict_total < k):
        raise ValueError('strict rank counts')
    if not (0 <= boundary_unreachable <= boundary_total and
            strict_total + boundary_total >= k):
        raise ValueError('boundary rank counts')
    rs = strict_total - strict_unreachable
    rb = boundary_total - boundary_unreachable
    id_missing = sum(v in unreachable_ids for v in gt_ids[:k])
    return dict(id_missing=id_missing, id_ceiling=(k-id_missing)/k,
                distance_threshold_ceiling=min(k,rs+rb)/k,
                rank_tie_ceiling=(rs+min(k-strict_total,rb))/k)


def prediction_scores(gt_ids, returned_ids, strict_ids, boundary_ids, k=10):
    if len(set(gt_ids[:k])) != k:
        raise ValueError('GT duplicates')
    values = list(returned_ids[:k])
    if len(set(values)) != len(values):
        raise ValueError('duplicate returned IDs must not inflate recall')
    strict, boundary = set(strict_ids), set(boundary_ids)
    if strict & boundary or len(strict) >= k or len(strict)+len(boundary) < k:
        raise ValueError('invalid rank partition')
    returned = set(values)
    a, b = len(returned & strict), len(returned & boundary)
    return dict(recall_id=len(returned & set(gt_ids[:k]))/k,
                distance_threshold_acceptance=(a+b)/k,
                rank_tie_recall=(a+min(k-len(strict),b))/k)


def matched_work(points, target=.95, metric='recall_id'):
    """First adjacent measured crossing in increasing search depth; no extrapolation."""
    ordered = sorted(points,key=lambda p:p['L'])
    for p in ordered:
        if p[metric] == target:
            return dict(covered=True,labels=[p['L'],p['L']],recall_bracket=[target,target],
                        interpolation_weight=0.,expanded=p['expanded'])
    for a,b in zip(ordered,ordered[1:]):
        if a[metric] < target < b[metric]:
            t=(target-a[metric])/(b[metric]-a[metric])
            return dict(covered=True,labels=[a['L'],b['L']],
                        recall_bracket=[a[metric],b[metric]],interpolation_weight=t,
                        expanded=a['expanded']+t*(b['expanded']-a['expanded']))
    return dict(covered=False,reason='95% is not inside an adjacent measured bracket',
                measured_recall_range=[min((p[metric] for p in ordered),default=None),
                                       max((p[metric] for p in ordered),default=None)])
