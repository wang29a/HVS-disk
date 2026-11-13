"""Explain changes in GT discovery by replaying the observed expansion edges."""
from functools import lru_cache
from pathlib import Path
import struct

import followup as common
import numpy as np

base, monitor = common.base, common.monitor
PARENT = Path(__file__).resolve().parent.parent
ROOT = PARENT/'edge-audit'
LEVEL = 60


def traces(path):
    words = np.fromfile(str(path),dtype='<u4')
    if tuple(words[:2])!=(0x48565331,1000):
        raise RuntimeError('bad trace header')
    offset = 2
    for _ in range(1000):
        batches = int(words[offset])
        offset += 1
        frontier = []
        for _ in range(batches):
            count = int(words[offset])
            offset += 1
            frontier.extend(map(int,words[offset:offset+count]))
            offset += count
        count = int(words[offset])
        offset += 1
        candidates = set(map(int,words[offset:offset+count]))
        offset += count
        yield frontier,candidates
    if offset!=len(words):
        raise RuntimeError('bad trace length')


def replay(c, truth):
    with Path(c['meta']).open('rb') as stream:
        meta = stream.read(8192)
    n,_,_,degree,budget,npp,count = struct.unpack_from('<IIIIIQI',meta)
    if budget!=2 or not count:
        raise RuntimeError('expected fixed M=2 endpoint topology')
    width = 4+degree*8
    entries = set(struct.unpack_from('<{}I'.format(count),meta,32))
    topo = np.memmap(c['topo'],mode='r',dtype=np.uint8)
    pair = struct.Struct('<I4b')

    @lru_cache(maxsize=65536)
    def neighbors(node):
        offset = node//npp*8192 + node%npp*width
        nn, = struct.unpack_from('<I',topo,offset)
        if node>=n or nn>degree:
            raise RuntimeError('bad node in trace')
        result = []
        for i in range(nn):
            dst,lo1,hi1,lo2,hi2 = pair.unpack_from(topo,offset+4+i*8)
            if dst>=n:
                raise RuntimeError('bad destination in trace')
            active = False
            for lo,hi in ((lo1,hi1),(lo2,hi2)):
                if not (0<=lo<=100 and 0<=hi<=100) or (lo,hi)==(0,0):
                    break
                if lo<=100<=hi:
                    active = True
                    break
            result.append((dst,active))
        return result

    totals = dict(filter_discovered=0,bypass_discovered=0,filter_candidates=0,bypass_candidates=0,
                  added_discoveries=0,lost_discoveries=0,added_with_direct_inactive_edge_from_filtered_frontier=0,
                  added_first_admitted_by_inactive_edge=0,added_without_direct_inactive_edge_from_filtered_frontier=0)
    active_trace = traces(ROOT/'analysis'/('filter.L'+str(LEVEL)))
    bypass_trace = traces(ROOT/'analysis'/('bypass.L'+str(LEVEL)))
    for gt, (expanded_on,candidates_on), (expanded_off,candidates_off) in zip(truth,active_trace,bypass_trace):
        target = set(map(int,gt))
        seen_on, seen_off = entries.copy(), entries.copy()
        blocked_targets, first_inactive = set(),set()
        for node in expanded_on:
            for dst,active in neighbors(node):
                if active:
                    seen_on.add(dst)
                elif dst in target:
                    blocked_targets.add(dst)
        for node in expanded_off:
            for dst,active in neighbors(node):
                if dst in target and not active and dst not in seen_off:
                    first_inactive.add(dst)
                seen_off.add(dst)
        discovered_on, discovered_off = target & seen_on, target & seen_off
        if not candidates_on<=seen_on or not candidates_off<=seen_off:
            raise RuntimeError('trace replay failed to rediscover a rerank candidate')
        added = discovered_off-discovered_on
        direct = added & blocked_targets
        totals['filter_discovered'] += len(discovered_on)
        totals['bypass_discovered'] += len(discovered_off)
        totals['filter_candidates'] += len(target & candidates_on)
        totals['bypass_candidates'] += len(target & candidates_off)
        totals['added_discoveries'] += len(added)
        totals['lost_discoveries'] += len(discovered_on-discovered_off)
        totals['added_with_direct_inactive_edge_from_filtered_frontier'] += len(direct)
        totals['added_first_admitted_by_inactive_edge'] += len(added & first_inactive)
        totals['added_without_direct_inactive_edge_from_filtered_frontier'] += len(added-direct)
    return totals


def run(cpus):
    for directory in ('analysis','results','logs'):
        (ROOT/directory).mkdir(parents=True,exist_ok=True)
    saved = common.ROOT,base.ROOT,monitor.ROOT,monitor.CPUS
    common.ROOT = base.ROOT = monitor.ROOT = ROOT
    monitor.CPUS = cpus
    try:
        c = next(c for c in base.read(monitor.OLD/'analysis/inputs.json') if c['name']=='openimage')
        monitor.verify_inputs(c)
        c = base.prepare_budget(c,'2')
        folder = PARENT/'scratch/openimage-m2-x'
        truth = common.ARCHIVE/'analysis/x-top10-ids.bin'
        points = {}
        archived = base.read(PARENT/'analysis/diagnostics.json')
        for v in ('filter','bypass'):
            spec = c,folder,'S_T_X',c['prefix']+'/',truth,True,v=='bypass'
            points[v] = common.run_search(spec,[LEVEL],v,diagnostic=True,trace=True)[0]
            base.compare_logical([points[v]],{p['L']:p for p in archived[v]},v+' edge audit')
        totals = replay(c,np.fromfile(str(truth),dtype='<u4',offset=8).reshape(1000,10))
        for v in points:
            for suffix in ('discovered','candidates'):
                if abs(totals[v+'_'+suffix]/100-points[v]['gt_'+suffix+'_pct'])>1e-8:
                    raise RuntimeError('replayed discovery counts disagree with search')
        base.save(ROOT/'analysis/edge-audit.json',dict(status='PASS',L=LEVEL,queries=1000,
                  truth_slots=10000,totals=totals))
    finally:
        common.ROOT,base.ROOT,monitor.ROOT,monitor.CPUS = saved
