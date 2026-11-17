# -*- coding: utf-8 -*-
"""Generate frozen-executor diagnostics from completed raw measurements."""
from pathlib import Path
import json,statistics as st
ROOT=Path(__file__).resolve().parent
R=ROOT/'runtime'
def matched(rows,target=.95):
 rows=sorted(rows,key=lambda x:x['recall'])
 for lo,hi in zip(rows,rows[1:]):
  if lo['recall']<=target<=hi['recall'] and hi['recall']>lo['recall']:
   t=(target-lo['recall'])/(hi['recall']-lo['recall'])
   return dict(latency_us=lo['mean_us']+t*(hi['mean_us']-lo['mean_us']),bracket=[lo['L'],hi['L']],recall_bracket=[lo['recall'],hi['recall']])
 raise RuntimeError('95% target is not in a measured bracket')
def main():
 data=R/'baseline' if (R/'baseline/held-results.json').exists() else R
 held=json.loads((data/'held-results.json').read_text())
 rows=[json.loads(x) for x in (data/'static-query.jsonl').read_text().splitlines()]
 assert len(rows)==180
 lines=['# 冻结原版：锁与静态查询诊断','', 'E1的两种首轮已归档，后续重复与消融仍暂停。首轮度数、满度重剪边数及相对检查点0的五权重展开数见 [E1-COMPARISON.md](E1-COMPARISON.md)。','', '| 模式 | 全局锁持有ms/点 | 等锁ms/点 | 持锁期间拓扑读页/点 | 持锁期间坐标读页/点 |','|---|---:|---:|---:|---:|']
 for x in held:lines.append('| %s | %.4f | %.4f | %.3f | %.3f |'%(x['mode'],x['mean_hold_ms'],x['mean_wait_ms'],x['mean_held_topology_reads'],x['mean_held_coordinate_reads']))
 lines+=['','以上是初始180万图之后前1000次插入，四更新加四查询线程的新增单批观察。页读计数包围全局degree_mutex_真正持锁区间：拓扑原页读取发生在获取该锁之前，所以该列为0；写页在持锁区间，表中只计读。exact约257页坐标读取在全局锁内，证实了原向量I/O与剪枝共同串行化的问题。三轮主阶段分解和同盘OdinANN参照沿用已完成的 [性能核验](../update-perf-audit-2026-11-17/RESULTS.md)：四线程纯插入吞吐中位数分别为PQ 86.860、exact 15.993、OdinANN完成写入422.345点/秒。OdinANN为原生512维、4KiB同页Vamana，本系统为512+512维、8KiB分离DEG，只作实现参照。','', '“原系统查询”在本实验中特指E1沿用的 `multigraph-20261113/scripts/multigraph.cpp` 的Engine，源码SHA256为 `0b2bdfac57a34aabd3bbc3d1b9426aa0fc907ff87186649b0194169abd0cc3b9`，与已发布3bd0a59检查点一致；不代表根目录所有SSDIndex搜索变体。', '', '静态对照在同一张180万B图、相同坐标/PQ、同一500条报告查询上进行，均为NVMe3 O_DIRECT Linux AIO、四线程、五权重、三轮。每个路径与权重先暖一批；原Engine工作缓冲复用但数据页缓存每查询清空，执行器的页和向量缓存同样只属于单次查询。计时不含索引加载、入度初始化扫描、GT处理和JSON写盘。','', '| alpha | 原系统L96延迟ms | 执行器L96延迟ms | 原系统ID召回 | 执行器ID召回 |','|---|---:|---:|---:|---:|']
 data=[]
 for a in (.1,.3,.5,.7,.9):
  groups={p:[x for x in rows if x['path']==p and round(x['alpha'],1)==a] for p in ('original','executor')}
  fixed={p:dict(ms=st.median([x['mean_us'] for x in v if x['L']==96])/1000.,recall=st.median([x['recall'] for x in v if x['L']==96])) for p,v in groups.items()}
  ms={p:[matched([x for x in v if x['rep']==r]) for r in range(3)] for p,v in groups.items()}
  data.append(dict(alpha=a,fixed=fixed,matched=ms))
  lines.append('| %.1f | %.4f | %.4f | %.4f | %.4f |'%(a,fixed['original']['ms'],fixed['executor']['ms'],fixed['original']['recall'],fixed['executor']['recall']))
 lines+=['','主比较采用匹配95% ID Recall@10的实测夹点内插值，每轮独立内插后取三轮中位数；不内插尾延迟。两条路径的五个权重均由L48和L96夹住95%。','', '| alpha | 原系统95%延迟ms | 执行器95%延迟ms | 执行器/原系统 | 额外延迟ms |','|---|---:|---:|---:|---:|']
 for x in data:
  a=st.median([v['latency_us'] for v in x['matched']['original']])/1000.;b=st.median([v['latency_us'] for v in x['matched']['executor']])/1000.
  lines.append('| %.1f | %.4f | %.4f | %.3fx | %.4f |'%(x['alpha'],a,b,b/a,b-a))
 lines+=['','这是执行器查询路径整体的额外延迟，包含单页同步等待、每读fstat、页锁/版本判断、临时分配、PQ计算、遍历与精排实现差异。原Engine为beam4与SIMD距离，执行器为逐节点遍历与double累加距离；固定L时展开数和召回也不完全相同，因此不能将该倍数全部归因于页锁或某一项机制。匹配召回消除了查询质量目标差异，仍不等于完成了逐组件因果消融。', '', '结合原版分解：PQ的I/O调用约占累计插入延迟58.6%；exact全局入度锁等待约52.1%，该锁还覆盖约257页/点的坐标读取。按用户授权进入独立优化分支，仅验证正确性和并发，不重跑E1。']
 (ROOT/'BASELINE-RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 (R/'baseline-analysis.json').write_text(json.dumps(dict(held=held,static=data),indent=2)+'\n')
if __name__=='__main__':main()
