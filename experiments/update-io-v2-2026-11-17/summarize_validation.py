# -*- coding: utf-8 -*-
"""Only completed measurements/gates become final claims; no E1 launch."""
from pathlib import Path
import json,statistics as st
ROOT=Path(__file__).resolve().parent
R=ROOT/'runtime'
def read(p):return json.loads(p.read_text())
def main():
 v=R/'validation-v2';state=read(v/'status.json')
 assert state['stage']=='OPTIMIZED_G1_G3_PASSED_E1_NOT_RUN'
 parity=read(v/'g1-parity.json');san=read(v/'sanitizers.json');build=read(v/'build-identity.json')
 assert len(parity)==2 and len(san)==4 and all(x['result']['passed'] for x in san)
 lines=['# 优化分支验证结果','', 'E1首轮已归档，后续重复和消融保持暂停。本轮没有启动E1；以下结果不代表优化版已经完成E1质量评估，完成汇报后等待用户确认。默认仍为PQ，即新点exact、反向重剪PQ。','', '首次TSan运行触发了检测器内部64把持锁上限；已将批量读改成32页分批持锁，增加96页加外层操作锁的定向回归，并完整重跑优化版门槛。首轮现场保留，最终结果以validation-v2为准；冻结原版对照未变，复用其已通过的摘要。', '', '独立分支完成锁外读页与剪枝、加页写锁校验版本、每节点原子入度预留、候选/精排批量读取以及去除逐读fstat。原始剪枝源码与距离算术保持不变，上游单空间默认路径和旧执行器不修改。','', '在Laion1万节点初始B图上，100次单线程插入的图、坐标与状态快照在两种距离下均与原版逐位相同。进一步对PQ与exact各做1000次固定种子混合操作（591插入、309删除、100合并，另有最终清理），图、坐标、正常退出快照和操作序列均逐位相同；两种模式G1各完成1002次独立审计。','', '| 模式 | 图SHA256 | 混合操作逐位一致 |','|---|---|---|']
 for x in parity:
  assert x['versions'][0]==x['versions'][1]
  lines.append('| %s | `%s` | 是 |'%(x['mode'],x['versions'][0]['graph']))
 lines+=['','定向测试通过：1000轮双写者争夺最后入边、失败回滚、新增边延后可用、过期页版本拒绝、追加页、批量读，以及查询与同一物理拓扑页覆盖写入的交错。','', '| Sanitizer | 重剪距离 | 查询窗口秒 | 总秒（含排空） | 窗口内/总更新数 | 查询数 | 乐观重试数 | 入度预留冲突 | 最大批量页数 | 独立入度审计 |','|---|---|---:|---:|---|---:|---:|---:|---:|---|']
 coverage=[]
 for x in san:
  a=x['result'];assert a['update_operations']==1000 and a['post_stress_indegree_verified'] and a['initial_burst_operations']==32
  ops=[json.loads(line) for line in (v/x['sanitizer']/x['mode']/'concurrent-operations.jsonl').read_text().splitlines()]
  within=sum(op['seconds']<=a['requested_seconds'] for op in ops);assert len(ops)==1000
  coverage.append(dict(sanitizer=x['sanitizer'],mode=x['mode'],query_window_seconds=a['requested_seconds'],updates_within_query_window=within,total_updates=len(ops),total_seconds_including_drain=a['elapsed_seconds']))
  lines.append('| %s | %s | %d | %.3f | %d/%d | %d | %d | %d | %d | 通过 |'%(x['sanitizer'],x['mode'],a['requested_seconds'],a['elapsed_seconds'],within,a['update_operations'],sum(w['queries'] for w in a['weights']),a['optimistic_retries'],a['degree_reservation_retries'],a['max_read_batch']))
 lines+=['','每组从Laion1万节点B图开始，包含四查询线程、双更新线程、alpha=0.1/0.5/0.9；首32次操作突发，随后按原时间表限制最早启动时间。查询窗口固定600秒；若TSan插桩使更新落后，剩余操作在查询结束后排空，总秒包含该尾部。窗口内操作数单独列出，不能将1000总更新全部算作在600秒内完成。查询使用区间保守召回oracle，检查整体及500查询滑窗下界均不低于93%；写者结束后独立重算结构和入度，核对原子账目并检查存活节点未失去最后入边。没有ASan/TSan错误。插桩覆盖新执行器、剪枝、测试驱动和Linux reader，未覆盖所有预编译上游支持库与外部依赖。']
 micro=R/'micro'
 if (micro/'identity.json').exists():
  identity=read(micro/'identity.json');assert identity['static_recall_and_expansion_equal'] and not identity['E1_run']
  rows=read(micro/'measurements.json');assert len(rows)==6
  oldpath=ROOT.parent/'update-perf-audit-2026-11-17/runtime/analysis.json'
  if not oldpath.exists() or not read(oldpath).get('upstream'):oldpath=ROOT.parent/'update-perf-audit-2026-11-17/runtime/evidence/analysis.json'
  old=read(oldpath)
  lines+=['','## 有限性能复核（不是E1）','', '仅对初始180万B图后的前1000点做三轮四线程纯插入，坐标和拓扑均在NVMe3，准备/加载/fsync在计时外；每轮从同一初始图重新复制工作文件。多线程更新顺序和保护决策可能不同，这里不作后续20万点质量结论。该表评估组合优化，不将收益拆分归因给单一组件。','', '| 模式 | 冻结原版中位数 点/秒 | 优化版中位数 点/秒 | 优化版最小–最大 |','|---|---:|---:|---:|']
  for mode in ('pq','exact'):
   before=[x['result']['insert_qps'] for x in old['measurements'] if x['mode']==mode and x['query_threads']==0]
   after=[x['result']['insert_qps'] for x in rows if x['mode']==mode]
   lines.append('| %s | %.3f | %.3f | %.3f–%.3f |'%(mode,st.median(before),st.median(after),min(after),max(after)))
  now=[json.loads(x) for x in (micro/'static-query.jsonl').read_text().splitlines()]
  baseline=[json.loads(x) for x in (R/'baseline/static-query.jsonl').read_text().splitlines()]
  lines+=['','同一静态图、L=rerank=96、四线程、三轮的查询复核；优化版与冻结执行器的五个权重召回和展开数均保持一致。该表是同参数验证，匹配95%召回的冻结基线另见 [BASELINE-RESULTS.md](BASELINE-RESULTS.md)。','', '| alpha | 冻结执行器ms/查询 | 优化执行器ms/查询 | 原查询Engine重测ms/查询 |','|---|---:|---:|---:|']
  for alpha in (.1,.3,.5,.7,.9):
   def median(rows,path):return st.median([x['mean_us'] for x in rows if x['path']==path and round(x['alpha'],1)==alpha and x['L']==96])/1000.
   lines.append('| %.1f | %.4f | %.4f | %.4f |'%(alpha,median(baseline,'executor'),median(now,'executor'),median(now,'original')))
 lines+=['','首轮图质量以检查点0为分母的完整对照见 [E1-COMPARISON.md](E1-COMPARISON.md)；锁内读盘及静态查询实现开销见 [BASELINE-RESULTS.md](BASELINE-RESULTS.md)。','', '状态：代码与验证材料准备审查；E1后续任务不自动恢复，等待用户确认。']
 lines+=['', 'Linux编译参数、源码SHA256、逐位对照摘要与Sanitizer计数见 [VALIDATION-IDENTITY.json](VALIDATION-IDENTITY.json)。该文件只包含身份和汇总，不包含向量、图或查询轨迹。']
 identity=dict(base_commit='3bd0a59b58ccb18cf06fcfc0f27efe33591d6ba1',branch='codex/dual-space-update-io-v2',validation_run='validation-v2',E1_run=False,build=build,parity100=read(v/'parity100.json'),g1_parity=parity,sanitizers=san,g3_coverage=coverage,reference_reuse=read(v/'reference-reuse.json'))
 if (micro/'identity.json').exists():identity['micro']=read(micro/'identity.json')
 (ROOT/'VALIDATION-IDENTITY.json').write_text(json.dumps(identity,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__':main()
