# -*- coding: utf-8 -*-
"""Publish only completed gates and measurements; never launches E1."""
from pathlib import Path
import json,statistics as st,hashlib
ROOT=Path(__file__).resolve().parent
R=ROOT/'runtime'
def read(p):return json.loads(p.read_text())
def median(rows,key):return st.median(x[key] for x in rows)
def main():
 g=R/'gates';p=R/'perf';b=R/'preflight-v2'
 assert read(g/'status.json')['stage']=='G1_G2_G3_BOTH_PASSED_E1_NOT_RUN'
 assert read(p/'status.json')['stage']=='PERF_COMPLETE_E1_NOT_RUN'
 build=read(b/'identity.json');quality=read(g/'g2-comparisons.json');san=read(g/'sanitizers.json');rows=read(p/'measurements.json');native=read(p/'odin-measurements.json');controls=read(p/'observer-controls.json')
 assert len(quality)==24 and all(x['passed'] for x in quality if x['cohort']=='report') and len(san)==4 and len(rows)==12 and len(native)==3
 static=[json.loads(x) for x in (b/'static-query.jsonl').read_text().splitlines()];assert len(static)==30
 lines=['# 束搜索执行器验证与复测结果','', '本轮以已接受的c61836b9为基线。默认已改为exact，PQ为对照：新点选边始终exact，反向重剪及合并使用所选距离；插入候选搜索仍是双空间PQ。E1后续三轮主对照尚未启动，消融另定，等待用户确认后才恢复。','', '查询内核抽取原Engine的beam4、InsertIntoPool与游标回退，复用批量PQ评分及SIMD精排；插入保留无查询权重的Pareto排序并复用束提取/批读/评分。生产coordinates()返回const引用，查询精排直接读取操作私有页；仅离线审计接口显式返回副本。缓冲仅复用分配，Context结束时清空文件/页映射，后续查询重新持锁读页。每批48页，统一页锁、页版本与原子入度保护保留。','', '## 同参数静态查询','', '同一180万B图、NVMe3、CPU72–75、NUMA3、四线程、500条报告查询、L=rerank=96、五权重、三轮，每个路径先暖一批。表为每轮查询平均延迟的三轮中位数。C++驱动对每条查询的Top10 ID和展开数与原Engine逐项核对，全部一致；这不是要求新旧增量图逐位一致。','', '| alpha | 原Engine ms | 新执行器ms | 新/原 | 原P95 ms | 新P95 ms | ID召回 | 平均展开数 |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for a in (.1,.3,.5,.7,.9):
  old=[x for x in static if x['path']=='original' and abs(x['alpha']-a)<1e-5];new=[x for x in static if x['path']=='executor' and abs(x['alpha']-a)<1e-5]
  before=median(old,'mean_us')/1000.;after=median(new,'mean_us')/1000.
  assert median(old,'recall')==median(new,'recall') and median(old,'expanded')==median(new,'expanded')
  lines.append('| %.1f | %.4f | %.4f | %.3fx | %.4f | %.4f | %.4f | %.3f |'%(a,before,after,after/before,median(old,'p95_us')/1000.,median(new,'p95_us')/1000.,median(new,'recall'),median(new,'expanded')))
 lines+=['','原Engine特指E1沿用的multigraph.cpp查询路径，源码SHA256见身份记录；本对照不代表根目录所有SSDIndex变体。准备、加载、GT和JSON输出在计时外，两者均无跨查询数据缓存。','', '## G2：新展开顺序下的增量图质量','', '在同一Laion10万子集，从冻结9万B图重新插入1万点，PQ与exact分别生成新图。原10万全量B图、GT与静态评估器按SHA256核验复用；报告查询三轮，采用原精确导航DRAM口径，在95% ID Recall@10的相邻实测夹点内插值，不外推。门槛为每个权重、每个报告轮次的展开数相对全量图增幅≤15%。','', '| 模式 | alpha | 全量B展开数 | 增量展开数 | 增幅 | 门槛 |','|---|---:|---:|---:|---:|---|']
 for mode in ('exact','pq'):
  for a in (.1,.5,.9):
   z=[x for x in quality if x['mode']==mode and x['cohort']=='report' and abs(x['alpha']-a)<1e-5]
   lines.append('| %s | %.1f | %.4f | %.4f | %+.3f%% | 通过 |'%(mode,a,st.median(x['baseline']['expanded'] for x in z),st.median(x['incremental']['expanded'] for x in z),100*median(z,'increase')))
 lines+=['','两种模式最终图均做度数、自环、重复边、ID及新增零入度检查。这里的参照是10万全量B图，与E1报告相对检查点0的展开数增幅分开。','', '## G1与G3','', 'Laion1万初始B图上，两种模式各重跑1000次固定种子混合操作的G1，每次操作独立审计；新旧增量图逐位一致不再作为门槛。定向用例包括1000轮最后入边原子争夺、回滚与新增边延后生效、过期页版本、追加页、外层操作锁下96页批读，以及查询与同一拓扑页写入交错。','', '| 构建 | 模式 | 查询窗口秒 | 总秒含排空 | 窗口内/总更新 | 查询数 | 页版本重试 | 最大读批次 |','|---|---|---:|---:|---|---:|---:|---:|']
 for x in san:
  a=x['result'];assert a['passed'] and a['post_stress_indegree_verified'] and a['update_operations']==1000
  lines.append('| %s | %s | %d | %.3f | %d/%d | %d | %d | %d |'%(x['sanitizer'],x['mode'],a['requested_seconds'],a['elapsed_seconds'],x['updates_within_query_window'],a['update_operations'],sum(w['queries'] for w in a['weights']),a['optimistic_retries'],a['max_read_batch']))
 lines+=['','每组4查询线程、2更新线程并发10分钟，alpha=0.1/0.5/0.9，整体及500查询滑窗的区间保守召回下界均≥93%。全部更新完成后独立核对结构、原子入度与最后入边，未出现ASan/TSan错误；若更新落后于查询窗口，表中单独列出排空尾部。插桩覆盖新执行器、剪枝、测试驱动与Linux reader，未覆盖所有预编译支持库和外部依赖。','', '## 四线程纯插入复测','', '同一180万初始B图后的前1000点，四更新线程、零查询线程，CPU72–75、NUMA3，拓扑与工作坐标页均在NVMe3，每轮重新创建工作文件；表为三轮中位数。准备/加载/审计/GT/JSON/快照/fsync在插入计时外，完成写入不等于崩溃持久化。','', '| 路径 | 点/秒中位数 | 最小–最大 |','|---|---:|---:|']
 for mode in ('exact','pq'):
  v=[x['result']['insert_qps'] for x in rows if x['kind']=='insert' and x['mode']==mode]
  lines.append('| 双空间%s | %.3f | %.3f–%.3f |'%(mode,st.median(v),min(v),max(v)))
 # Native schema comes directly from the unchanged, pinned completion harness.
 sample=native[0]['result'];key='drained_insert_qps'
 assert key in sample,sorted(sample)
 values=[x['result'][key] for x in native]
 lines.append('| OdinANN原生完成写入 | %.3f | %.3f–%.3f |'%(st.median(values),min(values),max(values)))
 lines+=['','OdinANN固定官方0.1.1提交7840ce96ac74a461b8efe2739b6aed5b14c15b9a，源码不改；复用冻结原生180万索引，在新目录产生并排空工作影子文件，重新测量三轮。其单空间512维、4KiB同页Vamana与本系统512+512维、8KiB分离DEG不同，作为实现参照。','', '## 1000点阶段分解','', '同一纯插入负载，两种模式各三轮1000点；表中为每轮累计插入延迟/1000后的三轮中位数。满度重剪包含在反向边读改写内，页锁等待及I/O是叠加观察项，不能与主阶段相加。','', '| 阶段或观察项 | exact ms/点 | PQ ms/点 |','|---|---:|---:|']
 groups={m:[x for x in rows if x['kind']=='profile' and x['mode']==m] for m in ('exact','pq')}
 for i,label in enumerate(('分配/坐标写入/PQ编码','候选束搜索','新点exact剪枝','新点拓扑写入','反向边读改写（含重剪）','提交与入口维护','私有上下文清理')):
  lines.append('| %s | %.4f | %.4f |'%(label,st.median(x['phases']['mean_phase_ms'][i] for x in groups['exact']),st.median(x['phases']['mean_phase_ms'][i] for x in groups['pq'])))
 for key,label in [('mean_full_reprune_ms','其中：满度重剪'),('mean_page_lock_wait_ms','叠加：页锁等待'),('mean_io_ms','叠加：I/O提交等待完成')]:
  lines.append('| %s | %.4f | %.4f |'%(label,st.median(x['phases'][key] for x in groups['exact']),st.median(x['phases'][key] for x in groups['pq'])))
 lines+=['','阶段之和在每个样本上与其内部插入总时间精确相等。观察器另以同一新版的100点单线程重放核对图、坐标与状态快照，两种模式插桩/未插桩均逐位相同。四线程正常版与插桩版的调度及图演化可能不同，不能把吞吐净差全部当作计时器CPU开销。','', '| 模式 | 读批次/点 | 写批次/点 | 坐标首次物化MiB/点 | 坐标缓存命中/点（返回引用） |','|---|---:|---:|---:|---:|']
 for mode in ('exact','pq'):
  v=groups[mode]
  lines.append('| %s | %.3f | %.3f | %.4f | %.1f |'%(mode,st.median(x['phases']['mean_read_batches'] for x in v),st.median(x['phases']['mean_write_batches'] for x in v),st.median(x['phases']['mean_coordinate_materialized_MiB'] for x in v),st.median(x['phases']['mean_coordinate_cache_hits'] for x in v)))
 lines+=['','仅首次从操作私有页物化坐标向量；缓存命中返回const引用，不再复制整个向量。查询精排直接读私有页，离线copy_coordinates_for_audit()不在插入计时路径。','', '代码与验证身份见 [VALIDATION-IDENTITY.json](VALIDATION-IDENTITY.json)。本轮不恢复E1，等待用户确认定版后再运行exact默认、PQ对照的三轮主实验，消融另定。']
 reader=ROOT.parent.parent/'source/src/utils/linux_aligned_file_reader.cpp'
 reader_identity=dict(source='source/src/utils/linux_aligned_file_reader.cpp',sha256=hashlib.sha256(reader.read_bytes()).hexdigest())
 proof=dict(reader_source=reader_identity,base_commit=build['base_commit'],default_reprune='exact',build=build,static_query=static,g1=read(g/'g1.json'),g2=quality,g2_baseline_identity=read(g/'g2-baseline-identity.json'),sanitizers=san,observer_controls=controls,performance=rows,upstream=native,upstream_identity=read(p/'upstream-identity.json'),performance_identity=read(p/'identity.json'),E1_run=False,awaiting_user_confirmation=True)
 (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');(ROOT/'VALIDATION-IDENTITY.json').write_text(json.dumps(proof,indent=2)+'\n')
if __name__=='__main__':main()
