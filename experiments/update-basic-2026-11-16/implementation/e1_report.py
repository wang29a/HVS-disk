"""Render only measured E1 evidence; incomplete variants remain explicit."""
import json
from pathlib import Path


def render(root):
    root=Path(root);state=json.loads((root/'status.json').read_text())
    lines=['# E1：只插入实验','',
        '当前阶段：`'+state['stage']+'`。使用已验证 B 图、默认 PQ 重剪，启用唯一入边保护，不做全图连通性修复或去重。入度保护不保证全局可达。召回主口径为 ID Recall@10；表中理论上限损失是不可达 GT 条目比例，单位为百分点，不是实测召回损失。',
        '', '展开数列是95% ID召回下的DRAM精确导航工作量，用于观察图质量；SSD延迟列使用PQ导航加精确重排，分别取得同一召回目标的实测夹点。', '', '正式重测要求拓扑与坐标都在NVMe3；旧的NVMe3拓扑加NVMe2坐标结果仅作为跨盘试运行归档。若当前读取的是旧运行目录，不将其计为正式单设备E1结果。PMEM仅保存准备数据与归档。', '',
        '| 模式 | 新增点数 | 存活点数 | 不可达存活点数 | 平均度数 | 总边数 | 满度数占比 | 新增零入度 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    variants=[]
    directory=root/'variants'
    names=[name+'-r'+str(rep) for rep in (1,2,3) for name in ['pq-t4','exact-t4','pq-C50-t4','pq-always-t4','pq-t1','pq-t8']]
    for name in names:
        p=directory/name/'checkpoints.json'
        if not p.exists():continue
        points=json.loads(p.read_text());variants.append((name,points))
        for point in points:
            s=point['structure'];lines.append('| %s | %s | %s | %s | %.6f | %s | %.3f%% | %s |'%(name,point['checkpoint']*50000,point['n'],s['unreachable_live'],s['average_degree'],s['edges'],100*s.get('full_degree_fraction',0),s.get('new_zero_indegree_live','未测')))
    if not variants:lines.append('| 尚无完成测量的检查点 | — | — | — | — | — | — | — |')
    lines+=['','| 模式 | 新增点数 | 保护触发次数 | 保留旧边数 | 保留新点首入边数 | 满度重剪次数 | 重剪前边数 | 重剪后边数 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,points in variants:
        for point in points:
            p=directory/name/('checkpoint-%d'%point['checkpoint'])/'protection-batch.json'
            if not p.exists():continue
            v=json.loads(p.read_text())
            lines.append('| %s | %s | %s | %s | %s | %s | %s | %s |'%(name,point['checkpoint']*50000,v['protection_events'],v['protected_old_edges'],v['protected_new_edges'],v['full_reprunes'],v['full_before_edges'],v['full_after_edges']))
    lines+=['','度数分布的第d项为出度d的节点数；每个检查点保留完整41项直方图。每批另存满度节点重剪后直方图（原始剪枝/保护后分别记录），对应度数变化为d-40。全图BFS不可达数为观测项，新增零入度为硬门槛。']
    lines+=['','| 模式 | 新增点数 | alpha | 95%展开数 | 相对R1增幅 | 对应GT理论上限损失（百分点） | SSD平均延迟（微秒） |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for name,points in variants:
        for point in points:
            ssd={v['alpha']:v for v in point['comparison'] if v['medium']=='ssd'}
            cp=directory/name/('checkpoint-%d'%point['checkpoint']);rows=json.loads((cp/'matched.json').read_text())
            for v in point['comparison']:
                if v['medium']!='dram':continue
                a=v['alpha'];times=[r['matched']['interpolated']['latency_us'] for r in rows if r['cohort']=='report' and r['medium']=='ssd' and r['alpha']==a]
                latency=sum(times)/len(times)
                lines.append('| %s | %s | %s | %.6f | %+.3f%% | %.4f | %.3f |'%(name,point['checkpoint']*50000,a,v['expanded'],100*v['increase_vs_R1'],v['GT_theoretical_limit']['theoretical_ceiling_loss_pp'],latency))
    lines+=['','趋势以同一个200万全量B图R1的展开数为分母，同时保留绝对展开数。各检查点的GT和规模会变化，因此该趋势只是描述，不能单独归因于PQ误差。']
    if (root/'pq-expansion-trend-r1.json').exists():
        lines+=['','| 轨迹 | alpha | 已完成新增点数序列 | 展开数增幅序列 | 已完成点是否严格递增 |', '|---|---|---|---|---|']
        for rep in (1,2,3):
            p=root/('pq-expansion-trend-r%d.json'%rep)
            if not p.exists():continue
            for a,value in json.loads(p.read_text()).items():
                lines.append('| r%d | %s | %s | %s | %s |'%(rep,a,', '.join(map(str,value['inserted'])),', '.join('%+.3f%%'%(100*v) for v in value['increase_vs_R1']),
                    ('是（完整轨迹未结束）' if value['strictly_increasing_all_checkpoints'] else '尚不能确认') if len(value['inserted'])<5 else ('是' if value['strictly_increasing_all_checkpoints'] else '否')))
    lines+=['','表中r1/r2/r3是三轮独立更新重放，初始不可变图相同，增量图各自重新生成；每轮使用相同的调参/报告查询切分。共享R1和检查点0各自实测三轮，后续检查点每条独立轨迹测一轮，形成三轮独立图状态的报告证据。三轮完成前不将r1当作最终结论。',
            '', 'PQ和exact的平均度数按模式分别列出。候选数、反向边与线程数是独立消融，不改变主线默认值。未完成项目不作通过结论。E1完成或出现停止条件后先汇报，不自动进入E2/E3。',
            '', '原始逐点ID召回与理论上限标注见各检查点的 `*-annotated.json`；原始返回ID、距离、I/O计数与尾延迟见对应 `*-raw.json`。尾延迟不内插。']
    (root/'E1-RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    import sys
    render(Path(sys.argv[1]))
