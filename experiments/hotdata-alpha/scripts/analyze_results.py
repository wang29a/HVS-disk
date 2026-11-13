#!/usr/bin/env python3
"""Read raw diagnostic runs; retain wins and losses with repeat variation."""
import collections
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent.parent


def interpolate(points, target):
    ordered = sorted(points, key=lambda p: p['recall'])
    for a, b in zip(ordered, ordered[1:]):
        if a['recall'] <= target <= b['recall']:
            f = (target-a['recall'])/(b['recall']-a['recall']) if b['recall'] != a['recall'] else 0
            return {k: a[k]+f*(b[k]-a[k]) for k in
                    ('L', 'qps', 'outer_latency_us', 'comparisons', 'topology_ios', 'rerank_ios')}
    return None


def main():
    runs=[json.loads(p.read_text()) for p in sorted((ROOT/'results').glob('*.json'))]
    grouped=collections.defaultdict(list)
    for r in runs:
        for target in (90,95):
            point=interpolate(r['points'], target)
            if point:
                grouped[r['backend'],r['threads'],r['scheme'],target].append(point)
    matched={}
    for key, samples in grouped.items():
        if len(samples) != 3:
            continue
        row={k:statistics.mean(p[k] for p in samples) for k in samples[0]}
        row['qps_stddev']=statistics.stdev(p['qps'] for p in samples)
        row['qps_median']=statistics.median(p['qps'] for p in samples)
        row['qps_repeats']=[p['qps'] for p in samples]
        matched[key]=row
    comparisons=[]
    for backend in ('all-hot','coord-hot','ssd'):
        for t in (1,4):
            for target in (90,95):
                m=matched.get((backend,t,'merge',target))
                n=matched.get((backend,t,'no-alpha',target))
                if m is not None and n is not None:
                    comparisons.append(dict(backend=backend,threads=t,target_recall=target,
                        merge=m,no_alpha=n,ratio=m['qps']/n['qps'],
                        observed_mean_win=m['qps']>n['qps']))
    (ROOT/'analysis/comparison.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    complete=len(runs)==36
    title='Laion 热数据机制诊断结果' if complete else 'Laion 热数据机制诊断阶段结果（未完成）'
    lines=['# '+title,'',
           '已完成 {}/36 个任务。原始数据为已有 Laion、alpha=0.7 的 1,000 条查询及配套 GT；没有计算新 GT。以下数值是三次重复的均值与样本标准差，在各次实测曲线覆盖范围内插值，未外推。'.format(len(runs)),'',
           '| 后端 | 线程 | Recall@10 | Merge QPS ± SD | No-alpha QPS ± SD | Merge/No-alpha |',
           '|---|---:|---:|---:|---:|---:|']
    for r in comparisons:
        lines.append('| {} | {} | {}% | {:.1f} ± {:.1f} | {:.1f} ± {:.1f} | {:.3f}x |'.format(
            r['backend'],r['threads'],r['target_recall'],r['merge']['qps'],r['merge']['qps_stddev'],
            r['no_alpha']['qps'],r['no_alpha']['qps_stddev'],r['ratio']))
    if complete:
        lines += ['',
          '本轮没有找到可靠的 Merge 反超。全热、单线程、90% Recall 的均值比为 1.007x，但 Merge 三次 QPS 约为 1230/1235/1231，No-alpha 为 1049/1309/1312；优势由 No-alpha 一次偏低的重复拉动。三次中位数分别为 1231 与 1309 QPS，不能把 0.7% 的均值差写成性能改善。其他 11 个匹配工作点的 Merge 平均 QPS 均低于 No-alpha。', '',
          '全热数据还存在明显 CPU 供给波动。按各测量窗口对齐 pidstat，Merge 单线程 rep2 的 L=140/160/200 点只获得约 30% CPU，而同次较小 L 时接近 100%；No-alpha 单线程 rep1 也出现相似下降。四线程 Merge rep3 的部分点只有约 190% CPU。它们说明计时受到调度/争用等 CPU 供给变化影响；具体竞争来源没有追踪，不能归因给某个进程，也不能把变慢误写成 SSD 等待。所有点都保留，没有删除异常重复。', '',
          '因而目前支持的结论是：SSD 和仅坐标热的已测工作点中，No-alpha 的平均吞吐更高；全热实验尚不支持“消除存储等待即可使 Merge 胜出”的假设。全热反超及其幅度需要在 CPU 供给稳定的代表点进一步复核。']
    audit_path=ROOT/'analysis/audit.json'
    if audit_path.exists():
        audit=json.loads(audit_path.read_text())
        lines += ['', '完整性审计：正式队列从 {} 至 {}，约 {:.1f} 分钟；324 个点均满足至少十秒，峰值 RSS {:.2f} GiB，驱动/二进制哈希一致，同方案同 L 的召回率在三种后端与各次重复间一致。全热 {} 个点中，有 {} 个点的进程 CPU 供给低于请求核数的 80%，这是诊断标记，不用于删除数据。'.format(
          audit['formal_started_at'],audit['formal_finished_at'],audit['formal_elapsed_hours']*60,
          audit['max_rss_gib'],audit['all_hot_points'],audit['all_hot_low_cpu_supply_points'])]
    lines += ['',
              'all-hot 将拓扑和坐标的同样对齐请求改由内存拷贝完成；coord-hot 仅坐标由内存完成；ssd 保留原 Linux AIO 路径。原检索静态库、查询、过滤、PQ、候选维护与精排保持不变。该实验改变了存储服务和异步等待，all-hot 的胜出也不能写成 SSD 场景胜出。', '',
              '长期任务严格限制在 CPU 32、35、39、41，搜索仅 1/4 线程、BLAS 单线程、串行排队、nice=10；内存跨 NUMA 0、1 交错。这个亲和性与旧大队列不同，跨队列绝对 QPS 不作严格对照。新 SSD 小规模控制组用于衔接。', '',
              '每点独立预热整个 1,000-query 集合，再连续回放至少十秒。回放次数不增加独立样本量。三次重复和时间轮换只能描述波动，不能把很小的均值差直接认作显著优势。跨后端顺序固定，跨后端因果结论还受到时间漂移和共享机器负载影响。', '',
              '逐查询 IDs 与精确距离的字节一致性、原生计数一致性见 analysis/validation.json。内存映像逐文件 SHA256 及容量见 analysis/memory-images.json；构建身份见 analysis/build.json。原生 I/O 字段是逻辑计数，实际内存服务和 SSD 提交计数分别保存；设备级计数不能归属于单一查询。', '',
              '本报告由 scripts/analyze_results.py 从 results/*.json 生成；不删除未胜出的工作点。', '']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print('runs',len(runs),'complete comparisons',len(comparisons),
          'observed mean wins',sum(r['observed_mean_win'] for r in comparisons))


if __name__ == '__main__':
    main()
