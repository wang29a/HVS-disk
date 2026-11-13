"""Regenerate the completed-run audit, Chinese report and paper plots."""
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'raw' if (ROOT / 'raw/results').is_dir() else ROOT
THREADS = (1, 2, 4, 8, 16, 32, 64)
SCHEMES = ('baseline', 'full', 'merge', 'no-alpha')
NAMES = {'openimage': 'OpenImages', 'cc3m': 'CC3M',
         'howto100m': 'HowTo100M', 'laion': 'Laion'}
CONDITIONS = {d: 'fixed-0.5' if d == 'laion' else 'mixed' for d in NAMES}


def interpolate(points, target):
    curve = sorted(points, key=lambda p: p['recall'])
    for a, b in zip(curve, curve[1:]):
        if a['recall'] <= target <= b['recall']:
            f = ((target - a['recall']) / (b['recall'] - a['recall'])
                 if b['recall'] != a['recall'] else 0)
            keys = ('L', 'qps', 'latency_us', 'p99_us', 'topology_ios', 'rerank_ios')
            return {k: a[k] + f * (b[k] - a[k]) for k in keys}
    return None


def main():
    runs = [json.loads(p.read_text()) for p in sorted((RAW / 'results').glob('*.json'))]
    summary = json.loads((RAW / 'analysis/summary.json').read_text())
    points = [p for r in runs for p in r['points']]
    assert len(runs) == 672 and len(points) == 6720
    assert all(len(r['points']) == 10 for r in runs)
    assert all(p['seconds'] >= 5 and p['unique_queries'] == 1000 and
               p['samples'] % 1000 == 0 and math.isfinite(p['qps']) for p in points)
    assert len(summary['mean_curves']) == 2240
    assert all(p['repeats'] == 3 for p in summary['mean_curves'])
    assert max(p['recall_stddev'] for p in summary['mean_curves']) == 0
    build = json.loads((RAW / 'analysis/build.json').read_text())
    assert hashlib.sha256((RAW / 'scripts/steady_search.cpp').read_bytes()).hexdigest() == build['driver_sha256']
    for name in ('baseline', 'separated'):
        binary = RAW / 'bin' / ('steady-' + name)
        assert hashlib.sha256(binary.read_bytes()).hexdigest() == build[name]['binary_sha256']
    grouped = collections.defaultdict(list)
    for r in runs:
        p = interpolate(r['points'], 90)
        if p:
            grouped[r['dataset'], r['condition'], r['threads'], r['scheme']].append(p)
    matched = []
    for key, samples in sorted(grouped.items()):
        assert len(samples) == 3
        row = dict(zip(('dataset', 'condition', 'threads', 'scheme'), key))
        for metric in samples[0]:
            row[metric] = statistics.mean(p[metric] for p in samples)
            row[metric + '_stddev'] = statistics.stdev(p[metric] for p in samples)
        matched.append(row)
    lookup = {(r['dataset'], r['condition'], r['threads'], r['scheme']): r for r in matched}
    def at(d, t, s, c=None):
        return lookup[d, c or CONDITIONS[d], t, s]
    cvs = sorted(p['qps_stddev'] / p['qps'] for p in summary['mean_curves'])
    audit = dict(completed_runs=len(runs), measured_points=len(points),
                 min_seconds=min(p['seconds'] for p in points),
                 max_seconds=max(p['seconds'] for p in points),
                 recall_stddev_max=0, query_count=1000, repetitions=3,
                 qps_cv_median=statistics.median(cvs),
                 qps_cv_p90=cvs[int(.9 * (len(cvs) - 1))], qps_cv_max=max(cvs),
                 elapsed_hours=(max(r['finished_unix'] for r in runs) -
                                min(r['started_unix'] for r in runs)) / 3600,
                 note='Recall interpolation is within measured curves; no extrapolation. CV across 3 runs is descriptive, not a confidence interval.')
    (ROOT / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    (ROOT / 'matched-recall-90.json').write_text(json.dumps(matched, indent=2) + '\n')

    lines = ['# 四数据集线程数与 alpha 补测结果', '',
             '实验已全部完成。正式队列从 2026-10-27 19:35:19 运行至 2026-10-28 10:16:56（北京时间），约 14 小时 42 分钟。672 个进程、6,720 个测量点均完成，各参数组合重复三次。未新增 GT；复用了现有索引、查询、alpha 和配套 GT。', '',
             '核心结果是：低并发与高并发的排名不同。在相同 90% Recall@10 下，No-alpha 在四个数据集的 1–4 线程测试中均优于 baseline；在 32 线程时四者均由 baseline 领先。Full→Merge 在当前四个默认工作负载的 32 线程点都有小幅收益，但当前解耦执行包尚未在这些吞吐点超过 baseline。不能把这个结果归结为单一布局因素。', '',
             '## 配置与验证', '',
             '源码为 d2a39e2；baseline 与解耦版本分别复用其既有编译宏及静态库，新驱动仅用于延长测量和采集统计。K=10，beam=4，L=20/40/60/80/100/120/160/200/300/400，线程=1/2/4/8/16/32/64，NUMA 0、1 绑定且内存交错。两节点共 48 个物理核，64 线程包含 SMT。四种方法共用 nvme3 上的索引；任务串行排队，方法/线程顺序在重复间轮换。', '',
             '每点连续循环原有 1,000 条查询至少 5 秒，pilot 和未达时长的校准不计入；实际测量时长为 %.3f–%.3f 秒。循环没有增加独立查询样本数。六个原始入口与新入口的一致性检查通过，覆盖全部 OpenImages 方法、异维度 CC3M 和 Laion。所有重复的 Recall 完全相同；驱动和二进制哈希已与构建记录核对。' % (audit['min_seconds'], audit['max_seconds']), '',
             '以下 QPS 来自三次重复的平均曲线，在实测点间按召回率线性插值到 90%，没有外推，也不是直接在某个整数 L 上测得的数值。首三个数据集是原有 mixed-alpha 工作负载；Laion 默认 alpha=0.5。', '',
             '## 32 线程、90% Recall@10', '',
             '| 数据集 | Baseline QPS | Full QPS | Merge QPS | No-alpha QPS | Merge 相对 Full 增幅 |',
             '|---|---:|---:|---:|---:|---:|']
    for d, n in NAMES.items():
        q = [at(d, 32, s)['qps'] for s in SCHEMES]
        lines.append('| %s | %s | %.1f%% |' % (n, ' | '.join(f'{v:,.0f}' for v in q), (q[2] / q[1] - 1) * 100))
    lines += ['', '## 线程扩展', '',
              '下表为 No-alpha 相对 baseline 的 QPS 比值，仍在相同召回率处比较。', '',
              '| 数据集 | T1 | T2 | T4 | T8 | T16 | T32 | T64 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for d, n in NAMES.items():
        ratios = [at(d, t, 'no-alpha')['qps'] / at(d, t, 'baseline')['qps'] for t in THREADS]
        lines.append('| %s | %s |' % (n, ' | '.join(f'{v:.3f}x' for v in ratios)))
    lines += ['', 'baseline 在四个默认工作负载上均以 32 线程取得本轮最高平均 QPS。64 相对 32 线程的 baseline QPS 下降约 9.7%–10.1%；解耦方案多数接近平台期，个别上下波动。Full/Merge/No-alpha 在 32→64 时没有普遍成比例增长，因此不宜继续只用 64 线程代表方法优劣。接近的峰值须结合重复波动解释，不代表统计显著性结论。', '',
              '![Concurrency at matched recall](figures/concurrency-recall90.png)', '',
              '## Laion 已有固定 alpha 的结果', '',
              '以下全部固定 32 线程，保持比较控制条件一致；0.1/0.3/0.5/0.7/0.9 均使用相应的现有 GT。', '',
              '| alpha | Baseline QPS | Full QPS | Merge QPS | No-alpha QPS | No-alpha/Baseline |', '|---|---:|---:|---:|---:|---:|---:|']
    for alpha in (.1, .3, .5, .7, .9):
        c = f'fixed-{alpha:.1f}'
        q = [at('laion', 32, s, c)['qps'] for s in SCHEMES]
        lines.append('| %.1f | %s | %.3fx |' % (alpha, ' | '.join(f'{v:,.0f}' for v in q), q[3] / q[0]))
    lines += ['', '![Fixed-alpha sensitivity](figures/laion-alpha-recall90.png)', '',
              '前三个数据集没有新增固定 alpha GT，只沿用每条查询自带的 alpha，并保留五个 alpha 桶内的统计。不同桶包含不同查询且混合执行，不能将这些桶解释为固定 alpha 的独立 QPS 实验。端点 alpha=0/1 未测。', '',
              '## 原生 I/O 计数与执行路径', '',
              '下表是 32 线程、90% Recall@10 的每查询原生计数。baseline 第一列计数对应含向量和拓扑的单文件页；解耦方案分别记录 topology 和 rerank 计数。计数口径与缓存行为存在差异，不把它们直接解释为物理 SSD 读取量。', '',
              '| 数据集 | Baseline 页计数 | Full topo / rerank | Merge topo / rerank | No-alpha topo / rerank |', '|---|---:|---:|---:|---:|']
    for d, n in NAMES.items():
        pairs = [f"{at(d, 32, s)['topology_ios']:.1f} / {at(d, 32, s)['rerank_ios']:.1f}" for s in SCHEMES[1:]]
        lines.append('| %s | %.1f | %s |' % (n, at(d, 32, 'baseline')['topology_ios'], ' | '.join(pairs)))
    lines += ['', '这轮支持的判断是：Merge 能降低 topology 计数，并改善相对于 Full 的吞吐；它没有同步消除坐标精排开销。解耦路径的 rerank 计数随候选深度 L 增长，而 baseline 的精确距离计算发生在单文件遍历中。由于还同时涉及缓存、PQ/精确距离、重排和执行入口差异，baseline→Full 仍是整个执行包的比较，不是“只分离索引”的消融。要解释高并发转折，需要进一步把 topology、坐标读取、等待和共享缓存行为按同一口径拆开测量。', '',
              '## 波动与使用边界', '',
              '所有 2,240 个原生 L 配置的 QPS 重复间变异系数，中位数 %.2f%%，90 分位 %.2f%%，最大 %.2f%%；部分低线程、较大 L 的 Laion 点存在明显波动。因此报告保留全部三次原始数据和误差条，不能把个位数收益都描述为显著提升。设备级 iostat 与进程级 pidstat 已留档，但设备繁忙百分比不能单独证明 NVMe 饱和，也不能排除共享机器上的外部干扰。' % (100 * audit['qps_cv_median'], 100 * audit['qps_cv_p90'], 100 * audit['qps_cv_max']), '',
              '本轮是固定 1,000 条查询反复回放的持续负载，能减少旧短跑的启动/收尾影响，但不是未见查询的冷缓存实验。结果不直接代表更大查询集、开放环负载或其他硬件。未改动检索实现，未重建索引，未新增 GT。', '',
              '## 归档与再生成', '',
              '本地 completed-evidence.tar.gz 包含完整 scripts/bin/logs/results/analysis 和最终状态；raw/ 是其展开内容。summary.json 为队列汇总；matched-recall-90.json 保留逐次插值后均值与标准差；audit.json 是完整性审计。运行 analyze_completed.py 可从 raw/ 再生成本文和 PNG/PDF/SVG 图。node3 同名实验目录保留原件及本报告。', '']
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines))
    make_plots(at)
    print(json.dumps(audit, indent=2))


def make_plots(at):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#374151', '#377eb8', '#e69f00', '#009e73']
    labels = ['Baseline', 'Full', 'Merge-2', 'No-alpha']
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    out = ROOT / 'figures'
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.8), constrained_layout=True)
    for ax, (d, n) in zip(axes.flat, NAMES.items()):
        for s, color, label in zip(SCHEMES, colors, labels):
            p = [at(d, t, s) for t in THREADS]
            ax.errorbar(THREADS, [r['qps'] for r in p],
                        yerr=[r['qps_stddev'] for r in p], marker='o', markersize=3,
                        linewidth=1.5, capsize=2, color=color, label=label)
        ax.set_xscale('log', base=2)
        ax.set_xticks(THREADS, [str(t) for t in THREADS])
        ax.set_xlabel('Search threads')
        ax.set_ylabel('QPS at 90% Recall@10')
        ax.set_title(n + (' (alpha=0.5)' if d == 'laion' else ' (mixed alpha)'))
        ax.grid(alpha=.18)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Three repeats; error bars: sample standard deviation\nInterpolation within measured recall curves; NUMA 0,1; beam=4', fontsize=10)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(out / ('concurrency-recall90.' + ext), dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    alphas = (.1, .3, .5, .7, .9)
    for s, color, label in zip(SCHEMES, colors, labels):
        p = [at('laion', 32, s, f'fixed-{a:.1f}') for a in alphas]
        ax.errorbar(alphas, [r['qps'] for r in p],
                    yerr=[r['qps_stddev'] for r in p], marker='o', markersize=4,
                    linewidth=1.5, capsize=3, color=color, label=label)
    ax.set_xticks(alphas)
    ax.set_xlabel('Query alpha (existing matching GT)')
    ax.set_ylabel('QPS at 90% Recall@10')
    ax.grid(alpha=.18)
    ax.legend(fontsize=8, loc='lower center', ncol=2)
    ax.set_title('Laion: 32 threads, three repeats (mean +/- sample SD)')
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(out / ('laion-alpha-recall90.' + ext), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
