#!/usr/bin/env python3
"""Aggregate only the new native CC3M batch; do not pool old timing samples."""
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent.parent
METHODS = ['baseline', 'full', 'merge', 'no-alpha']
METRICS = ['qps', 'latency_us', 'p95_us', 'p99_us', 'external_service_us',
           'reader_total_us', 'reader_coupled_us', 'reader_coupled_backend_us',
           'reader_topology_us', 'reader_topology_submit_us', 'reader_topology_wait_us',
           'reader_coordinate_us', 'coupled_backend_requests', 'topology_backend_requests',
           'coordinate_backend_requests', 'coupled_backend_bytes', 'topology_backend_bytes',
           'coordinate_backend_bytes', 'topology_ios', 'rerank_ios', 'comparisons',
           'normalized_candidate_evaluations', 'legacy_exact_distance_us', 'io_us']


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def matched(points, target):
    points = sorted(points, key=lambda p: p['L'])
    for p in points:
        if p['recall'] == target:
            return dict(L=p['L'], L_low=p['L'], L_high=p['L'], **{k: p[k] for k in METRICS})
    for a, b in zip(points, points[1:]):
        if a['recall'] < target < b['recall']:
            f = (target-a['recall'])/(b['recall']-a['recall'])
            return dict(L=a['L']+f*(b['L']-a['L']), L_low=a['L'], L_high=b['L'],
                        **{k: a[k]+f*(b[k]-a[k]) for k in METRICS})
    return None


def main():
    data = ROOT/'data'
    data.mkdir(exist_ok=True)
    measurements = []
    paired = []
    coverage = []
    controls = []
    means = []
    for method in METHODS:
        for repeat in [1, 2, 3]:
            name = '{}__r{}__on.json'.format(method, repeat)
            run = json.loads((ROOT/'results'/name).read_text())
            assert run['threads'] == 4 and run['observer_enabled'] and not run['one_pass']
            for p in run['points']:
                assert p['seconds'] >= 10
                measurements.append(dict(method=method, repeat=repeat, **p))
            r = matched(run['points'], 95)
            coverage.append(dict(method=method, repeat=repeat, target_recall=95,
                covered=r is not None, maximum_L=max(p['L'] for p in run['points']),
                maximum_recall=max(p['recall'] for p in run['points'])))
            if r is not None:
                r.update(method=method, repeat=repeat, target_recall=95, threads=4)
                r['submitted_pages'] = sum(r[k+'_backend_requests'] for k in ['coupled', 'topology', 'coordinate'])
                r['submitted_bytes'] = sum(r[k+'_backend_bytes'] for k in ['coupled', 'topology', 'coordinate'])
                r['reader_io_share_pct'] = 100*r['reader_total_us']/r['external_service_us']
                paired.append(r)
            off = json.loads((ROOT/'results'/name.replace('__on.json', '__off.json')).read_text())
            assert len(off['points']) == 1 and not off['observer_enabled'] and not off['one_pass']
            b = off['points'][0]
            assert b['seconds'] >= 10
            a = next(p for p in run['points'] if p['L'] == b['L'])
            for k in ['recall', 'comparisons', 'topology_ios', 'rerank_ios']:
                assert abs(a[k]-b[k]) < 1e-7
            ratio = a['qps']/b['qps']
            controls.append(dict(method=method, repeat=repeat, L=b['L'], on_qps=a['qps'],
                off_qps=b['qps'], on_over_off_ratio=ratio, review_needed=not .85 <= ratio <= 1.15))
        samples = [r for r in paired if r['method'] == method]
        if samples:
            assert len(samples) == 3
            row = dict(method=method, threads=4, target_recall=95, repeats=3)
            for k in ['L']+METRICS+['submitted_pages', 'submitted_bytes', 'reader_io_share_pct']:
                row[k] = statistics.mean(s[k] for s in samples)
                row[k+'_sd'] = statistics.stdev(s[k] for s in samples)
            row['io_timer_scope'] = 'reader-call wall time; parent/child not double-counted'
            row['latency_timer_scope'] = 'internal QueryStats.total_us; external service exported separately'
            means.append(row)
        parity = json.loads((ROOT/'analysis'/(method+'-parity.json')).read_text())
        assert parity['status'] == 'PASS'
    write_csv(data/'measurements.csv', measurements)
    write_csv(data/'recall95-per-repeat.csv', paired)
    write_csv(data/'recall95-mean-sd.csv', means)
    write_csv(data/'recall95-coverage.csv', coverage)
    write_csv(data/'observer-controls.csv', controls)
    save(data/'measurements.json', dict(batch='cc3m-native-recall95-t4-20261104',
        means=means, matched_per_repeat=paired, coverage=coverage, measurements=measurements, controls=controls))
    audit = dict(status='PASS', timed_runs=24, parity_checks=4,
        measured_on_points=len(measurements), target95_covered_methods=[r['method'] for r in means],
        target95_uncovered_methods=[m for m in METHODS if m not in [r['method'] for r in means]],
        observer_control_warnings=sum(c['review_needed'] for c in controls),
        no_extrapolation=True, no_new_GT=True, native_source_commit='d2a39e2')
    save(ROOT/'analysis/completeness.json', audit)
    lines = ['# CC3M 原生四方案：四线程 95% Recall 补测', '',
        '本表只用本次三次重复的新测量，不与十月的绝对时间样本合并。各重复内部匹配召回，无外推；K=10、beam=4、原 mixed-alpha / GT / PQ / 索引不变。', '',
        '| 方法 | QPS | 内部延迟 ms | 外部服务 ms | 单文件 I/O ms | 拓扑 I/O ms | 向量 I/O ms | I/O 合计 ms | 后端提交页/查询 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in means:
        lines.append('| %s | %.1f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.1f |' %
            (r['method'], r['qps'], r['latency_us']/1000, r['external_service_us']/1000,
             r['reader_coupled_us']/1000, r['reader_topology_us']/1000,
             r['reader_coordinate_us']/1000, r['reader_total_us']/1000, r['submitted_pages']))
    if audit['target95_uncovered_methods']:
        lines += ['', 'L≤1600 未覆盖 95% 的方法：'+', '.join(audit['target95_uncovered_methods'])+'；不补值。']
    lines += ['', 'Baseline 单文件 reader 包含内部缓存／缓冲与同步读取，后端读取子项不重复相加。解耦拓扑为提交＋等待，向量为阻塞读取；向量 I/O 不含精确距离计算。软件后端提交不等于物理 SSD 读取。', '',
        'reader I/O 占比以 external_service 为分母。旧 accuracy_compute 的 Baseline 零值表示没有完整插桩，不能当作无计算。外部服务减去 reader 时间也不能直接当作纯 CPU 时间。', '',
        '观测开／关控制中需复核的运行数：%d / 12。异常点不删除，应结合 pidstat / iostat 判断计时开销或环境波动。逐次、标准差、覆盖及控制记录保存在 data/。' % audit['observer_control_warnings'], '']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    (ROOT/'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    colors = ['#374151', '#2878b5', '#df8f13', '#009879']
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7))
    for ax, metric, ylabel in zip(axes, ['latency_us', 'qps'], ['Internal latency (ms)', 'QPS']):
        for method, color in zip(METHODS, colors):
            points = [p for p in measurements if p['method'] == method]
            ls = sorted(set(p['L'] for p in points))
            x = [statistics.mean(p['recall'] for p in points if p['L'] == l) for l in ls]
            scale = .001 if metric == 'latency_us' else 1
            y = [statistics.mean(p[metric] for p in points if p['L'] == l)*scale for l in ls]
            err = [statistics.stdev(p[metric] for p in points if p['L'] == l)*scale for l in ls]
            ax.errorbar(x, y, yerr=err, color=color, marker='o', capsize=3, label=method)
        ax.set_xlabel('Recall@10 (%)')
        ax.set_ylabel(ylabel)
        ax.axvline(95, color='#9ca3af', ls=':')
        ax.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4)
    fig.suptitle('CC3M | Native high-recall supplement | 4 threads | 3 repeats')
    fig.tight_layout(rect=(0, .08, 1, .92))
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(ROOT/'figures'/('recall-latency-qps-t4.'+ext), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
