#!/usr/bin/env python3
"""Audit the downloaded completion snapshot and regenerate its status report."""
import collections
import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent.parent
SNAP = ROOT / 'completed'
DATASETS = ['openimage', 'cc3m', 'howto100m', 'laion']
TITLES = dict(zip(DATASETS, ['OpenImages', 'CC3M', 'HowTo100M', 'Laion']))
LAYOUTS = ['TXY', 'TX_Y', 'TY_X', 'T_XY', 'T_X_Y']
THREADS = [1, 4, 32, 64]
LS = [20, 40, 60, 80, 100, 120, 160, 200, 300, 400]


def read(path):
    return json.loads(path.read_text())


def interpolate(rows, target):
    rows = sorted(rows, key=lambda r: r['recall'])
    for a, b in zip(rows, rows[1:]):
        if a['recall'] <= target <= b['recall']:
            fraction = ((target - a['recall']) / (b['recall'] - a['recall'])
                        if b['recall'] != a['recall'] else 0)
            return {k: a[k] + fraction * (b[k] - a[k])
                    for k in ['qps', 'latency_us', 'submitted_pages', 'L']}
    return None


def main():
    status = read(SNAP / 'status.json')
    assert status['status'] == 'complete'
    expected = set(itertools.product(DATASETS, LAYOUTS, [1, 2, 3], THREADS))
    runs = {}
    logical = {}
    physical = {}
    points = []
    for path in (SNAP / 'results').glob('*__query.json'):
        dataset, layout, rep, thread, _ = path.stem.split('__')
        key = dataset, layout, int(rep[1:]), int(thread[1:])
        rows = read(path)
        assert [r['L'] for r in rows] == LS
        assert all(r['threads'] == key[3] and r['queries'] == 1000 and
                   r['seconds'] >= 10 and r['samples'] % 1000 == 0
                   for r in rows)
        for row in rows:
            assert all(math.isfinite(row[k]) and row[k] >= 0 for k in
                       ['qps', 'latency_us', 'recall', 'submitted_pages'])
            values = tuple(row[k] for k in ['recall', 'pq_comparisons',
                           'expanded_nodes', 'reranked_nodes', 'inspected_edges',
                           'alpha_rejected_edges'])
            logical_key = dataset, row['L']
            assert logical.setdefault(logical_key, values) == values
            page_key = dataset, layout, row['L']
            assert physical.setdefault(page_key, row['submitted_pages']) == row['submitted_pages']
        assert key not in runs
        runs[key] = rows
        points.extend(rows)
    assert set(runs) == expected
    replays = [read(p) for p in (SNAP / 'results').glob('*__replay.json')]
    assert len(replays) == 240 and all(r['seconds'] >= 10 for r in replays)
    patches = [read(p) for p in (SNAP / 'results').glob('*__patch.json')]
    assert len(patches) == 480
    assert all(r['status'] == 'PASS' and r['operations'] == 10000 and
               r['modified_page_mismatches'] == 0 and
               r['restored_page_mismatches'] == 0 for r in patches)
    parity = [read(p) for p in (SNAP / 'analysis').glob('*-parity-check.json')]
    assert len(parity) == 60 and all(r['status'] == 'PASS' and
                                   r['IDs_and_distances_byte_equal'] for r in parity)
    builds = [read(SNAP / 'analysis' / ('%s__%s__r%d-build.json' % (d, l, r)))
              for d, l, r in itertools.product(DATASETS, LAYOUTS, [1, 2, 3])]
    assert all(r['status'] == 'PASS' and r['record_mismatches'] == 0 for r in builds)
    assert all(read(SNAP / 'analysis' / (d + '-completed.json'))[
        'original_source_hashes_preserved'] for d in DATASETS)
    matched = []
    for dataset, layout, thread, target in itertools.product(DATASETS, LAYOUTS, THREADS, [90, 95]):
        values = [interpolate(runs[dataset, layout, rep, thread], target) for rep in [1, 2, 3]]
        if any(v is None for v in values):
            continue
        row = dict(dataset=dataset, layout=layout, threads=thread, target_recall=target)
        for metric in values[0]:
            samples = [v[metric] for v in values]
            row[metric] = statistics.mean(samples)
            row[metric + '_stddev'] = statistics.stdev(samples)
        matched.append(row)
    def get(d, l, t, target=90):
        return next(r for r in matched if (r['dataset'], r['layout'], r['threads'],
                    r['target_recall']) == (d, l, t, target))
    audit = dict(status='PASS', global_status=status,
                 query_runs=len(runs), query_points=len(points), replay_runs=len(replays),
                 patch_runs=len(patches), patch_operations=sum(r['operations'] for r in patches),
                 parity_checks=len(parity), full_layout_validations=len(builds),
                 minimum_measured_query_seconds=min(r['seconds'] for r in points),
                 all_layouts_threads_repeats_same_logical_counts=True,
                 original_source_hashes_preserved=True, matched_recall=matched,
                 scope='New Full-alpha controlled layout harness; no dynamic ANN update or new GT.')
    (ROOT / 'AUDIT.json').write_text(json.dumps(audit, indent=2) + '\n')
    lines = ['# 四数据集五布局预实验结果（2026-11-03）', '',
             '队列从 11 月 2 日 19:26:09 运行到 11 月 3 日 09:26:41，约 14 小时；包含编译、校验、构建、哈希、预热与测量。全部串行，T=1/4/32/64，BLAS=1，NUMA 0/1。完成后 tmux 会话已经退出。', '',
             '已审计 240 次搜索、2,400 个 L 测量点、240 次同轨迹回放和 480 次字段写入探针（480 万次逻辑操作）。60 次布局全字段校验、60 次原生搜索等价检查全部通过。不同布局、线程和重复的 Recall、PQ 比较数、扩展节点等逻辑计数完全一致；写入后与恢复后的整页核对无不一致，四数据集的源索引哈希保持不变。', '',
             '以下都是新统一 harness、完整 alpha、原始 ID 顺序、固定 16 MiB 查询内页缓存、已有查询及 GT 的结果。没有 Starling 重排，也不是原生 baseline/Full/Merge/No-alpha 的新一轮排名。先在每个重复内部插值匹配召回，再取三次均值；只在覆盖范围内插值。', '',
             '## 查询结果', '',
             '90% Recall@10、32 线程的 QPS：', '',
             '| 数据集 | TXY | TX\u007cY | TY\u007cX | T\u007cXY | T\u007cX\u007cY |',
             '|---|---:|---:|---:|---:|---:|']
    for d in DATASETS:
        lines.append('| ' + TITLES[d] + ' | ' + ' | '.join('%.0f' % get(d, l, 32)['qps'] for l in LAYOUTS) + ' |')
    lines += ['', '同一工作点下的提交页数与平均服务延迟：', '',
              '| 数据集 | TXY 页/查询 | T\u007cXY 页/查询 | TXY 延迟 ms | T\u007cXY 延迟 ms |',
              '|---|---:|---:|---:|---:|']
    for d in DATASETS:
        a, b = get(d, 'TXY', 32), get(d, 'T_XY', 32)
        lines.append('| %s | %.1f | %.1f | %.2f | %.2f |' %
                     (TITLES[d], a['submitted_pages'], b['submitted_pages'],
                      a['latency_us']/1000, b['latency_us']/1000))
    lines += ['', 'TXY 在四数据集的 90% 工作点均领先。结果和页请求计数一致：共置页在读取拓扑后可以继续提供精排坐标；原始 ID 顺序下，分离拓扑页的密度尚未转化为足够的页内访问复用。这个解释得到计数支持，但不代表元数据压缩和局部性重排之后仍然相同。', '',
              'TXY 从 32 到 64 线程的 QPS 变化分别为 ' + '、'.join('%s %+.1f%%' %
                 (TITLES[d], 100*(get(d, 'TXY', 64)['qps']/get(d, 'TXY', 32)['qps']-1))
                 for d in DATASETS) + '，平均服务延迟约翻倍。这说明本配置在 32 线程附近已接近吞吐平台；具体瓶颈还需结合阶段时间和设备遥测判断。64 线程包括 SMT。', '',
              'L≤400 的最高 Recall 分别为 ' + '、'.join('%s %.2f%%' % (TITLES[d],
                max(r['recall'] for r in runs[d, 'TXY', 1, 1])) for d in DATASETS) +
              '。CC3M 尚未覆盖 95%，没有为它外推 95% 工作点。', '',
              '## 字段写入探针', '',
              '本轮没有观察到仅靠物理解耦就获得稳定的单字段页写入优势。X-only、Y-only、T-only 的每次操作，各布局均提交 8 KiB 读取和 8 KiB 写入；字段变小并不自动缩小 O_DIRECT 页级读改写。以四数据集的均匀 ID 探针为例，各布局单字段吞吐约 9.2k–9.5k operations/s，不能把这种小差异当成明确收益。', '',
              'XY 同时修改时，TXY 和 T|XY 只需读改写一页，T|X|Y 需要两页；后者提交写入为 16 KiB/操作，吞吐约减半。热点探针保留同样的页数关系。这些是串行、无写缓存合并的字段 patch 成本；没有图/PQ 维护、并发版本控制、WAL 或恢复验证，不能称为动态 ANN 更新性能。', '',
              '## 证据与边界', '',
              '原始逐查询聚合、trace、pidstat/iostat/mpstat 完整日志仍保存在 node3 实验目录；本地 completed/ 保存最终结果 JSON、分析 JSON 和队列日志，completed-results.tar.gz 是下载快照。AUDIT.json 记录本次完整性检查及逐重复匹配召回结果，scripts/summarize_completed.py 可重建本报告。', '',
              '当前计时包含细粒度插桩，尚无关闭插桩的开销对照；每查询服务延迟也不包含开放到达率下的排队。1,000 条独立查询的循环回放不增加独立样本。设备遥测在首条曲线期间才启动，逐 PID 遥测随测量进程启动；旧版 mpstat 的 CPU 范围语法问题已修复并保留重启记录。', '',
              '单空间/双空间真实语义对照、M=0/1/2/3/Full 的布局预算轴、Starling 局部性轴，以及完整动态 ANN 与查询更新混合负载尚未运行。此次结果不支持“解耦总是更快”，也不能据此否定带压缩和重排的完整 CGL。', '']
    # Escape vertical bars inside table cells for CommonMark renderers.
    lines = [line.replace('TX|Y', 'TX\\|Y').replace('TY|X', 'TY\\|X')
             .replace('T|XY', 'T\\|XY').replace('T|X|Y', 'T\\|X\\|Y')
             if line.startswith('| 数据集 |') else line for line in lines]
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines))
    print('PASS:', len(points), 'query points;', len(replays), 'replays;', len(patches), 'patches')


if __name__ == '__main__':
    main()
