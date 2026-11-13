#!/usr/bin/env python3
"""Read-only SSH watcher; import completed evidence and append local summaries."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = Path(os.environ.get('HVS_RESULT_ARCHIVE', str(ROOT.parent)))
MAIN = ARCHIVE/'2026-11-04-local-results'
REMOTE = '/mnt/nvme3/wz/hvs-disk-build-20261027/experiments/'
HIGH = 'cc3m-highrecall-t4-20261104'
IO = 'io-time-20261104'
LOCAL_IO = ARCHIVE/'2026-11-04-io-time'
BEGIN = '<!-- NATIVE-RECALL95-SUPPLEMENT-BEGIN -->'
END = '<!-- NATIVE-RECALL95-SUPPLEMENT-END -->'


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def write_json(path, data):
    tmp = Path(str(path)+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    os.replace(str(tmp), str(path))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def remote_states():
    script = ('import json; from pathlib import Path; p=Path('+repr(REMOTE)+'); '
              'names='+repr(['budget-reorder-20261103', IO, HIGH])+'; '
              'print(json.dumps({n:json.loads((p/n/"status.json").read_text()) for n in names}))')
    value = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
        'node3', 'python3', '-'], input=script, capture_output=True, text=True, timeout=25, check=True)
    return json.loads(value.stdout)


def fetch(remote, local):
    subprocess.run(['rsync', '-az', '--exclude=bin/', '--exclude=*.trace', '--exclude=__pycache__/',
        '--exclude=sync.lock', '--exclude=sync.pid', '--exclude=sync.log', '--exclude=local-sync-status.json',
        '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=10',
        'node3:'+REMOTE+remote+'/', str(local)+'/'], check=True, timeout=300)


def refresh_checksums(folder, archive=None):
    files = sorted(p for p in folder.rglob('*') if p.is_file() and p.name not in
        ['SHA256SUMS', archive, '.DS_Store', 'sync.lock', 'sync.pid', 'sync.log']
        and '__pycache__' not in p.parts)
    (folder/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(folder))+'\n' for p in files))
    if archive:
        with zipfile.ZipFile(folder/archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for p in files+[folder/'SHA256SUMS']:
                z.write(p, str(p.relative_to(folder)))


def append_completed(states):
    assert states[HIGH]['status'] == 'complete'
    include_io = states[IO]['status'] == 'complete'
    if include_io:
        fetch(IO, LOCAL_IO)
        subprocess.run([sys.executable, str(LOCAL_IO/'scripts/analyze_phase.py')], check=True)
    fetch(HIGH, ROOT)
    subprocess.run([sys.executable, str(ROOT/'scripts/summarize.py')], check=True)
    audit = json.loads((ROOT/'analysis/completeness.json').read_text())
    assert audit['status'] == 'PASS' and audit['timed_runs'] == 24
    supplements = [(ROOT, 'cc3m-recall95', 'RESULTS.md')]
    if include_io:
        supplements.append((LOCAL_IO, 'native-io-time', 'PHASE-RESULTS.md'))
    for source, slug, report in supplements:
        target = MAIN/'supplements'/slug
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/report, target/report)
        for directory in ['data', 'figures']:
            if (source/directory).exists():
                shutil.copytree(source/directory, target/directory, dirs_exist_ok=True)
    means = json.loads((ROOT/'data/measurements.json').read_text())['means']
    lines = [BEGIN, '## CC3M 四线程 95% Recall 补测与统一 I/O 时间', '',
        '本批在 NUMA 3 上绑定 CPU 和内存。先在每次重复内匹配召回，再汇总三次结果。沿用原图、PQ、mixed-alpha、GT、K=10 和 beam=4。', '',
        '| 方法 | QPS | 内部延迟 ms | 单文件 I/O ms | 拓扑 I/O ms | 原始向量 I/O ms | reader I/O 合计 ms | 后端提交页/查询 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in means:
        lines.append('| %s | %.1f | %.3f | %.3f | %.3f | %.3f | %.3f | %.1f |' %
            (r['method'], r['qps'], r['latency_us']/1000, r['reader_coupled_us']/1000,
             r['reader_topology_us']/1000, r['reader_coordinate_us']/1000,
             r['reader_total_us']/1000, r['submitted_pages']))
    if audit['target95_uncovered_methods']:
        lines += ['', 'L≤1600 仍未覆盖 95% 的方法：'+', '.join(audit['target95_uncovered_methods'])+'；不外推。']
    lines += ['', 'Baseline 单文件 reader 包含缓存／缓冲处理和同步读取；解耦拓扑为提交＋等待，向量读取排除精确距离计算。父子调用不重复相加；后端提交页数不等于物理 SSD 读取次数。观测开／关异常需复核 %d / 12 条，原值与误差均保留。' % audit['observer_control_warnings'], '',
        '[完整补测结果](supplements/cc3m-recall95/RESULTS.md) · [逐次匹配与标准差 CSV](supplements/cc3m-recall95/data/recall95-mean-sd.csv)', '']
    if include_io:
        lines += ['[四数据集原生 I/O 分项时间](supplements/native-io-time/PHASE-RESULTS.md)', '']
    if (MAIN/'supplements/cc3m-recall95/figures/recall-latency-qps-t4.png').exists():
        lines += ['![CC3M high-recall supplement](supplements/cc3m-recall95/figures/recall-latency-qps-t4.png)', '']
    if include_io and (MAIN/'supplements/native-io-time/figures/native-query-time-breakdown-t4.png').exists():
        lines += ['![Native query time breakdown](supplements/native-io-time/figures/native-query-time-breakdown-t4.png)', '']
    lines += [END]
    path = MAIN/'README.md'
    text = path.read_text()
    if BEGIN in text:
        before, remainder = text.split(BEGIN, 1)
        _, after = remainder.split(END, 1)
        text = before+'\n'.join(lines)+after
    else:
        text += '\n\n'+'\n'.join(lines)+'\n'
    path.write_text(text)
    figure_index = MAIN/'FIGURES.md'
    text = figure_index.read_text()
    marker = 'supplements/cc3m-recall95/RESULTS.md'
    if marker not in text:
        text += '\n\n## 追加补测\n\n[CC3M 四线程 95% Recall](supplements/cc3m-recall95/RESULTS.md)\n'
    phase_link = 'supplements/native-io-time/PHASE-RESULTS.md'
    if include_io and phase_link not in text:
        text += '\n[原生四数据集 I/O 分项时间]('+phase_link+')\n'
    figure_index.write_text(text)
    write_json(MAIN/'data/supplement-index.json', dict(imported_at=now(), source_statuses=states,
        cc3m_completeness=audit, source_cc3m_measurements_sha256=sha(ROOT/'data/measurements.json'),
        source_phase_csv_sha256=sha(LOCAL_IO/'data/native-phase-matched-recall.csv') if include_io else None,
        original_measurements_unchanged=True, original_main_scope='983 runs / 9646 points; supplemental runs separate'))
    write_json(ROOT/'local-sync-status.json', dict(status='complete', finished_at=now(),
        summary=str(MAIN/'README.md'), timed_runs=24, original_results_preserved=True))
    refresh_checksums(ROOT)
    if include_io:
        refresh_checksums(LOCAL_IO)
    refresh_checksums(MAIN, 'local-results-2026-11-04.zip')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    lock = (ROOT/'sync.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        try:
            states = remote_states()
            write_json(ROOT/'remote-status.json', dict(checked_at=now(), experiments=states))
            if states[HIGH]['status'] == 'complete':
                append_completed(states)
                print(now(), 'LOCAL_SUMMARY_APPEND_COMPLETE', flush=True)
                return
            if states[HIGH]['status'] in ['failed', 'paused_by_user']:
                write_json(ROOT/'local-sync-status.json', dict(status=states[HIGH]['status'], checked_at=now(),
                    detail=states[HIGH], summary_not_changed=True))
                print(now(), 'REMOTE_FAILED', states[HIGH], flush=True)
                return
            write_json(ROOT/'local-sync-status.json', dict(status='waiting_remote_completion',
                checked_at=now(), remote_state=states[HIGH]['status'], summary_not_changed=True))
            if not args.watch:
                return
        except Exception as e:
            print(now(), 'SYNC_RETRY', type(e).__name__, str(e), flush=True)
            if not args.watch:
                raise
        time.sleep(60)


if __name__ == '__main__':
    main()
