#!/usr/bin/env python3
"""Plot audited, existing SSD curves. Does not launch experiments."""
import collections
import os
import hashlib
import json
from pathlib import Path
import statistics
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator, FuncFormatter

ROOT = Path(__file__).resolve().parent
ARCHIVE = Path(os.environ.get('HVS_RESULT_ARCHIVE', str(ROOT.parent)))
SOURCE = ARCHIVE / '2026-10-27-concurrency-alpha'
THREADS = (1, 2, 4, 8, 16, 32)
LS = (20, 40, 60, 80, 100, 120, 160, 200, 300, 400)
SCHEMES = ('baseline', 'full', 'merge', 'no-alpha')
LABELS = ('Baseline (single-file)', 'Full-alpha', 'Merge-2', 'No-alpha')
COLORS = ('#374151', '#377eb8', '#e69f00', '#009e73')
MARKERS = ('o', '^', 's', 'D')
CONFIGS = [
    ('openimage', 'mixed', 'OpenImages', 'mixed alpha', 'main'),
    ('cc3m', 'mixed', 'CC3M', 'mixed alpha', 'main'),
    ('howto100m', 'mixed', 'HowTo100M', 'mixed alpha', 'main'),
    ('laion', 'fixed-0.5', 'Laion', 'alpha = 0.5', 'main'),
] + [('laion', 'fixed-' + a, 'Laion', 'alpha = ' + a, 'supplement')
     for a in ('0.1', '0.3', '0.7', '0.9')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_and_audit():
    groups = collections.defaultdict(list)
    identities = {}
    raw_count = 0
    for path in sorted((SOURCE / 'raw/results').glob('*.json')):
        run = json.loads(path.read_text())
        if run['threads'] not in THREADS:
            continue
        assert [p['L'] for p in run['points']] == list(LS)
        identities[str(path.relative_to(SOURCE))] = sha(path)
        for point in run['points']:
            assert point['seconds'] >= 5 and point['unique_queries'] == 1000
            groups[run['dataset'], run['condition'], run['threads'],
                   run['scheme'], point['L']].append(point)
            raw_count += 1
    assert len(identities) == 576 and raw_count == 5760
    rows = []
    metrics = ('recall', 'qps', 'latency_us')
    for key, samples in sorted(groups.items()):
        assert len(samples) == 3
        row = dict(zip(('dataset', 'condition', 'threads', 'scheme', 'L'), key))
        row['repeats'] = 3
        for metric in metrics:
            row[metric] = statistics.mean(p[metric] for p in samples)
            row[metric + '_stddev'] = statistics.stdev(p[metric] for p in samples)
            row[metric + '_repeats'] = [p[metric] for p in samples]
        assert row['recall_stddev'] == 0
        rows.append(row)
    assert len(rows) == 1920
    summary = json.loads((SOURCE / 'summary.json').read_text())
    previous = {(r['dataset'], r['condition'], r['threads'], r['scheme'], r['L']): r
                for r in summary['mean_curves']}
    for r in rows:
        p = previous[r['dataset'], r['condition'], r['threads'], r['scheme'], r['L']]
        for metric in metrics:
            assert abs(r[metric] - p[metric]) < 1e-7
            assert abs(r[metric + '_stddev'] - p[metric + '_stddev']) < 1e-7
    for d, c, *_ in CONFIGS:
        for t in THREADS:
            for s in SCHEMES:
                assert len([r for r in rows if (r['dataset'], r['condition'], r['threads'], r['scheme']) == (d,c,t,s)]) == 10
    (ROOT / 'plot-data.json').write_text(json.dumps(rows, indent=2) + '\n')
    return rows, identities


def qps_ticks(value, position):
    if value >= 1000:
        return '{:g}k'.format(value / 1000)
    return '{:g}'.format(value)


def draw(rows, config, metric):
    dataset, condition, title, alpha, category = config
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.1), sharex=True)
    fig.subplots_adjust(left=.075, right=.985, bottom=.19, top=.86, hspace=.31, wspace=.24)
    for ax, t in zip(axes.flat, THREADS):
        for scheme, label, color, marker in zip(SCHEMES, LABELS, COLORS, MARKERS):
            points = sorted([r for r in rows if
                (r['dataset'],r['condition'],r['threads'],r['scheme']) ==
                (dataset,condition,t,scheme)], key=lambda r:r['L'])
            factor = .001 if metric == 'latency_us' else 1
            ax.errorbar([p['recall'] for p in points],
                        [p[metric]*factor for p in points],
                        yerr=[p[metric+'_stddev']*factor for p in points],
                        color=color, marker=marker, markersize=3.7,
                        linewidth=1.6, elinewidth=.8, capsize=2,
                        alpha=.95, label=label)
        ax.set_title('Threads = {}'.format(t), fontsize=11)
        ax.set_xlim(54, 100)
        ax.set_xticks([60,70,80,90,95,100])
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        if metric == 'qps':
            ax.yaxis.set_major_formatter(FuncFormatter(qps_ticks))
        ax.grid(axis='both', color='#e5e7eb', linewidth=.6)
        ax.set_axisbelow(True)
        for target in (90,95):
            ax.axvline(target, color='#9ca3af', linestyle=':', linewidth=.7, zorder=0)
    short = 'Recall–Latency' if metric == 'latency_us' else 'Recall–QPS'
    fig.suptitle('{} | {} ({})'.format(short, title, alpha), fontsize=16, y=.975)
    fig.text(.5,.913,'K = 10   |   beam = 4   |   L = 20–400   |   three repeats',
             ha='center', fontsize=10, color='#4b5563')
    fig.supxlabel('Recall@10 (%)', fontsize=11, y=.135)
    ylabel = 'Mean search latency (ms; internal timer)' if metric == 'latency_us' else 'Throughput (queries/s)'
    fig.supylabel(ylabel, fontsize=11, x=.012)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5,.068),
               ncol=4, frameon=False, fontsize=10, columnspacing=2)
    note = 'Points: measured L values; bars: sample SD across three runs; independent y scale in each panel.'
    if metric == 'latency_us':
        note += '\nInternal total_us excludes some query preparation; this is not full entry-to-return latency.'
    else:
        note += '\nQPS is measured over the continuous replay loop; lines connect native points without extrapolation.'
    fig.text(.5,.018,note,ha='center',va='bottom',fontsize=8.5,color='#4b5563')
    return fig


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9,
        'axes.spines.top':False, 'axes.spines.right':False,
        'pdf.fonttype':42, 'svg.fonttype':'none', 'axes.unicode_minus':False})
    for d in ('main', 'supplement'):
        (ROOT/d).mkdir(exist_ok=True)
    rows, identities = read_and_audit()
    outputs=[]
    with PdfPages(ROOT/'recall-latency-all.pdf') as latpdf, PdfPages(ROOT/'recall-qps-all.pdf') as qpspdf:
        for config in CONFIGS:
            dataset, condition, title, alpha, category = config
            for metric, slug, document in (('latency_us','latency',latpdf),('qps','qps',qpspdf)):
                fig=draw(rows,config,metric)
                filename='{}-{}-recall-{}-t1-t32'.format(dataset,condition,slug)
                for ext in ('png','pdf','svg'):
                    out=ROOT/category/(filename+'.'+ext)
                    fig.savefig(out,dpi=180,facecolor='white')
                    outputs.append(str(out.relative_to(ROOT)))
                document.savefig(fig)
                plt.close(fig)
    manifest=dict(source=str(SOURCE), source_commit='d2a39e2bd50e6d03716c4268f7b90f931e5a6a1e',
        source_summary_sha256=sha(SOURCE/'summary.json'), script_sha256=sha(Path(__file__)),
        threads=THREADS, native_L=LS, schemes=SCHEMES, conditions=CONFIGS,
        raw_runs=576, raw_measured_points=5760, mean_points=1920,
        six_panel_sheets=16, native_curves=192, error_bars='sample standard deviation across three repeats',
        latency='mean of original internal QueryStats.total_us in milliseconds, excludes some query preparation',
        throughput='mean of per-run measured replay QPS',
        axes='Recall@10 percent on x; linear independent y axes in each panel',
        source_hashes=identities, outputs=outputs,
        matplotlib_version=matplotlib.__version__)
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    filelist=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.png','.svg','.pdf','.json','.md','.py'))
    (ROOT/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in filelist))
    with zipfile.ZipFile(ROOT/'recall-curves-t1-t32.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in filelist+[ROOT/'SHA256SUMS']:
            archive.write(p,str(p.relative_to(ROOT)))
    print('PASS: 576 raw runs / 5760 measurements / 1920 means; all agree with archived summary.')
    print('Generated 16 six-panel sheets in PNG/PDF/SVG and two 8-page metric PDFs.')


if __name__=='__main__':
    main()
