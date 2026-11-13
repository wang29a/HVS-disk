#!/usr/bin/env python3
"""Recover stored native T4 I/O times without inventing missing split times."""
from pathlib import Path
import collections,csv,json,statistics,os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent.parent
EXP=Path(os.environ.get('HVS_RESULT_ARCHIVE', str(ROOT.parent)))
DATASETS={'openimage':'OpenImages','cc3m':'CC3M','howto100m':'HowTo100M','laion':'Laion'}
METHODS=['baseline','full','merge','no-alpha']
LABELS=['Baseline','Full','Merge-2','No-alpha']
COLORS=['#374151','#2878b5','#df8f13','#009879']

def main():
    groups=collections.defaultdict(list);curves=collections.defaultdict(list);raw=[]
    for f in sorted((EXP/'2026-10-27-concurrency-alpha/raw/results').glob('*__T4__*.json')):
        d=json.loads(f.read_text())
        if d['condition'] not in ['mixed','fixed-0.5']:continue
        for p in d['points']:
            assert 0<=p['io_us']<=p['latency_us']
            curves[d['dataset'],d['scheme'],p['L']].append(p)
        for a,b in zip(d['points'],d['points'][1:]):
            if a['recall']<=90<=b['recall']:
                f=(90-a['recall'])/(b['recall']-a['recall'])
                row=dict(dataset=d['dataset'],method=d['scheme'],repeat=d['rep'],threads=4,target_recall=90)
                for k in ['L','io_us','latency_us','cpu_us','qps']:
                    row[k]=a[k]+f*(b[k]-a[k])
                row['io_share_pct']=100*row['io_us']/row['latency_us'];raw.append(row)
                groups[d['dataset'],d['scheme']].append(row);break
    rows=[]
    for (d,m),samples in sorted(groups.items()):
        assert len(samples)==3
        r=dict(dataset=d,method=m,threads=4,target_recall=90,repeats=3)
        for k in ['io_us','latency_us','io_share_pct','qps']:
            v=[s[k]for s in samples];r[k]=statistics.mean(v);r[k+'_sd']=statistics.stdev(v)
        r['single_file_io_us']=r['io_us'] if m=='baseline' else None
        r['topology_io_us']=None;r['coordinate_io_us']=None
        r['timer_scope']='legacy synchronous reader interval' if m=='baseline' else 'legacy topo poll-wait + coordinate blocking read; topo submission excluded'
        rows.append(r)
    (ROOT/'data').mkdir(exist_ok=True);(ROOT/'figures').mkdir(exist_ok=True)
    for name,values in [('legacy-native-t4-recall90-per-repeat.csv',raw),('legacy-native-t4-recall90.csv',rows)]:
        with (ROOT/'data'/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    lookup={(r['dataset'],r['method']):r for r in rows}
    fig,axes=plt.subplots(2,2,figsize=(10.8,7.1));fig.subplots_adjust(left=.09,right=.985,top=.88,bottom=.20,hspace=.34,wspace=.24)
    for ax,(d,title) in zip(axes.flat,DATASETS.items()):
        rs=[lookup[d,m]for m in METHODS]
        bars=ax.bar(range(4),[r['io_us']/1000 for r in rs],yerr=[r['io_us_sd']/1000 for r in rs],color=COLORS,capsize=3)
        for b,r in zip(bars,rs):ax.text(b.get_x()+b.get_width()/2,(r['io_us']+r['io_us_sd'])/1000+.05,f'{r["io_us"]/1000:.2f}',ha='center',fontsize=9)
        ax.set_xticks(range(4),LABELS);ax.set_ylabel('Recorded I/O time (ms/query)');ax.set_title(title);ax.grid(axis='y',alpha=.25);ax.set_axisbelow(True)
        ax.set_ylim(0,max((r['io_us']+r['io_us_sd'])/1000 for r in rs)*1.18)
    fig.suptitle('Four-thread recorded I/O time | Matched Recall@10=90%',fontsize=15)
    fig.text(.5,.035,'Legacy counters: baseline wraps a blocking reader call; separated topology records poll-wait only.\nCoordinate blocking read is included. These are not yet uniform submit+wait phase timers. Three repeats; bars: sample SD.',ha='center',fontsize=8.5,color='#4b5563')
    for ext in ['png','pdf','svg']:fig.savefig(ROOT/'figures'/('legacy-io-time-t4-recall90.'+ext),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10.8,7.1));fig.subplots_adjust(left=.09,right=.985,top=.88,bottom=.20,hspace=.34,wspace=.24)
    for ax,(d,title) in zip(axes.flat,DATASETS.items()):
        for m,label,color in zip(METHODS,LABELS,COLORS):
            points=[(l,ss)for (ds,method,l),ss in curves.items()if(ds,method)==(d,m)];points.sort()
            ax.errorbar([statistics.mean(p['recall']for p in ss)for l,ss in points],
                [statistics.mean(p['io_us']for p in ss)/1000 for l,ss in points],
                yerr=[statistics.stdev(p['io_us']for p in ss)/1000 for l,ss in points],label=label,color=color,marker='o',markersize=3,capsize=2)
        ax.set_title(title);ax.set_xlabel('Recall@10 (%)');ax.set_ylabel('Recorded I/O time (ms)');ax.grid(alpha=.2);ax.set_ylim(bottom=0)
    h,l=axes.flat[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,.08),ncol=4,frameon=False)
    fig.suptitle('Recall–recorded I/O time | Four-thread native methods',fontsize=15)
    fig.text(.5,.025,'Existing local archive only. Legacy timer boundaries differ; split phases require the queued supplemental measurement.',ha='center',fontsize=8.5,color='#4b5563')
    for ext in ['png','pdf','svg']:fig.savefig(ROOT/'figures'/('legacy-recall-io-time-t4.'+ext),dpi=180)
    plt.close(fig)
    text=['# 四线程 I/O 时间补充','',
        '此表从已有原生日志恢复 io_us，并在每次重复内匹配 90% Recall 后聚合。单位为 ms/查询，三次重复。', '',
        '| 数据集 | Baseline 单文件 I/O | Full 总 I/O | Merge 总 I/O | No-alpha 总 I/O |',
        '|---|---:|---:|---:|---:|']
    for d,title in DATASETS.items():text.append('| '+title+' | '+' | '.join(f'{lookup[d,m]["io_us"]/1000:.3f}'for m in METHODS)+' |')
    text+=['','原生旧驱动没有导出 io_us1/io_us2，因此不能从上述总时间倒推出拓扑与坐标时间。Baseline 的旧 io_us 包含同步 reader 调用，解耦拓扑旧 io_us1 只包围 poll_wait，未包含 send_io；坐标旧 io_us2 包围 blocking read_fd。它们可按历史定义报告，统一 submit+wait 的比较使用后续补测。','',
        '统一补测将 baseline 单文件读取，与 Full/Merge/No-alpha 的拓扑提交/等待、坐标阻塞读取分别计时，同时导出实际传给后端的请求数和字节。父子时间不重复相加；原始精确距离计算不计入向量 I/O 时间。','']
    (ROOT/'LEGACY-TIMES.md').write_text('\n'.join(text))
    print('Recovered 48 native T4 runs / 480 measured points; no split-time reconstruction.')

if __name__=='__main__':main()
