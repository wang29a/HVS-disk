#!/usr/bin/env python3
"""Export matched native reader times; optionally render plots when installed."""
from pathlib import Path
import csv,json

ROOT=Path(__file__).resolve().parent.parent
LABEL={'openimage':'OpenImages','cc3m':'CC3M','howto100m':'HowTo100M','laion':'Laion'}
METHODS=['baseline','full','merge','no-alpha']

def main():
    data=json.loads((ROOT/'analysis/summary.json').read_text());rows=data['mean_matched']
    out=[]
    for r in rows:
        row=dict(r)
        row['reader_io_share_pct']=100*r['reader_total_us']/r['external_service_us']
        row['topology_io_share_pct']=100*r['reader_topology_us']/r['external_service_us']
        row['coordinate_io_share_pct']=100*r['reader_coordinate_us']/r['external_service_us']
        row['other_service_us']=r['external_service_us']-r['reader_total_us']
        out.append(row)
    (ROOT/'data').mkdir(exist_ok=True)
    with (ROOT/'data/native-phase-matched-recall.csv').open('w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    with (ROOT/'data/native-phase-per-repeat.csv').open('w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(data['matched_per_repeat'][0]));w.writeheader();w.writerows(data['matched_per_repeat'])
    controls=[]
    for p in sorted((ROOT/'results').glob('*__r*__on.json')):
        off_path=p.with_name(p.name.replace('__on.json','__off.json'))
        if not off_path.exists():continue
        on=json.loads(p.read_text());off=json.loads(off_path.read_text())['points'][0]
        point=next(r for r in on['points']if r['L']==off['L'])
        ratio=point['qps']/off['qps']
        controls.append(dict(run=p.stem,dataset=on['dataset'],method=on['method'],L=off['L'],
            on_qps=point['qps'],off_qps=off['qps'],on_over_off_ratio=ratio,
            review_needed=ratio<.85 or ratio>1.15,
            note='Paired overhead/environment diagnostic; preserve all points and inspect pidstat/iostat if abnormal.'))
    if controls:
        with (ROOT/'data/observer-controls.csv').open('w',newline='')as f:
            w=csv.DictWriter(f,fieldnames=list(controls[0]));w.writeheader();w.writerows(controls)
    lines=['# 原生四线程统一 reader 计时','',
      '本批使用 NUMA 3 的 CPU 和内存，四个搜索线程。先在各重复内匹配召回，再汇总三次结果。时间单位为 ms/查询。Baseline reader 总时间已包含后端读取子项。', '',
      '| 数据集 | 方法 | 重复 | 单文件读取 | 拓扑提交＋等待 | 原始向量读取 | reader 总时间 | 外部服务时间 | reader 占比 |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for d in LABEL:
        for m in METHODS:
            r=next((r for r in out if r['dataset']==d and r['method']==m and r['target_recall']==90),None)
            if not r:continue
            lines.append('| %s | %s | %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.1f%% |'%
                (LABEL[d],m,r['repeats'],r['reader_coupled_us']/1000,r['reader_topology_us']/1000,
                 r['reader_coordinate_us']/1000,r['reader_total_us']/1000,r['external_service_us']/1000,r['reader_io_share_pct']))
    lines+=['',
      'Baseline 单文件 reader 时间包括 reader 内缓存查找／缓冲处理及同步读取；另导出的 coupled_backend_us 是其子项。拓扑时间为提交和阻塞等待的非重叠调用时间，坐标读取不含精确距离计算。其余服务时间包含计算、数据结构、准备及其他等待，不能直接称为纯 CPU 时间。', '',
      '后端请求数是在 reader 传向后端的边界记录，baseline 已经过 reader 缓冲判定；仍不等于物理 SSD 读取次数。旧 QueryStats 的 io_us/io_us1/io_us2 同时保留用于核对，未与统一时间混算。控制运行关闭新增观测，用于检查计时开销；查询输出和原生计数须保持一致。', '']
    (ROOT/'PHASE-RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:return
    (ROOT/'figures').mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(10.8,7.3));fig.subplots_adjust(left=.09,right=.985,bottom=.20,top=.88,hspace=.35,wspace=.25)
    for ax,(d,label)in zip(axes.flat,LABEL.items()):
        selected=[next(r for r in out if r['dataset']==d and r['method']==m and r['target_recall']==90)for m in METHODS]
        bottom=[0]*4
        for field,title,color in [('reader_coupled_us','Coupled reader','#374151'),('reader_topology_us','Topology submit + wait','#2878b5'),
                                  ('reader_coordinate_us','Original vectors read','#df8f13'),('other_service_us','Other service time','#cbd5e1')]:
            values=[r[field]/1000 for r in selected];ax.bar(range(4),values,bottom=bottom,label=title,color=color)
            bottom=[a+b for a,b in zip(bottom,values)]
        ax.set_xticks(range(4),['Baseline','Full','Merge','No-alpha']);ax.set_ylabel('Service time (ms/query)');ax.set_title(label);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    h,l=axes.flat[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,.08),ncol=4,frameon=False,fontsize=8.5)
    fig.suptitle('Native four-thread query time | Matched recall=90%',fontsize=15)
    fig.text(.5,.025,'Disjoint reader-call wall time + remaining external service time; original vectors exclude exact-distance CPU work.',ha='center',fontsize=8.5)
    for ext in ['png','pdf','svg']:fig.savefig(ROOT/'figures'/('native-query-time-breakdown-t4.'+ext),dpi=180)
    plt.close(fig)

if __name__=='__main__':main()
