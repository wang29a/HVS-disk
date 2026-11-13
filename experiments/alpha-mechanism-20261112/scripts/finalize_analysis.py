"""Local figures/reports and bounded final-result collection; no remote compute."""
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent.parent
REMOTE='/mnt/nvme3/wz/hvs-disk-build-20261027/experiments/alpha-mechanism-20261112'


def read(name):return json.loads((ROOT/'analysis'/name).read_text())


def savefig(fig,name):
    folder=ROOT/'figures';folder.mkdir(exist_ok=True)
    fig.tight_layout()
    for ext in ('png','svg','pdf'):fig.savefig(folder/(name+'.'+ext),dpi=180)
    plt.close(fig)


def initial():
    scales=read('distance-scale.json')['reports'];sel=read('filter-selectivity.json')
    fig,ax=plt.subplots(figsize=(6.4,4.3))
    for r in scales:
        ax.plot([x['nominal_alpha'] for x in r['reference_sweep']],
                [100*x['image_contribution'] for x in r['reference_sweep']],marker='o',label=r['dataset']+' fixed-neighbor reference')
    laion=next(x for x in scales if x['dataset']=='laion')
    ordered=sorted(laion['cases'],key=lambda x:x['fixed_alpha'])
    ax.plot([x['fixed_alpha'] for x in ordered],[100*x['native_image_contribution']['mean'] for x in ordered],
            marker='s',ls='--',color='#2269a8',label='Laion: alpha-specific GT')
    ax.plot([0,1],[0,100],ls=':',color='gray',label='Contribution equals alpha')
    ax.set(xlabel='Nominal alpha',ylabel='Mean image-term contribution (%)',xlim=(0,1),ylim=(0,100),title='Neighbor-conditioned distance contribution')
    ax.grid(alpha=.2);ax.legend(fontsize=8);savefig(fig,'distance-scale-contribution')
    fig,ax=plt.subplots(figsize=(6.4,4.3))
    names=['All inspected\nadjacency edges','Unique evaluated\ndiscovery edges','Final-candidate\ndiscovery edges']
    for i,r in enumerate(sel['records']):
        v=r['rates'];ax.bar(np.arange(3)+(i-.5)*.32,[100*v[k] for k in ('rejected_inspected','rejected_unique_evaluated','rejected_candidate_parents')],
             width=.3,label='alpha='+str(r['alpha']))
    ax.set_xticks(range(3),names);ax.set(ylabel='Would be filtered (%)',ylim=(0,70),title='No-filter traces at similar recall (95.40%, 95.74%)')
    ax.legend();ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);savefig(fig,'filter-selectivity-matched-near95')
    md='# 五项执行顺序与已完成离线结果\n\n前三项已按顺序在本地完成，node3 只读取所需文件和数据切片。两数据集各抽 256 条既有报告查询。距离使用 float64 欧氏距离，贡献为 α·d_image/(α·d_image+(1−α)·d_text)，不改变现有距离目标。\n\n'
    md+='| 名义 α | Laion 对应权重 GT 上的图像贡献 | Laion 固定邻居参考集 | OpenImages 原 mixed-GT 固定邻居参考集 |\n|---|---:|---:|---:|\n'
    oi=next(x for x in scales if x['dataset']=='openimage')
    for i,c in enumerate(ordered):
        md+='| {} | {:.2f}% | {:.2f}% | {:.2f}% |\n'.format(c['fixed_alpha'],100*c['native_image_contribution']['mean'],100*laion['reference_sweep'][i]['image_contribution'],100*oi['reference_sweep'][i]['image_contribution'])
    md+='\n固定参考集的平均贡献均衡点：Laion {:.3f}，OpenImages {:.3f}。Laion 参考集取五个权重 GT 的逐查询去重并集；OpenImages 只有现有 mixed-GT，因此后者是同一邻居集合上的重算，不能称为五个固定权重的真实近邻结果。均衡点是样本邻域的描述，不是唯一的全局距离归一化参数。Laion 一条查询（原 ID 245）的十个 GT 邻居双空间距离均为零，贡献无定义，计入排除数量；有效贡献查询为 255 条。\n\n'.format(laion['reference_mean_contribution_balance_alpha'],oi['reference_mean_contribution_balance_alpha'])
    md+='| α | L/r | Recall@10 | 所有检查边会过滤 | 唯一评估边会过滤 | 最终候选首发现边会过滤 | 选择性差值（总体−候选） |\n|---|---|---:|---:|---:|---:|---:|\n'
    for x in sel['records']:
        v=x['rates'];md+='| {} | {}/{} | {:.2f}% | {:.2f}% | {:.2f}% | {:.2f}% | {:.2f} pp |\n'.format(x['alpha'],x['L'],x['r'],x['recall'],100*v['rejected_inspected'],100*v['rejected_unique_evaluated'],100*v['rejected_candidate_parents'],100*v['selectivity_gap'])
    gap=sel['difference_in_selectivity_gap_alpha01_minus_alpha05']
    md+='\n两组同为 500 条报告查询、r=80，召回相差 0.34 个百分点。选择性差值在 α=0.1 下额外大 {:.2f} pp；按查询配对 bootstrap 95% 区间为 [{:.2f}, {:.2f}] pp。这支持相对选择性更强，而不是候选首发现边的绝对被过滤比例相差很大。所有检查边含重复；唯一实际评估边等于非入口首次发现边。最终候选首发现边不是不可替代路径，不能把统计称为因果删除效果。\n\n'.format(100*gap['mean'],100*gap['ci95'][0],100*gap['ci95'][1])
    md+='| α | 全图活动边比例 |\n|---|---:|\n'
    for x in read('activity-step-three.json')['points']:
        if x['alpha'] in (.1,.3,.5,.7,.9):md+='| {} | {:.2f}% |\n'.format(x['alpha'],100*x['active_fraction'])
    md+='\n活动比例来自已完成的完整拓扑扫描，本地复用计数，没有在 node3 重扫。它是全图比例，区别于访问轨迹的边拒绝率。\n\n第 4 项继续现有局部加密，四线程、NUMA 3、同一全局锁。第 5 项只使用 0.1/0.3/0.5/0.7/0.9；初始 L=40/60/70/80/100/120/140/160/200/320，r=20/40/60/80/120/160/L（r≤L），训练不足 97% 才有界扩展 L=600/1000/1600。只在 DRAM 调参与采工作量；92/94/95/96/97% 训练召回带按原两条工作规则冻结，SSD 只计时冻结配置。已有同参数的合格点复用。等待前置分析和局部加密完成，不重建图、不占用其他 NUMA。\n\n'
    md+='五权重结果完成后，分别按名义 α 和对应权重 GT 上的图像贡献作图。曲线对称／不对称／近似平坦是预先声明的判读分支；对称只支持权重相关现象，不能单独排除任务难度、PQ 失真等因素并证明原因必在统一图。\n\n'
    md+='![距离贡献](figures/distance-scale-contribution.png)\n\n![过滤选择性](figures/filter-selectivity-matched-near95.png)\n\n'
    other=ROOT/'analysis/three-arm-layout-summary.json'
    if other.exists():
        md+='三臂对照此前已完成，本次只在本地复核汇总；95% 召回下 TXY→T｜XY：\n\n| 臂 | 规则 | TXY ms | T｜XY ms | 延迟变化 |\n|---|---|---:|---:|---:|\n'
        for x in read('three-arm-layout-summary.json')['contrasts']:
            if x['recall']==95:md+='| {} | {} | {:.3f} | {:.3f} | {:+.1f}% |\n'.format(x['arm'],x['family'],x['coupled']['latency_us']/1000,x['separated']['latency_us']/1000,x['latency_change_pct'])
        md+='\n三臂是等总维度单空间、双空间 No-alpha、双空间 Full。原始字节/PQ 总预算匹配，各有正确目标/GT；这些原顺序、无共享缓存结果均没有获得解耦收益，不能把差异直接说成双向量固有难度。双索引覆盖率另外在本地使用 OpenImages 原始向量、相同 256 条 mixed-GT 查询计算理想每空间 Top-M 的并集，M=10/20/40/80/160/320/640/1280；它是精确单空间候选截断的参考，不是任意近似双索引策略的理论上界，不能替代真实吞吐比较。数据从 nvme2 只读传输，本地计算，与第 5 项独立，不再作为第 5 项的队列依赖。\n'
    (ROOT/'FIVE-TASKS.md').write_text(md)


def plot_scan(a):
    scale_by_alpha={x['fixed_alpha']:x['native_image_contribution']['mean'] for x in next(y for y in a['distance_scale']['reports'] if y['dataset']=='laion')['cases']}
    reports=sorted(a['reports'],key=lambda x:x['alpha'])
    for axis in ('nominal','contribution'):
        x=[r['alpha'] if axis=='nominal' else scale_by_alpha[r['alpha']] for r in reports]
        fig,axs=plt.subplots(2,2,figsize=(10,7))
        for family,style in [('pages','-'),('compute','--')]:
            values={}
            for filtering in (False,True):
                rows=[next(z for z in r['records'] if z['family']==family and z['medium']=='DRAM' and z['filtering']==filtering) for r in reports]
                values[filtering]=[z.get('mean') if z['status']=='COVERED' else None for z in rows]
                axs[0,0].plot(x,[v['expanded_nodes'] if v else np.nan for v in values[filtering]],style,marker='o',label=family+(' on' if filtering else ' off'))
            saved=[u['pq_space_evals']-v['pq_space_evals'] if u and v else np.nan for u,v in zip(values[False],values[True])]
            extra=[v['expanded_nodes']-u['expanded_nodes'] if u and v else np.nan for u,v in zip(values[False],values[True])]
            axs[0,1].plot(x,saved,style,marker='o',label=family)
            axs[1,0].plot(x,[s/e if e>0 else np.nan for s,e in zip(saved,extra)],style,marker='o',label=family)
            for medium in ('DRAM','SSD'):
                ratios=[]
                for r in reports:
                    off=next(z for z in r['records'] if z['family']==family and z['medium']==medium and not z['filtering'])
                    on=next(z for z in r['records'] if z['family']==family and z['medium']==medium and z['filtering'])
                    ratios.append(off['mean']['latency_us']/on['mean']['latency_us'] if off['status']==on['status']=='COVERED' else np.nan)
                axs[1,1].plot(x,ratios,style,marker='o',label=family+' '+medium)
        titles=['Expansions at 95% Recall@10','PQ space evaluations saved by filtering','Saved evaluations / extra expansion','Filter speedup: off latency / on latency']
        for ax,title in zip(axs.flat,titles):
            ax.set_title(title,fontsize=10);ax.grid(alpha=.2);ax.legend(fontsize=7)
            ax.set_xlabel('Nominal alpha' if axis=='nominal' else 'Image contribution on corresponding GT sample')
        axs[1,1].axhline(1,color='gray',ls=':')
        savefig(fig,'five-weight-'+axis)


def watch():
    initial()
    start=time.monotonic()
    while time.monotonic()-start<43200:
        cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','node3','cat '+REMOTE+'/weight-scan/status.json']
        result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
        if result.returncode==0:
            status=json.loads(result.stdout.decode())
            if status['stage'] in ('complete','complete_with_gaps'):
                dest=ROOT/'analysis/five-weight-results.json'
                subprocess.run(['scp','-q','node3:'+REMOTE+'/weight-scan/analysis/five-weight-results.json',dest],check=True,timeout=120)
                plot_scan(json.loads(dest.read_text()))
                (ROOT/'analysis/local-finalization-status.json').write_text(json.dumps(dict(stage='complete',remote_status=status))+'\n')
                return
            if status['stage']=='failed':raise RuntimeError('weight scan failed: '+str(status.get('error')))
        time.sleep(300)
    raise RuntimeError('weight scan did not complete within 12-hour bounded collection window')


if __name__=='__main__':
    try:
        watch() if '--watch' in sys.argv else initial()
    except Exception as error:
        (ROOT/'analysis/local-finalization-status.json').write_text(json.dumps(dict(stage='failed',error=str(error)))+'\n')
        raise
