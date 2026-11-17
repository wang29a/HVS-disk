"""Summarize disjoint phases and explicitly overlapping I/O/lock observers."""
from pathlib import Path
import json
import statistics as st

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runtime'
E1=Path('/mnt/pmem0/hvs-update-basic-2026-11-16/e1-degree-protected-v2')
PHASES=['分配/坐标写入/PQ编码','候选搜索','新点选边/精确剪枝','新点拓扑写入','反向边读改写（含重剪）','提交/入口维护','私有缓存与临时对象清理']


def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def median(v):return st.median(v) if v else None


def profile(folder):
    rows=[json.loads(v) for v in (Path(folder)/'inserts.jsonl').read_text().splitlines()]
    assert len(rows)==1000
    total=sum(r['insert_total_ns'] for r in rows)
    phase=[sum(r['phase_ns'][i] for r in rows) for i in range(7)]
    assert sum(phase)==total,'phase partition must sum exactly, including object destruction'
    assert all(r['reprune_ns']<=r['phase_ns'][4] for r in rows)
    overlays={key:sum(r[key] for r in rows) for key in ('reprune_ns','page_lock_wait_ns','ledger_wait_ns','ledger_hold_ns','guard_ns','degree_replace_ns','invariant_ns')}
    io=sum(sum(r['read_io_ns_'+str(i)]+r['write_io_ns_'+str(i)] for i in (0,1)) for r in rows)
    return dict(insertions=len(rows),total_insert_latency_ns=total,
        phases=[dict(name=name,mean_ms=value/len(rows)/1e6,percent=100*value/total) for name,value in zip(PHASES,phase)],
        reverse_excluding_reprune=dict(mean_ms=(phase[4]-overlays['reprune_ns'])/len(rows)/1e6,percent=100*(phase[4]-overlays['reprune_ns'])/total),
        overlays={key:dict(mean_ms=v/len(rows)/1e6,percent=100*v/total) for key,v in overlays.items()},
        io_overlay=dict(mean_ms=io/len(rows)/1e6,percent=100*io/total),
        ledger_accounting_software_estimate=dict(mean_ms=sum(overlays[k] for k in ('guard_ns','degree_replace_ns','invariant_ns'))/len(rows)/1e6,
            percent=100*sum(overlays[k] for k in ('guard_ns','degree_replace_ns','invariant_ns'))/total),
        per_insert={key:sum(r[key] for r in rows)/len(rows) for key in ('topology_read_pages','coordinate_read_pages','topology_write_pages','coordinate_write_pages','exact_pair_calls','coordinate_calls','coordinate_cache_hits','cache_hit_copy_bytes')})


def main():
    rows=read(OUT/'measurements.json') if (OUT/'measurements.json').exists() else []
    upstream=read(OUT/'odin-measurements.json') if (OUT/'odin-measurements.json').exists() else []
    result=dict(measurements=rows,profiles=[],overheads=[],upstream=upstream)
    lines=['# 更新性能核验','',
        '当前执行器通过Linux AIO读写，但每次只向read_fd/write_fd提交一个8192字节请求，并等待这一批完成。它没有调用同步pread系统调用，也没有使用io_uring批量/流水读；单个调用线程实际只有一个未完成请求。下面的软件I/O回合按一次提交并等待完成计数，不代表设备内部NAND读取次数。','',
        '| E1首批5万插入 | 拓扑读/点 | 坐标读/点 | 写/点 | 总提交完成回合/点 | 请求字节/点MiB |',
        '|---|---:|---:|---:|---:|---:|']
    for mode in ('pq','exact'):
        p=E1/'variants'/(mode+'-t4-r1')/'checkpoint-1/update-summary.json'
        if not p.exists():continue
        v=read(p);n=v['insertions'];a=v['topology_read_pages']/n;b=v['coordinate_read_pages']/n;w=(v['topology_write_pages']+v['coordinate_write_pages'])/n
        lines.append('| %s | %.2f | %.2f | %.2f | %.2f | %.3f |'%(mode,a,b,w,a+b+w,(a+b+w)*8192/2**20))
    if (OUT/'clock-cost.json').exists():
        clock=read(OUT/'clock-cost.json');result['clock_cost']=clock
        lines+=['','单线程时钟微基准估计每次steady_clock调用额外%.2fns。按首批5万点的软件I/O回合数给出保守上界，原有页锁计时的两次时钟调用约占：'%clock['net_clock_ns_estimate']]
        for mode in ('pq','exact'):
            v=read(E1/'variants'/(mode+'-t4-r1')/'checkpoint-1/update-summary.json')
            calls=sum(v[k] for k in ('topology_read_pages','coordinate_read_pages','topology_write_pages','coordinate_write_pages'))/v['insertions']
            upper=100*2*calls*clock['net_clock_ns_estimate']/(v['mean_insert_us']*1000)
            lines.append('%s约不超过%.4f%%（仅时钟调用的估计，不包含算法锁本身；真实采样与频率变化会影响该估计）。'%(mode,upper))
    lines+=['','结构审计、BFS、全图入度重算、GT处理、召回与理论上限统计、快照及fsync、逐条JSON序列化与写盘均在插入计时之外，其对已报告插入QPS区间的直接占比为0%。整个E1流水线的墙钟时间另包括这些检查点工作，不能与插入区间混淆。',
        '', '计时内包括每操作样本的内存追加、时钟读取、进度原子计数和每1000次一条进度输出。四个连续查询线程是原E1规定的混合负载。入度更新、唯一入边保护、提交状态及入口维护属于生产路径的算法工作，不能直接删除后仍把结果叫同一个正确性方案。下面用相同负载下的lean/e1驱动估算观察开销，并用额外插桩版本报告算法内部时间。',
        '', '| 距离 | 原始lean批次秒（中位数） | E1记录批次秒（中位数） | E1记录额外占比估计 | 插桩相对E1额外耗时 |', '|---|---:|---:|---:|---:|']
    for mode in ('pq','exact'):
        groups={kind:[r for r in rows if r['mode']==mode and r['query_threads']==4 and r['kind']==kind] for kind in ('lean','logged','nostats','profile')}
        if not all(groups.values()):continue
        a=median([r['result']['wall_seconds'] for r in groups['lean']]);b=median([r['result']['wall_seconds'] for r in groups['logged']]);c=median([r['result']['wall_seconds'] for r in groups['profile']])
        overhead=dict(mode=mode,repetitions_each={k:len(v) for k,v in groups.items()},lean_seconds=a,e1_seconds=b,profile_seconds=c,e1_recording_fraction_estimate=1-a/b,observer_relative_overhead=c/b-1,original_counter_fraction_estimate=1-median([r['result']['wall_seconds'] for r in groups['nostats']])/a)
        result['overheads'].append(overhead)
        lines.append('| %s | %.6f | %.6f | %+.3f%% | %+.3f%% |'%(mode,a,b,100*(1-a/b),100*(c/b-1)))
    lines.append('')
    for v in result['overheads']:
        lines.append('原有页锁计时/I/O计数/度数遥测在%s下的额外占比估计为%+.3f%%；此对照保留全部算法锁、入度修改、保边与提交检查，只用于诊断。'%(v['mode'],100*v['original_counter_fraction_estimate']))
    lines+=['','开销估计来自三轮旋转顺序的独立重放。混合负载是四个闭环连续查询线程，记录机制也可能改变查询发出速率，因此差值是混合吞吐的净影响，不是纯CPU开销。负差值表示在这次样本中没有检出正开销，不能解释为记录机制会加速算法。单次1000点、小范围图状态差异、I/O波动和插桩扰动均限制其精度。',
        '', '1000点样本限定为初始180万图之后的前1000次插入，不代表全部20万更新的平均。以下占比的分母是1000次插入累计延迟，不是四线程批次墙钟；各主阶段互斥并包含私有缓存析构。每轮主阶段之和精确等于该轮插入总时间；表格各项分别取三轮中位数，其和可能因取中位数而略偏离100%。满度重剪嵌套在反向边读改写内，页锁、入度锁和I/O是叠加观察项，不能与主阶段直接求和。I/O计时包围read_fd/write_fd调用，包含上下文获取、提交、等待和完成处理，不是设备内部延迟。']
    for row in rows:
        if row['kind']!='profile':continue
        data=profile(row['output']);result['profiles'].append(dict(name=row['name'],mode=row['mode'],rep=row['rep'],profile=data))
    for mode in ('pq','exact'):
        values=[x['profile'] for x in result['profiles'] if x['mode']==mode]
        if not values:continue
        lines+=['','## %s：四更新线程与四查询线程'%mode,'','| 主阶段 | 平均每插入毫秒（三轮中位数） | 累计延迟占比 |','|---|---:|---:|']
        for i,name in enumerate(PHASES):lines.append('| %s | %.4f | %.3f%% |'%(name,median([v['phases'][i]['mean_ms'] for v in values]),median([v['phases'][i]['percent'] for v in values])))
        for key,label in [('reprune_ns','其中：满度重剪'),('page_lock_wait_ns','叠加：页锁等待'),('ledger_wait_ns','叠加：入度锁等待'),('ledger_hold_ns','叠加：入度锁持有（含内部计算和I/O）')]:
            lines.append('| %s | %.4f | %.3f%% |'%(label,median([v['overlays'][key]['mean_ms'] for v in values]),median([v['overlays'][key]['percent'] for v in values])))
        lines.append('| 叠加：I/O调用（含提交、等待及完成处理） | %.4f | %.3f%% |'%(median([v['io_overlay']['mean_ms'] for v in values]),median([v['io_overlay']['percent'] for v in values])))
        lines.append('| 入度替换/保护/边界核验软件段 | %.4f | %.3f%% |'%(median([v['ledger_accounting_software_estimate']['mean_ms'] for v in values]),median([v['ledger_accounting_software_estimate']['percent'] for v in values])))
        copies=median([v['per_insert']['cache_hit_copy_bytes'] for v in values])/2**20
        lines+=['','每次插入的坐标缓存命中返回会产生约%.3fMiB的向量值复制（中位数）；这是按返回值复制的源码路径计数，不是存储I/O字节。'%copies]
    lines+=['','## 同机同盘上游参照','',
        'OdinANN固定为官方0.1.1（FAST26 AE），commit7840ce96ac74a461b8efe2739b6aed5b14c15b9a，源文件不改，启用原生BG_IO_THREAD与DELTA_PRUNING，使用支持的Linux AIO配置。原生单空间512维、4KiB同页布局、Vamana图与码本，对照当前双空间512+512维、8KiB分离布局及DEG图。因此这是实现吞吐参照，不是严格同算法或同查询质量的加速比。',
        '', '主参照都使用180万初始点、前1000个新增点、四个插入线程、零查询线程，数据页均在NVMe3。原生后台写入分别报告API返回吞吐与收齐全部预分配工作缓冲后的完成写入吞吐；仅检查后台队列为空不足以证明在途I/O结束。fsync在这两个计时区间之外，完成写入不等于崩溃持久化。',
        '', '| 路径 | 三轮吞吐中位数（点/秒） | 最小–最大 |', '|---|---:|---:|']
    for mode in ('pq','exact'):
        vals=[r['result']['insert_qps'] for r in rows if r['mode']==mode and r['query_threads']==0]
        if vals:lines.append('| 双空间%s，纯插入 | %.3f | %.3f–%.3f |'%(mode,median(vals),min(vals),max(vals)))
    for field,label in [('api_insert_qps','OdinANN原生API返回'),('drained_insert_qps','OdinANN原生完成写入')]:
        vals=[r['result'][field] for r in upstream if r['query_threads']==0]
        if vals:lines.append('| %s | %.3f | %.3f–%.3f |'%(label,median(vals),min(vals),max(vals)))
    if not upstream:lines+=['','上游参照尚未完成，不提供预测吞吐。']
    probe_path=OUT/'extra-measurements/exact-pure-profile/analysis.json'
    if probe_path.exists():
        probe=read(probe_path);result['pure_exact_probe']=probe
        data=probe['profile'];mixed=[v['profile'] for v in result['profiles'] if v['mode']=='exact']
        lines+=['','## exact纯插入的补充观察','',
            '三轮纯插入吞吐与四查询线程混合负载不同，因此另取相同初始图和前1000点做一次纯插入剖析。该样本使用相同插桩驱动、四更新线程、零查询线程；它是单批补充观察，不替代三轮未插桩吞吐，也不证明查询负载能够加速更新。并发调度、图状态和I/O行为会随负载变化，原因尚未通过单因素实验隔离。',
            '', '| 叠加观察项 | 四查询线程，三轮中位数ms/点 | 零查询线程，单批ms/点 |','|---|---:|---:|']
        for key,label in [('ledger_wait_ns','入度锁等待'),('ledger_hold_ns','入度锁持有'),('reprune_ns','满度重剪')]:
            lines.append('| %s | %.4f | %.4f |'%(label,median([v['overlays'][key]['mean_ms'] for v in mixed]),data['overlays'][key]['mean_ms']))
        lines.append('| I/O调用 | %.4f | %.4f |'%(median([v['io_overlay']['mean_ms'] for v in mixed]),data['io_overlay']['mean_ms']))
    lines+=['','纯插入结果不能直接与原E1的四查询线程混合吞吐相比。原始每点分解、命令、源码/二进制身份、独立重放结果与观察器字节一致控制保留在本实验runtime目录。']
    save(OUT/'analysis.json',result)
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
