"""Regenerate completed A/B tables from archived summaries; no latency tuning."""
import argparse
import hashlib
import json
import statistics as st
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive', type=Path, required=True,
                    help='directory containing dram-summary.json, ssd-summary.json, graphs-summary.json and elapsed.json')
ROOT = parser.parse_args().archive.resolve()
if not ROOT.is_dir():
    parser.error('archive directory does not exist')
ssd = json.loads((ROOT / "ssd-summary.json").read_text())
dram = json.loads((ROOT / "dram-summary.json").read_text())
graphs = json.loads((ROOT / "graphs-summary.json").read_text())
elapsed = json.loads((ROOT / "elapsed.json").read_text())
assert ssd["status"] == "COMPLETE" and not ssd["pending"]
assert dram["status"] == "COMPLETE" and not dram["missing"]
assert len(ssd["records"]) == 72 and len(dram["records"]) == 40
for graph in graphs.values():
    assert graph["stats"]["status"] == "PASS"
    assert graph["stats"]["export_checked"]
    assert graph["stats"]["max_degree"] <= 40

means = {}
for rule in ("pages", "compute", "weighted_pages"):
    for version in ("A", "B"):
        for alpha in (.1, .5):
            for method in ("DEG_off", "DEG_on"):
                rows = [r["matched"] for r in ssd["records"] if
                        (r["rule"], r["version"], r["alpha"], r["method"]) ==
                        (rule, version, alpha, method)]
                assert len(rows) == 3 and all(r["covered"] for r in rows)
                keys = ("latency_us", "qps", "expanded", "pq_space_evals",
                        "reranked", "topology_submitted_pages",
                        "coordinate_submitted_pages", "topology_wait_us",
                        "coordinate_wait_us")
                value = {k: st.mean(r[k] for r in rows) for k in keys}
                value["repeat_latency_us"] = [r["latency_us"] for r in rows]
                value["raw_p95_by_repeat"] = [r["raw_p95_us"] for r in rows]
                means[rule, version, alpha, method] = value

lines = [
    "# Laion 200 万反向边 A/B：正式实验完成",
    "",
    "四张图、40组训练曲线、120次DRAM报告运行、24批SSD计时全部完成，"
    "DRAM的95%召回工作点无缺口，SSD pending=0。SSD汇总的72条记录是"
    "24批计时×3条选择规则，不是72个独立计时批次。正式阶段从A1启动至SSD收尾"
    f"共{elapsed['formal_wall_seconds']/3600:.2f}小时，不含前序高并发诊断和正确性门槛。",
    "",
    "A为当前的未满追加/复制区间、满员才重剪；B为每次反向插入均重剪，目标节点写锁覆盖"
    "读取、剪枝和替换。其余构图源码和编译选项相同，R40/L200/C750，无归一化，"
    "构图、PQ导航、精排和GT使用一致的开方L2融合目标。B的函数语义及修正版作者小图门槛"
    "已独立通过，见[前置验证](../function-v2/RESULTS.md)。",
    "",
    "本轮搜索为原顺序紧凑拓扑＋DRAM完整区间旁表、两空间各64-byte PQ、精确重排，"
    "不是Full磁盘页格式对No-alpha页格式的密度实验。开关过滤均加载相同旁表以控制内存；"
    "关过滤路径不访问区间，但这里的RSS不能写成不分配旁表的主系统RSS。"
    "未加入Starling或共享缓存。构图8线程、搜索4线程，NUMA3。",
    "",
    "## 四张图",
    "",
    "| 图 | 构图分钟 | 平均度数 | 总边数 | 拓扑MiB | 旁表MiB | 最终清理改动节点 |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for name in ("A1", "B1", "A2", "B2"):
    r = graphs[name]
    lines.append(f"|{name}|{r['build']['wall_seconds']/60:.2f}|"
                 f"{r['stats']['mean_degree']:.6f}|{r['stats']['edges']}|"
                 f"{r['files']['compact.topo']['bytes']/2**20:.2f}|"
                 f"{r['files']['ranges.bin']['bytes']/2**20:.2f}|"
                 f"{r['build']['reverse_insert_stats']['final_cleanup_changed_nodes']}|")
a = st.mean(graphs[n]["stats"]["mean_degree"] for n in ("A1", "A2"))
b = st.mean(graphs[n]["stats"]["mean_degree"] for n in ("B1", "B2"))
lines += ["", f"B的平均度数比A低{100*(1-b/a):.2f}%，但R40定宽槽位和旁表预算不变，"
          "所以拓扑及旁表字节没有下降。各图最大度数均40，结构和完整导出校验通过。"
          "完整度数分布、五权重原浮点/量化活动边比例及文件身份见[graphs-summary.json](graphs-summary.json)。",
          "", "## SSD 主结果：95% Recall@10", "",
          "采用预先声明的weighted_pages规则，在500调参查询上冻结前沿，"
          "500报告查询上仅在覆盖区间插值，各重复独立插值后取三轮均值。"
          "SSD每版只测第一次图A1/B1、alpha=0.1/0.5；p95保留实际测点，不插值。",
          "", "| 版本 | alpha | 关过滤ms | 开过滤ms | 开过滤延迟增加 | 关过滤QPS | 开过滤QPS |",
          "|---|---:|---:|---:|---:|---:|---:|"]
comparisons = []
for version in ("A", "B"):
    for alpha in (.1, .5):
        off, on = [means["weighted_pages", version, alpha, m]
                   for m in ("DEG_off", "DEG_on")]
        delta = 100 * (on["latency_us"] / off["latency_us"] - 1)
        comparisons.append(dict(version=version, alpha=alpha, filter_slowdown_pct=delta))
        lines.append(f"|{version}|{alpha}|{off['latency_us']/1000:.4f}|"
                     f"{on['latency_us']/1000:.4f}|+{delta:.2f}%|"
                     f"{off['qps']:.1f}|{on['qps']:.1f}|")
lines += ["", "B在两个SSD权重上开过滤仍慢16.35%和28.49%。A关过滤相对B没有达到"
          "10%的显著优势：B/A关过滤延迟变化在alpha=0.1为+2.94%，alpha=0.5为+1.33%。"
          "不能据此把此前结果归因于追加反向边让A特别占优。",
          "", "A→B确实改变过滤劣势的幅度：alpha=0.1从35.08%收窄到28.49%，"
          "alpha=0.5从11.27%扩大到16.35%；方向均未反转。不能写成所有图结构/工作量"
          "差异都不明显，因为平均度数已经改变约14%，部分精排深度也改变超过10%。",
          "", "## B图的成本拆分", "",
          "| alpha | 过滤 | 展开数 | PQ空间评估数 | 精排数 | 拓扑提交页 | 坐标提交页 | 拓扑等待ms | 坐标等待ms |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
for alpha in (.1, .5):
    for method in ("DEG_off", "DEG_on"):
        r = means["weighted_pages", "B", alpha, method]
        lines.append(f"|{alpha}|{'关' if method=='DEG_off' else '开'}|"
                     f"{r['expanded']:.2f}|{r['pq_space_evals']:.2f}|{r['reranked']:.2f}|"
                     f"{r['topology_submitted_pages']:.2f}|{r['coordinate_submitted_pages']:.2f}|"
                     f"{r['topology_wait_us']/1000:.4f}|{r['coordinate_wait_us']/1000:.4f}|")
lines += ["", "在B上，alpha=0.1开过滤少算约32.7%的PQ评估，但多展开约39.3%；"
          "alpha=0.5少算约29.7%，多展开约14.1%。额外拓扑等待与坐标等待抵消了计算节省。"
          "reader区间包含提交/等待；这些是执行器统计的后端页，不是物理设备读页，"
          "延迟扣除reader区间也不能直接称为纯CPU时间。",
          "", "## 三条选择规则均保留", "",
          "| 规则 | B alpha=0.1开过滤延迟增加 | B alpha=0.5开过滤延迟增加 |",
          "|---|---:|---:|"]
for rule in ("pages", "compute", "weighted_pages"):
    changes = []
    for alpha in (.1, .5):
        off, on = [means[rule, "B", alpha, m] for m in ("DEG_off", "DEG_on")]
        changes.append(100 * (on["latency_us"] / off["latency_us"] - 1))
    lines.append(f"|{rule}|+{changes[0]:.2f}%|+{changes[1]:.2f}%|")
lines += ["", "## 构图波动与判读边界", ""]
for metric, label in (("expanded", "展开数"), ("pq_space_evals", "PQ评估数")):
    noise = max(v["within_version_variation"][m]
                for r in dram["build_variation"] for k, v in r["metrics"].items()
                if k == metric for m in ("A", "B"))
    lines.append(f"五权重、两种过滤状态、三条规则中，两次构图的{label}最大相对差为{100*noise:.6f}%。")
lines += ["", "两次构图仅提供波动观察，不是置信区间；三轮查询工作量重复也不是独立构图。"
          "SSD只测A1/B1，没有直接估计SSD延迟的跨构图方差，不能将DRAM工作量方差冒充"
          "SSD延迟置信区间。>=10%的延迟效应量与极小的已测工作量波动分别报告。",
          "", "按预声明判读，支持在本轮Laion2M、PQ＋精排、4线程、95%召回、"
          "紧凑拓扑＋旁表条件下保留“过滤省计算但端到端不划算”的方向；"
          "后续DEG参照应优先采用经验证的B。没有发现A关过滤显著快于B，因此本轮"
          "不触发以该理由开展的松弛系数实验。",
          "", "这不是对全部数据集、全部alpha、精确导航或Full磁盘布局的普遍结论。"
          "若把论文的无旁表系统切到B，仍需对B确认不加载旁表的RSS与运行数字，"
          "本轮加载旁表的关过滤记录不能替代它。没有启动OpenImages或全量Laion重建。",
          "", "[SSD完整汇总](ssd-summary.json)保留三轮及原始p95夹点；"
          "[DRAM完整汇总](dram-summary.json)保留40组结果、三条规则、L/r、五权重和构图波动。"
          "原始逐查询文件、冻结配置、轨迹与NUMA/噪声guard仍保存在node3独立runtime中。"
          "本地归档为汇总与身份记录，不声称已下载全部逐查询原始日志。",
          "", "队列已完成，协调器已退出。通用status.json遗留的running标签在验证PID不存在后"
          "归档为status-before-completion-reconcile.json并修正，不改变任何测量结果。"]
(ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
(ROOT / "comparison.json").write_text(json.dumps(dict(
    means=[dict(rule=k[0], version=k[1], alpha=k[2], method=k[3], **v)
           for k, v in means.items()], primary_filter_comparisons=comparisons), indent=2) + "\n")
(ROOT / "SHA256.json").write_text(json.dumps({
    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(ROOT.rglob("*")) if p.is_file() and p.name != "SHA256.json"
}, indent=2) + "\n")
print("COMPLETE:4graphs,40train/120report DRAM runs,24SSD batches;72rule-level SSD records")
