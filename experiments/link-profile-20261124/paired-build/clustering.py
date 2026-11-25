"""Cluster pairing and ground-truth co-membership evaluation."""

from pathlib import Path

import numpy as np

from run_phase0 import FILES, N, SEED, WEIGHTS, digest, open_matrix, read, save

KS = (4, 8, 16)
METHODS = ("C1-t0", "C1-t3", "C2", "C3")


def distances(a, centers, norms=None):
    if norms is None:
        norms = np.einsum("ij,ij->i", a, a)
    values = a @ centers.T
    values *= -2
    values += norms[:, None]
    values += np.einsum("ij,ij->i", centers, centers)
    return np.maximum(values, 0., out=values)


def seed_centers(sample, k, rng, norms):
    idx = rng.randint(len(sample))
    chosen = [idx]
    scores = np.full(len(sample), np.inf)
    for _ in range(k - 1):
        scores = np.minimum(scores, distances(sample, sample[idx:idx + 1], norms)[:, 0])
        scores[chosen] = 0
        cdf = scores.cumsum()
        if cdf[-1] > 0:
            idx = np.searchsorted(cdf, rng.random_sample() * cdf[-1], side="right")
        else:
            idx = next(i for i in range(len(sample)) if i not in chosen)
        chosen.append(idx)
    return sample[chosen].copy()


def balanced_assign(squared):
    n, k = squared.shape
    quota = np.full(k, n // k)
    quota[:n % k] += 1
    labels = np.full(n, -1, dtype=np.int32)
    remaining = np.arange(n)
    while len(remaining):
        active = np.flatnonzero(quota > 0)
        choice = active[np.argmin(squared[np.ix_(remaining, active)], axis=1)]
        for c in active:
            candidates = remaining[choice == c]
            order = np.argsort(squared[candidates, c], kind="stable")
            accept = candidates[order[:quota[c]]]
            labels[accept] = c
            quota[c] -= len(accept)
        remaining = remaining[labels[remaining] < 0]
    return labels


def centroids(sample, labels, previous):
    result = previous.copy()
    for i in range(len(result)):
        chunks = sample[labels == i]
        if len(chunks):
            result[i] = chunks.mean(axis=0)
    return result


def lloyd(sample, k, rng, rounds, balanced=False, initial=None):
    norms = np.einsum("ij,ij->i", sample, sample)
    centers = seed_centers(sample, k, rng, norms) if initial is None else initial.copy()
    labels = None

    def assign_sample(current):
        ds = distances(sample, current, norms)
        return balanced_assign(ds) if balanced else ds.argmin(axis=1)

    for _ in range(rounds):
        new = assign_sample(centers)
        if labels is not None and np.array_equal(new, labels):
            break
        centers = centroids(sample, new, centers)
        labels = new
    return centers, assign_sample(centers)


def assign(matrix, centers):
    n = len(matrix)
    first, second = np.zeros(n, "<u4"), np.zeros(n, "<u4")
    ratio = np.empty(n, "<f8")
    total = 0.
    for begin in range(0, n, 16384):
        end = min(n, begin + 16384)
        values = distances(matrix[begin:end].astype(np.float64), centers)
        indices = np.argsort(values, kind="stable", axis=1)[:, :2]
        first[begin:end], second[begin:end] = indices.T
        a, b = values[np.arange(end - begin)[:, None], indices].T
        total += np.sqrt(a).sum()
        block = np.full(len(a), np.inf)
        np.divide(b, a, out=block, where=a > 0)
        block[(a == 0) & (b == 0)] = 1.
        ratio[begin:end] = np.sqrt(block) - 1
    return dict(first=first, second=second, ratio=ratio, mean_l2=float(total / n))


def greedy(table):
    k = len(table)
    score = table.copy()
    mapping = np.zeros(k, "<u4")
    for _ in range(k):
        row, col = np.unravel_index(score.argmax(), score.shape)
        mapping[col] = row
        score[row, :] = -1
        score[:, col] = -1
    return mapping


def optimal(table):
    # For K <= 16 the exact subset assignment is small and needs no SciPy.
    k = len(table)
    last = (1 << k) - 1
    best = np.full(last + 1, -1, dtype=np.int64)
    parent = np.full(last + 1, -1, dtype=np.int16)
    best[0] = 0
    for mask in range(last):
        row = bin(mask).count("1")
        todo = last ^ mask
        while todo:
            step = todo & -todo
            col = step.bit_length() - 1
            value = best[mask] + table[row, col]
            if value > best[mask | step]:
                best[mask | step] = value
                parent[mask | step] = col
            todo -= step
    mapping = np.zeros(k, "<u4")
    mask = last
    for row in reversed(range(k)):
        col = int(parent[mask])
        mapping[col] = row
        mask ^= 1 << col
    return mapping


def bit(shard):
    return np.left_shift(np.uint32(1), shard)


def coverage_rows(matched):
    values = matched.mean(axis=2)
    return [dict(alpha=a, mean=float(v.mean()), p01=float(np.percentile(v, 1)))
            for a, v in zip(WEIGHTS, values.T)]


def coverage(mask, ids, nbr):
    return coverage_rows((mask[ids, None, None] & mask[nbr]) != 0)


def legacy_coverage(x, y, ids, nbr):
    return coverage_rows((x[ids, None, None] == x[nbr]) | (y[ids, None, None] == y[nbr]))


def expand(mask, destinations, scores, proportion):
    if proportion == 0:
        return mask, 0., 0
    eligible = [np.where((mask & dest) == 0, score, np.inf) for dest, score in zip(destinations, scores)]
    best = np.minimum(*eligible)
    available = int(np.isfinite(best).sum())
    target = int(np.ceil(len(mask) * proportion))
    if target > available:
        return None, None, available
    eps = float(np.partition(best, target - 1)[target - 1])
    grown = mask.copy()
    for dest, score in zip(destinations, eligible):
        can = score <= eps
        grown[can] |= dest[can]
    return grown, eps, int(np.count_nonzero(grown != mask))


def estimate(max_nodes, degree=40):
    # Two f32, 512-dimensional spaces, using the repository RAM estimate.
    return max_nodes * 1.1 * (4 * 1024 + degree * 4 * 1.3) / (2 ** 30)


def member_summary(mask, k):
    sizes = []
    memberships = np.zeros(len(mask), dtype=np.uint8)
    for i in range(k):
        members = (mask & (1 << i)) != 0
        sizes.append(int(members.sum()))
        memberships += members
    if np.any(memberships == 0) or np.any(memberships > 4):
        raise ValueError("Invalid paired membership")
    counts = np.bincount(memberships, minlength=5).tolist()
    work = sum(sizes)
    return dict(sizes=sizes, minimum=min(sizes), maximum=max(sizes), mean=work / k, point_copies=work,
                work_ratio=work / len(mask), membership_histogram=counts[1:],
                estimated_build_peak_gib=estimate(max(sizes)))


def cluster(run, k):
    run.guard(memory=5 * 2 ** 30)
    ids = np.fromfile(run.root / "inputs/training.u32", "<u4")
    x, y = [open_matrix(run.root / "inputs" / name) for name in FILES]
    xs, ys = x[ids].astype(np.float64), y[ids].astype(np.float64)
    x_bal, train_x = lloyd(xs, k, np.random.RandomState(SEED + k), 20, balanced=True)
    y_derived = centroids(ys, train_x, np.zeros((k, ys.shape[1])))
    y_refined, _ = lloyd(ys, k, np.random.RandomState(SEED + 100 + k), 3, initial=y_derived)
    x_free, _ = lloyd(xs, k, np.random.RandomState(SEED + k), 20)
    y_free, _ = lloyd(ys, k, np.random.RandomState(SEED + 100 + k), 20)
    metrics = {}
    arrays = {}
    for key, data, centers in (("xb", x, x_bal), ("xf", x, x_free), ("yf", y, y_free),
                               ("y0", y, y_derived), ("y3", y, y_refined)):
        result = assign(data, centers)
        metrics[key] = result.pop("mean_l2")
        arrays[key + "_centers"] = centers
        arrays.update({key + "_" + field: values for field, values in result.items()})
    out = run.root / "analysis" / ("K%d-assignments.npz" % k)
    temp = out.with_suffix(".partial.npz")
    np.savez(temp, **arrays)
    temp.replace(out)
    return dict(assignment_sha256=digest(out), mean_l2=metrics, path=str(out))


def evaluate(run, k, source, rates):
    path = Path(source["path"])
    if digest(path) != source["assignment_sha256"]:
        raise ValueError("Cluster assignment hash changed at K=%d" % k)
    file = np.load(path)
    old_x, old_y = file["xf_first"], file["yf_first"]
    x = file["xb_first"]
    overlap = np.bincount((old_x * k + old_y).astype(np.int64), minlength=k * k).reshape((k, k))
    ids = np.fromfile(run.root / "inputs/coverage.u32", "<u4")
    nbr = np.fromfile(run.root / "analysis/exact-top40.bin", "<u4", offset=12).reshape(-1, 5, 40)
    sx, sy = np.bincount(old_x, minlength=k).tolist(), np.bincount(old_y, minlength=k).tolist()
    baseline = dict(x_sizes=sx, y_sizes=sy, work_ratio=2., point_copies=2 * N,
                    estimated_build_peak_gib=estimate(max(sx + sy), 2 * (40 // 3)),
                    coverage=legacy_coverage(old_x, old_y, ids, nbr))
    methods = {}
    mappings = np.arange(k), np.arange(k), greedy(overlap), optimal(overlap)
    for method, axis, mapping in zip(METHODS, ("y0", "y3", "yf", "yf"), mappings):
        is_c1 = method.startswith("C1")
        side = "xb" if is_c1 else "xf"
        xv = x if is_c1 else old_x
        yv = mapping[file[axis + "_first"]].astype("<u4")
        mask = bit(xv) | bit(yv)
        destinations = bit(file[side + "_second"]), bit(mapping[file[axis + "_second"]].astype("<u4"))
        scores = file[side + "_ratio"], file[axis + "_ratio"]
        paired = int((xv == yv).sum())
        variants = {}
        for rate in rates:
            allowed, eps, n = expand(mask, destinations, scores, rate)
            if allowed is None:
                variants[str(rate)] = dict(infeasible=True, eligible_nodes=n)
                continue
            item = member_summary(allowed, k)
            item.update(epsilon=eps, additional_point_ratio=n / N, expansion_target=rate,
                        coverage=coverage(allowed, ids, nbr))
            if rate == 0 and item["point_copies"] != 2 * N - paired:
                raise ValueError("Paired membership disagrees with N(2-d)")
            variants[str(rate)] = item
        methods[method] = dict(pair_rate=paired / N, mean_y_l2=source["mean_l2"][axis],
                               y_to_x=mapping.tolist(), cases=variants)
    file.close()
    return dict(k=k, overlap=overlap.tolist(), methods=methods, baseline=baseline)


def select(rows, gate):
    eligible = []
    for r in rows:
        for method, values in r["methods"].items():
            for case in values["cases"].values():
                if case.get("infeasible"):
                    continue
                d = values["pair_rate"]
                g1 = d >= gate["pair_rate"] and case["work_ratio"] <= gate["max_work_ratio"]
                g2 = min(x["mean"] for x in case["coverage"]) >= gate["coverage"]
                case.update(G1=g1, G2=g2)
                if g1 and g2:
                    eligible.append(dict(K=r["k"], method=method, rate=case["expansion_target"], pair_rate=d,
                                         workload=case["work_ratio"], coverage=case["coverage"]))
    return max(eligible, key=lambda x: (x["pair_rate"], x["method"].startswith("C1"), -x["workload"], -x["K"]),
               default=None)


def report(run, rows, gate=None):
    best = select(rows, gate) if gate else None
    doc = ["# 配对分块构建 · Laion 200 万 · Phase 0", "",
           "同一批 10,000 个基库样本，精确 top-40，排除自身。覆盖率是近邻落入查询所在任一分片的比例。",
           "C1 在训练样本中平衡 X 簇，全量分配取最近中心。C2、C3 共享独立聚类的 X、Y 中心。",
           "边界按真实新增分片的点占比校准。若阈值处并列，全部纳入并报告实际占比。",
           "内存使用仓库公式估算，假定逐分片串行构建，仅覆盖分片建图阶段。", "",
           "| K | 方案 | 配对率 | Y 平均 L2 | 边界实际 | 分片 min/mean/max | 构建量 / N | 峰值 GiB | 五权重覆盖率均值 | 五权重 P01 | G1/G2 |",
           "|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|"]

    def perc(xs, key):
        return "/".join("%.2f%%" % (100*v[key]) for v in xs)

    for r in rows:
        k, b = r["k"], r["baseline"]
        s = b["x_sizes"] + b["y_sizes"]
        doc.append("| %d | 现有双路 k_base=1 | — | — | 0%% | %d/%.0f/%d | 2.000 | %.2f | %s | %s | — |" %
                   (k, min(s), np.mean(s), max(s), b["estimated_build_peak_gib"],
                    perc(b["coverage"], "mean"), perc(b["coverage"], "p01")))
        for name, method in r["methods"].items():
            for target, case in method["cases"].items():
                if case.get("infeasible"):
                    doc.append("| %d | %s | %.2f%% | %.4f | %.0f%% 不可达 | — | — | — | — | — | — |" %
                               (k, name, 100*method["pair_rate"], method["mean_y_l2"], 100*float(target)))
                    continue
                doc.append("| %d | %s | %.2f%% | %.4f | %.2f%% | %d/%.0f/%d | %.3f | %.2f | %s | %s | %s |" %
                           (k, name, 100*method["pair_rate"], method["mean_y_l2"], 100*case["additional_point_ratio"],
                            case["minimum"], case["mean"], case["maximum"], case["work_ratio"],
                            case["estimated_build_peak_gib"], perc(case["coverage"], "mean"),
                            perc(case["coverage"], "p01"), "%s/%s" % (case["G1"], case["G2"]) if gate else "待确认"))
    doc += ["", "一次性构建的总构建量为 1.000N，按仓库公式估算 %.2f GiB。" % estimate(N), ""]
    if gate is None:
        doc.append("边界扩展比例和 G1/G2 判定待确认。")
    elif best is None:
        doc.append("没有配置同时达到 G1、G2。Phase 0 完成。")
    else:
        doc.append("入选 K=%d、%s、边界 %.0f%%。Phase 1 待确认。" % (best["K"], best["method"], 100*best["rate"]))
    result = dict(schema=1, nodes=N, weights=WEIGHTS, rows=rows, thresholds=gate, selected=best,
                  one_shot=dict(point_copies=N, estimated_build_peak_gib=estimate(N)),
                  identity=run.results["INPUTS"]["result"], exact=run.results["A3_EXACT_TOP40"]["result"])
    save(run.root / "analysis/phase0.json", result)
    path = run.root / "analysis/PHASE0-REPORT.md"
    temp = path.with_suffix(".partial")
    temp.write_text("\n".join(doc)+"\n", encoding="utf-8")
    temp.replace(path)
    return best


def analyse(run):
    known = run.results["A3_EXACT_TOP40"]["result"]
    if digest(run.root / "analysis/exact-top40.bin") != known["sha256"]:
        raise ValueError("Coverage ground truth changed")
    approval = run.root / "source/phase0-parameters.json"
    contract = read(approval) if approval.exists() else None
    gate = contract["thresholds"] if contract else None
    rates = tuple(float(r) for r in contract["boundary_point_rates"]) if contract else (0.,)
    if not rates or sorted(set(rates)) != list(rates) or any(r < 0 or r >= 1 for r in rates):
        raise ValueError("Invalid boundary rates")
    rows = []
    for k in KS:
        source = run.step("K%d-A1-A2" % k, lambda: cluster(run,k))["result"]
        tag = "K%d-A3-A4-edge0" % k if contract is None else "K%d-A3-A4-expanded" % k
        rows.append(run.step(tag, lambda: evaluate(run,k,source,rates))["result"])
        report(run, rows, gate)
    chosen = report(run, rows, gate)
    if contract is None:
        run.status("AWAITING_PHASE0_PARAMETERS")
    else:
        run.status("PHASE0_COMPLETE", selected=chosen)
