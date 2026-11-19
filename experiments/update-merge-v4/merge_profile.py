# -*- coding: utf-8 -*-
"""Instrument a copy of the frozen merge executor and check its output bytes."""
import shutil
from pathlib import Path

from common import ARCHIVE, ROOT, c, e, read, save

OUT=ARCHIVE/'merge-profile'


def replace(file, old, new):
    source=file.read_text()
    assert source.count(old)==1, '%s: %r' % (file,old)
    file.write_text(source.replace(old,new,1))


def prepare(source):
    frozen=ROOT/'implementation'
    shutil.copytree(str(frozen),str(source))
    shutil.copyfile(ROOT/'merge_profile.h',source/'merge_profile.h')
    header=source/'update_store.h'
    replace(header,'#include <array>','#include "merge_profile.h"\n#include <array>')
    replace(header,'struct IOStats {','struct IOStats {\n  MergeProfile *merge_profile = nullptr;')
    cpp=source/'dual_update.cpp'
    replace(cpp,'  std::vector<float> xy(d1_ + d2_);',
            '  if (context.stats.merge_profile)\n'
            '    context.stats.merge_profile->vector(id);\n'
            '  std::vector<float> xy(d1_ + d2_);')
    replace(cpp,'void DualUpdate::consolidate(IOStats &stats) {',
            '''void DualUpdate::consolidate(IOStats &stats) {
  auto *profile = stats.merge_profile;
  MergeTimer merge_total(profile ? &profile->total_ns : nullptr);''')
    replace(cpp,'''  const uint32_t n = allocated();
  auto dead =''','''  const uint32_t n = allocated();
  if (profile) {
    profile->merge_threads = 1;
    profile->seen_pages.resize(pages(n, coord_per_page_));
    profile->seen_vectors.resize(n);
  }
  auto dead =''')
    replace(cpp,'''        planned[id] =
            select(id, std::move(candidates), options_.exact_reprune, context);''',
            '''        if (profile) {
          MergeTimer reprune(&profile->reprune_ns);
          const auto before_io = profile->coordinate_io_ns;
          planned[id] =
              select(id, std::move(candidates), options_.exact_reprune, context);
          ++profile->repruned_nodes;
          profile->reprune_coordinate_io_ns += profile->coordinate_io_ns - before_io;
        } else {
          planned[id] =
              select(id, std::move(candidates), options_.exact_reprune, context);
        }''')
    replace(cpp,'  std::vector<uint32_t> orphans;',
            '  auto repair_start = MergeClock::now();\n  std::vector<uint32_t> orphans;')
    replace(cpp,'  std::vector<Entry> next_entries;',
            '  if (profile)\n    profile->repair_ns += merge_elapsed(repair_start);\n'
            '  std::vector<Entry> next_entries;')
    replace(cpp,'  auto bridge_degree = initial_degree;',
            '  repair_start = MergeClock::now();\n  auto bridge_degree = initial_degree;')
    replace(cpp,'  auto reachable = [&](const auto &roots',
            '  if (profile)\n    profile->repair_ns += merge_elapsed(repair_start);\n'
            '  auto reachable = [&](const auto &roots')
    replace(cpp,'  // Consolidation has no user alpha.',
            '  repair_start = MergeClock::now();\n  // Consolidation has no user alpha.')
    replace(cpp,'  // Verify the order before writing anything.',
            '  if (profile)\n    profile->repair_ns += merge_elapsed(repair_start);\n'
            '  // Verify the order before writing anything.')
    replace(cpp,'  for (uint64_t page = 0; page < page_count; ++page)\n'
                '    write_page(page, bridge_pages[page], intermediate);',
            '  const auto write_start = MergeClock::now();\n'
            '  for (uint64_t page = 0; page < page_count; ++page)\n'
            '    write_page(page, bridge_pages[page], intermediate);')
    replace(cpp,'    write_page(page, dead_pages[page], final_row);',
            '    write_page(page, dead_pages[page], final_row);\n'
            '  if (profile)\n    profile->write_ns += merge_elapsed(write_start);')
    cpp=source/'update_store.cpp'
    replace(cpp,'''  impl_->read_locked(f, page, p, stats);
  ++stats.read_pages[f];''','''  auto *profile = file == File::coordinates ? stats.merge_profile : nullptr;
  {
    MergeTimer read_time(profile ? &profile->coordinate_io_ns : nullptr);
    impl_->read_locked(f, page, p, stats);
  }
  if (profile)
    profile->page(page);
  ++stats.read_pages[f];''')
    replace(cpp,'''    impl_->reader.read_fd(impl_->fd[f], requests, impl_->reader.get_ctx());
#ifdef HVS_PROFILE_PHASES
    stats.read_io_ns[f] += profile_now() - io_tick;
#endif
    stats.read_pages[f] += n;''','''    auto *profile = file == File::coordinates ? stats.merge_profile : nullptr;
    {
      MergeTimer read_time(profile ? &profile->coordinate_io_ns : nullptr);
      impl_->reader.read_fd(impl_->fd[f], requests, impl_->reader.get_ctx());
    }
    if (profile)
      for (size_t j = 0; j < n; ++j)
        profile->page(pages[first + j]);
#ifdef HVS_PROFILE_PHASES
    stats.read_io_ns[f] += profile_now() - io_tick;
#endif
    stats.read_pages[f] += n;''')
    harness=source/'e2_driver.cpp'
    shutil.copyfile(ROOT/'e2_driver.cpp',harness)
    replace(harness,'    IOStats io;\n    try {',
            '    MergeProfile profile;\n    IOStats io;\n'
            '    if (check_deleted)\n      io.merge_profile = &profile;\n    try {')
    replace(harness,'''         << ",\\"merge_unreachable_after\\":" << io.merge_unreachable_after
         << "}\\n";''','''         << ",\\"merge_unreachable_after\\":" << io.merge_unreachable_after;
    if (check_deleted && !error) {
      file << ",\\"merge_profile\\":";
      profile.write(file);
    }
    file << "}\\n";''')
    smoke=source/'merge_check.cpp'
    shutil.copyfile(ROOT/'merge_check.cpp',smoke)
    replace(smoke,'    index.consolidate(io);\n    const auto after',
            '''    MergeProfile profile;
    io.merge_profile = &profile;
    index.consolidate(io);
    check(profile.merge_threads == 1 && profile.total_ns > 0 &&
              profile.coordinate_reads == io.read_pages[1] &&
              profile.vector_loads >= profile.distinct_vectors &&
              profile.coordinate_reads >= profile.distinct_pages,
          "unexpected merge profile counters");
    const auto after''')
    return harness,smoke


def compile_driver():
    OUT.mkdir(exist_ok=False)
    build=read(ARCHIVE/'e2-build-identity.json')
    for name,digest in build['executor']['sources'].items():
        assert c.sha(ROOT/'implementation'/name)==digest,name
    assert c.sha(ROOT/'e2_driver.cpp')==build['e2_driver']
    source=OUT/'source'
    harness,smoke=prepare(source)
    e.run(['clang-format','-style=LLVM','-i',source/'dual_update.cpp',
           source/'update_store.cpp',source/'update_store.h',source/'merge_profile.h',
           harness,smoke],'format-merge-profile',out=OUT)
    # Keep all flags, unit ordering, and linked libraries from the frozen build.
    command=build['command']
    assert command[1]=='-I'+str(ROOT/'implementation')
    executable=OUT/'merge-profile'
    units=[source/name for name in ('update_store.cpp','deg_prune.cpp','dual_update.cpp')]
    begin=command.index(str(ROOT/'implementation/update_store.cpp'))
    tail=command[begin+4:-2]
    prefix=[command[0],'-I'+str(source)]+command[2:begin]
    driver_command=prefix+units+[harness]+tail+['-o',executable]
    e.run(driver_command,'compile-merge-profile',out=OUT,timeout=1200)
    test=OUT/'merge-profile-check'
    e.run(prefix+['-DHVS_MERGE_V4']+units+[smoke]+tail+['-o',test],
          'compile-merge-profile-check',out=OUT,timeout=1200)
    from run_gates import directed_fixture
    baseline=ROOT/'runtime/gates-retain/g1/exact'
    for scenario in ('search','retain'):
        fixture=directed_fixture(baseline,OUT/('smoke-'+scenario),scenario)
        e.run([test,fixture,'exact',scenario],'smoke-'+scenario,out=OUT,timeout=120)
        assert ('MERGE_DIRECTED_PASS '+scenario+' exact') in (OUT/('smoke-'+scenario+'.log')).read_text()
    save(OUT/'identity.json',dict(command=list(map(str,driver_command)),binary=c.sha(executable),
         library=c.sha(c.BUILD/'src/libpipeann.a'),parent_driver=build['driver_binary'],
         profiled_sources={p.name:c.sha(p) for p in source.iterdir() if p.is_file()},
         smoke_cases=['search','retain']))
    return executable


def validate(post, rounds):
    result=read(post/'merge-summary.json')
    profile=result['merge_profile']
    total=profile['total_ns']
    stages=('planning','reprune','repair','write')
    assert sum(profile[s+'_ns'] for s in stages)==total
    assert 0<profile['reprune_coordinate_io_ns']<=profile['reprune_ns']<total
    assert profile['reprune_coordinate_io_ns']<=profile['coordinate_io_ns']<=total
    reads=profile['coordinate_read_pages']
    vectors=profile['coordinate_vector_loads']
    assert 0<profile['unique_coordinate_pages']<=reads==result['coordinate_read_pages']
    assert 0<profile['unique_coordinate_vectors']<=vectors
    assert 0<profile['repruned_nodes']<=result['merge_live_nodes'] and profile['merge_threads']==1
    assert result['coordinate_write_pages']==0
    products={name:c.sha(post/name) for name in ('graph.pages','index.meta')}
    controls={}
    for reference in ('exact-r1','exact-r2'):
        controls[reference]={name:c.sha(rounds/reference/'after-merge'/name) for name in products}
        assert controls[reference]==products, 'profile changed bytes relative to '+reference
    merged=dict(profile=profile, products=products, controls=controls, byte_identical=True,
                query_threads=4, hot_numa=3, profile_overhead_in_timing=True)
    save(OUT/'verified-profile.json',merged)
    lines=['# exact 合并耗时分解','',
           'Laion 200 万点、冻结删除集 20 万点。E2 exact 第三轮，1 个合并线程，另有 4 个查询线程。',
           '剖析版与 exact 第一、二轮的拓扑和入口元数据逐字节相同。','',
           '| 阶段 | 秒 | 占合并耗时 |','|---|---:|---:|']
    for label,phase in zip(('规划','重剪（含向量读取）','补边','写入'),stages):
        ns=profile[phase+'_ns']
        lines.append('| %s | %.3f | %.2f%% |'%(label,ns/1e9,100*ns/total))
    lines+=['','共重剪 %d 个节点。重剪内坐标读取用时 %.3f 秒，占重剪 %.2f%%。'%(
        profile['repruned_nodes'],profile['reprune_coordinate_io_ns']/1e9,
        100*profile['reprune_coordinate_io_ns']/profile['reprune_ns']),
        '整次合并提交坐标页读取 %d 次，涉及 %d 个唯一页，重复读比例 %.2f%%。'%(
        reads,profile['unique_coordinate_pages'],100*(1-profile['unique_coordinate_pages']/reads)),
        '私有上下文共载入向量 %d 次，涉及 %d 个唯一 ID，重复载入比例 %.2f%%。'%(
        vectors,profile['unique_coordinate_vectors'],100*(1-profile['unique_coordinate_vectors']/vectors)),
        '', '重复比例按（总次 − 唯一次数）/总次计算。读页为执行器向 SSD 提交的 8 KiB 请求，'
        '一次读页可携带两个双空间向量。计时包含插桩本身，独立结构审计和静态查询测量在计时外。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
