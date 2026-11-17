"""Generate observer-only copies of the frozen executor; never edit E1 code."""
from pathlib import Path
import hashlib
import json
import shutil

FILES=('update_store.cpp','update_store.h','dual_update.cpp','dual_update.h',
       'deg_prune.cpp','deg_prune.h','pruning_semantics.h','entry_frontier.h','in_degree_guard.h')


def once(text,old,new):
    assert text.count(old)==1,'instrumentation anchor count '+str(text.count(old))+': '+old[:80]
    return text.replace(old,new,1)


def prepare_no_stats(source,out,gate):
    """Diagnostic control only: retain all locks, degree changes and guards."""
    source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    for name in FILES:
        p=source/name
        if name in gate['sources']:assert hashlib.sha256(p.read_bytes()).hexdigest()==gate['sources'][name]
        shutil.copyfile(p,out/name)
    p=out/'update_store.cpp';s=p.read_text()
    for line,count in [('  uint64_t start = now_ns();\n',2),
        ('  stats.page_lock_wait_ns += now_ns() - start;\n',2),
        ('  ++stats.read_pages[f];\n',1),('    ++stats.read_pages[f];\n',1),('  ++stats.write_pages[f];\n',1)]:
        lines=s.splitlines(keepends=True);assert sum(v==line for v in lines)==count,(line,sum(v==line for v in lines));s=''.join(v for v in lines if v!=line)
    p.write_text(s)
    p=out/'dual_update.cpp';s=p.read_text()
    for old in ('  DegreeCounters before;\n','          before = degree_stats_;\n','      degree_stats_ = before;\n'):
        s=once(s,old,'')
    a=s.index('              ++degree_stats_.reverse_reprunes;')
    b=s.index('              replace_neighbors(page, target, protected_row.ids, undo);',a)
    s=s[:a]+s[b:];p.write_text(s)
    (out/'control-identity.json').write_text(json.dumps(dict(
        rule='remove only existing timing/IO counters and degree telemetry; keep degree lock scope, all ledger updates, sole-incoming protection and commit checks',
        sources={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in FILES}),indent=2)+'\n')


def prepare(source,out,gate):
    source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    for name in FILES:
        p=source/name
        if name in gate['sources']:assert hashlib.sha256(p.read_bytes()).hexdigest()==gate['sources'][name],name
        shutil.copyfile(p,out/name)
    p=out/'update_store.h';s=p.read_text()
    s=once(s,'  uint64_t page_lock_wait_ns = 0;', '''  uint64_t page_lock_wait_ns = 0;
  uint64_t phase_ns[7] = {}, insert_total_ns = 0, reprune_ns = 0;
  uint64_t read_io_ns[2] = {}, write_io_ns[2] = {};
  uint64_t ledger_wait_ns = 0, ledger_hold_ns = 0;
  uint64_t guard_ns = 0, degree_replace_ns = 0, invariant_ns = 0;
  uint64_t exact_pair_calls = 0, coordinate_calls = 0;
  uint64_t coordinate_cache_hits = 0, cache_hit_copy_bytes = 0;''');p.write_text(s)
    p=out/'update_store.cpp';s=p.read_text()
    s=once(s,'void read_locked(unsigned f, uint64_t page, Page &p) {','void read_locked(unsigned f, uint64_t page, Page &p, IOStats &stats) {')
    s=once(s,'    reader.read_fd(fd[f], requests, reader.get_ctx());','    auto io_start = now_ns();\n    reader.read_fd(fd[f], requests, reader.get_ctx());\n    stats.read_io_ns[f] += now_ns() - io_start;')
    s=once(s,'impl_->read_locked(f, page, p);','impl_->read_locked(f, page, p, stats);')
    s=once(s,'impl_->read_locked(f, page, private_page);','impl_->read_locked(f, page, private_page, stats);')
    s=once(s,'  impl_->reader.write_fd(impl_->fd[f], requests, impl_->reader.get_ctx());','  auto io_start = now_ns();\n  impl_->reader.write_fd(impl_->fd[f], requests, impl_->reader.get_ctx());\n  stats.write_io_ns[f] += now_ns() - io_start;');p.write_text(s)
    p=out/'dual_update.cpp';s=p.read_text()
    s=once(s,'#include <algorithm>','#include <algorithm>\n#include <chrono>')
    s=once(s,'namespace {\n','''namespace {
uint64_t perf_now() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
thread_local IOStats *perf_current = nullptr;
struct PerfCurrent {
  IOStats *old;
  explicit PerfCurrent(IOStats &s) : old(perf_current) { perf_current = &s; }
  ~PerfCurrent() { perf_current = old; }
};
struct PerfSpan {
  uint64_t *counter, start;
  explicit PerfSpan(uint64_t *c) : counter(c), start(c ? perf_now() : 0) {}
  ~PerfSpan() { if (counter) *counter += perf_now() - start; }
};
struct PerfFinish {
  IOStats &stats; uint64_t start; uint64_t &last;
  ~PerfFinish() { auto end=perf_now(); stats.phase_ns[6]+=end-last; stats.insert_total_ns+=end-start; }
};
''')
    s=once(s,'  DegreeChanges undo;\n  DegreeCounters before;','  DegreeChanges undo;\n  DegreeCounters before;\n  uint64_t hold_start = 0;')
    s=once(s,'          ledger.lock();','''          auto wait_start = perf_now();
          ledger.lock();
          stats.ledger_wait_ns += perf_now() - wait_start;
          hold_start = perf_now();''')
    s=once(s,'          change(buffer, undo);','''          change(buffer, undo);
          PerfSpan invariant_span(&stats.invariant_ns);''')
    s=once(s,'      rollback_degrees(undo);\n      degree_stats_ = before;','      rollback_degrees(undo);\n      degree_stats_ = before;\n      stats.ledger_hold_ns += perf_now() - hold_start;')
    s=once(s,'    throw;\n  }\n}\nvoid DualUpdate::replace_neighbors','    throw;\n  }\n  if (ledger.owns_lock()) stats.ledger_hold_ns += perf_now() - hold_start;\n}\nvoid DualUpdate::replace_neighbors')
    s=once(s,'                                   DegreeChanges &undo) {','                                   DegreeChanges &undo) {\n  PerfSpan accounting(perf_current ? &perf_current->degree_replace_ns : nullptr);')
    s=once(s,'std::vector<float> DualUpdate::coordinates(uint32_t id, Context &context) {','std::vector<float> DualUpdate::coordinates(uint32_t id, Context &context) {\n  ++context.stats.coordinate_calls;')
    s=once(s,'  if (saved != context.vectors.end())\n    return saved->second;','''  if (saved != context.vectors.end()) {
    ++context.stats.coordinate_cache_hits;
    context.stats.cache_hit_copy_bytes += uint64_t(d1_ + d2_) * 4;
    return saved->second;
  }''')
    s=once(s,'DistancePair DualUpdate::exact_pair(uint32_t a, uint32_t b, Context &context) {','DistancePair DualUpdate::exact_pair(uint32_t a, uint32_t b, Context &context) {\n  ++context.stats.exact_pair_calls;')
    s=once(s,'  finite_vector(x, d1_);\n  finite_vector(y, d2_);\n  std::shared_lock<std::shared_mutex> merging(merge_mutex_);','''  auto insert_start = perf_now(), phase_start = insert_start;
  PerfFinish finish{stats, insert_start, phase_start};
  PerfCurrent perf_scope(stats);
  auto phase = [&](unsigned i) { auto now = perf_now(); stats.phase_ns[i] += now - phase_start; phase_start = now; };
  finite_vector(x, d1_);
  finite_vector(y, d2_);
  std::shared_lock<std::shared_mutex> merging(merge_mutex_);''')
    s=once(s,'  auto candidates = insert_candidates(x, y, context);','  phase(0);\n  auto candidates = insert_candidates(x, y, context);\n  phase(1);')
    s=once(s,'  auto selected = select(id, std::move(ids), true, context);','  auto selected = select(id, std::move(ids), true, context);\n  phase(2);')
    s=once(s,'  std::map<uint64_t, std::vector<uint32_t> > groups;','  phase(3);\n  std::map<uint64_t, std::vector<uint32_t> > groups;')
    s=once(s,'              auto ranked = select(target, std::move(candidates),','              auto prune_start = perf_now();\n              auto ranked = select(target, std::move(candidates),')
    s=once(s,'              unsigned raw_degree = ranked.size();','              stats.reprune_ns += perf_now() - prune_start;\n              unsigned raw_degree = ranked.size();\n              auto guard_start = perf_now();')
    s=once(s,'                  [&](uint32_t v) { return !deleted_.test(v); }, id);','                  [&](uint32_t v) { return !deleted_.test(v); }, id);\n              stats.guard_ns += perf_now() - guard_start;')
    s=once(s,'  {\n    std::lock_guard<std::mutex> ledger(degree_mutex_);\n    DegreeChanges undo;','''  phase(4);
  {
    auto wait_start = perf_now();
    std::lock_guard<std::mutex> ledger(degree_mutex_);
    stats.ledger_wait_ns += perf_now() - wait_start;
    PerfSpan commit_hold(&stats.ledger_hold_ns);
    DegreeChanges undo;''')
    s=once(s,'  return id;\n}\n\nbool DualUpdate::erase','  phase(5);\n  return id;\n}\n\nbool DualUpdate::erase')
    # phase6 captures destruction of operation-private caches and buffers.
    p.write_text(s)
    (out/'instrumentation-identity.json').write_text(json.dumps(dict(
        original={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in FILES},
        instrumented={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in FILES},
        rule='observers only; original algorithm, lock scope, publication and I/O request shape unchanged',
        overlap='reverse includes full reprune; page waits, I/O and ledger spans overlap phases; do not sum overlay percentages'),indent=2)+'\n')


if __name__=='__main__':
    import sys
    prepare(Path(sys.argv[1]),Path(sys.argv[2]),json.loads(Path(sys.argv[3]).read_text()))
