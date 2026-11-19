#include "dual_update.h"
#include "entry_frontier.h"
#include "beam_search.h"
#include "profile_scope.h"
#include "ssd_index.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <thread>
#include <queue>
#include <stdexcept>
#include <unordered_set>

namespace hvs_update {
namespace {
void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
template <typename T> T get(std::istream &in) {
  T value;
  require(bool(in.read(reinterpret_cast<char *>(&value), sizeof(value))),
          "truncated update state");
  return value;
}
template <typename T> void put(std::ostream &out, T value) {
  require(
      bool(out.write(reinterpret_cast<const char *>(&value), sizeof(value))),
      "snapshot write");
}
uint64_t pages(uint64_t n, uint64_t per_page) {
  require(per_page > 0, "record exceeds an 8192-byte page");
  return (n + per_page - 1) / per_page;
}
float l2(const float *a, const float *b, unsigned n) {
  double squared = 0;
  for (unsigned j = 0; j < n; ++j) {
    double v = double(a[j]) - double(b[j]);
    squared += v * v;
  }
  return float(std::sqrt(squared));
}
// Reuse allocation only. No file/page identity or valid data survives Context.
struct PagePool {
  std::array<std::unique_ptr<Page>, 512> idle;
  size_t count = 0;
  std::unique_ptr<Page> take() {
    return count ? std::move(idle[--count])
                 : std::unique_ptr<Page>(new Page(false));
  }
  void give(std::unique_ptr<Page> p) noexcept {
    if (count < idle.size())
      idle[count++] = std::move(p);
  }
};
thread_local PagePool page_pool;
// Same batched PQ primitives and chunk accumulation order as Engine::traverse.
struct DualPQScorer {
  const uint8_t *codes1, *codes2;
  unsigned c1, c2;
  std::vector<float> t1, t2, e, g;
  std::vector<uint8_t> packed1, packed2;
  DualPQScorer(pipeann::FixedChunkPQTable<float> &p1,
               pipeann::FixedChunkPQTable<float> &p2, const uint8_t *a,
               const uint8_t *b, unsigned ca, unsigned cb, const float *x,
               const float *y)
      : codes1(a), codes2(b), c1(ca), c2(cb), t1(ca * 256), t2(cb * 256) {
    p1.populate_chunk_distances(x, t1.data());
    p2.populate_chunk_distances(y, t2.data());
  }
  std::vector<DistancePair> operator()(const std::vector<unsigned> &ids) {
    if (ids.empty())
      return {};
    packed1.resize(ids.size() * c1);
    packed2.resize(ids.size() * c2);
    e.resize(ids.size());
    g.resize(ids.size());
    aggregate_coords(ids.data(), ids.size(), codes1, c1, packed1.data());
    aggregate_coords(ids.data(), ids.size(), codes2, c2, packed2.data());
    pq_dist_lookup(packed1.data(), ids.size(), c1, t1.data(), e.data());
    pq_dist_lookup(packed2.data(), ids.size(), c2, t2.data(), g.data());
    std::vector<DistancePair> out;
    out.reserve(ids.size());
    for (size_t j = 0; j < ids.size(); ++j)
      out.emplace_back(std::sqrt(std::max(0.f, e[j])),
                       std::sqrt(std::max(0.f, g[j])));
    return out;
  }
};
void finite_vector(const float *x, unsigned n) {
  for (unsigned j = 0; j < n; ++j)
    require(std::isfinite(x[j]), "non-finite vector");
}
std::vector<pipeann::NeighborH>
pareto_order(std::vector<pipeann::NeighborH> pool, unsigned limit) {
  std::sort(pool.begin(), pool.end());
  std::vector<pipeann::NeighborH> ranked;
  int layer = 0;
  while (!pool.empty() && ranked.size() < limit) {
    float minimum = std::numeric_limits<float>::max();
    std::vector<pipeann::NeighborH> remaining;
    for (auto point : pool) {
      if (point.distance1 < minimum) {
        minimum = point.distance1;
        point.layer = layer;
        ranked.push_back(point);
      } else
        remaining.push_back(point);
    }
    pool.swap(remaining);
    ++layer;
  }
  if (ranked.size() > limit)
    ranked.resize(limit);
  return ranked;
}
} // namespace

// A context is owned by one query/update. Version checks prevent a pre-append
// private page from hiding a newly committed record on that same page.
struct DualUpdate::Context {
  struct Cached {
    std::unique_ptr<Page> page;
    uint64_t version;
  };
  std::unordered_map<uint64_t, Cached> cache[2];
  std::unordered_map<uint32_t, std::vector<float> > vectors;
  PageStore &store;
  IOStats &stats;
  explicit Context(PageStore &s, IOStats &io) : store(s), stats(io) {}
  ~Context() {
    for (auto &file : cache)
      for (auto &item : file)
        page_pool.give(std::move(item.second.page));
  }

  void prefetch(File file, std::vector<uint64_t> numbers) {
    unsigned f = static_cast<unsigned>(file);
    std::sort(numbers.begin(), numbers.end());
    numbers.erase(std::unique(numbers.begin(), numbers.end()), numbers.end());
    std::vector<uint64_t> missing;
    std::vector<std::unique_ptr<Page> > owners;
    std::vector<Page *> buffers;
    for (auto number : numbers) {
      auto i = cache[f].find(number);
      if (i != cache[f].end() &&
          i->second.version == store.version(file, number))
        continue;
      missing.push_back(number);
      owners.push_back(page_pool.take());
      buffers.push_back(owners.back().get());
    }
    auto versions = store.read_many(file, missing, buffers, stats);
    for (size_t i = 0; i < missing.size(); ++i)
      cache[f][missing[i]] = Cached{ std::move(owners[i]), versions[i] };
  }
  const unsigned char *read(File file, uint64_t number) {
    unsigned f = static_cast<unsigned>(file);
    auto old = cache[f].find(number);
    if (old != cache[f].end() &&
        old->second.version == store.version(file, number))
      return old->second.page->data();
    Cached fresh;
    fresh.page = page_pool.take();
    fresh.version = store.read(file, number, *fresh.page, stats);
    const auto *data = fresh.page->data();
    cache[f][number] = std::move(fresh);
    return data;
  }
};

Options DualUpdate::resolve(const std::string &dir, Options options) {
  std::ifstream in(dir + "/index.meta", std::ios::binary);
  require(bool(in), "missing compact index.meta");
  uint32_t n = get<uint32_t>(in);
  require(n > 0 && n < std::numeric_limits<uint32_t>::max() / 2,
          "invalid initial count");
  if (!options.max_points)
    options.max_points = n + (n + 1) / 2;
  require(options.max_points >= n &&
              options.max_points < std::numeric_limits<uint32_t>::max() - 1,
          "max-points below initial count or outside ID range");
  require(options.L_ins && options.C_ins && options.C_ins <= options.L_ins &&
              options.degree == 40 && options.base_graph == "B" &&
              options.consolidate_ratio > 0 && options.consolidate_ratio <= 1,
          "unsupported update options");
  return options;
}

DualUpdate::DualUpdate(const std::string &dir, Options options)
    : options_(resolve(dir, options)), directory_(dir),
      committed_(options_.max_points), deleted_(options_.max_points),
      accounted_(options_.max_points), next_id_(0),
      ledger_(options_.max_points) {
  std::ifstream meta(dir + "/index.meta", std::ios::binary);
  query_distance_.reset(
      pipeann::get_distance_function<float>(pipeann::Metric::L2));
  initial_ = get<uint32_t>(meta);
  d1_ = get<uint32_t>(meta);
  d2_ = get<uint32_t>(meta);
  require(get<uint32_t>(meta) == options_.degree, "degree mismatch");
  require(get<uint32_t>(meta) == 0, "update requires no-alpha metadata");
  record_bytes_ = 4 + 4 * options_.degree;
  topo_per_page_ = page_bytes / record_bytes_;
  require(get<uint64_t>(meta) == topo_per_page_, "topology packing mismatch");
  require(d1_ && d2_ && uint64_t(d1_ + d2_) * 4 <= page_bytes,
          "unsupported dimensions");
  coord_per_page_ = page_bytes / (uint64_t(d1_ + d2_) * 4);
  unsigned ne = get<uint32_t>(meta);
  require(ne > 0 && ne <= initial_, "invalid entry set");
  std::vector<uint32_t> initial_entries;
  for (unsigned j = 0; j < ne; ++j) {
    auto id = get<uint32_t>(meta);
    require(id < initial_, "entry outside initial count");
    initial_entries.push_back(id);
  }
  auto load_codes = [&](const std::string &file,
                        std::unique_ptr<uint8_t[]> &out, unsigned &chunks) {
    std::ifstream in(file, std::ios::binary);
    require(bool(in), "missing PQ codes");
    require(get<uint32_t>(in) == initial_, "PQ code count mismatch");
    chunks = get<uint32_t>(in);
    require(chunks > 0 && chunks <= 512, "invalid PQ chunks");
    out = std::make_unique<uint8_t[]>(uint64_t(options_.max_points) * chunks);
    require(bool(in.read(reinterpret_cast<char *>(out.get()),
                         uint64_t(initial_) * chunks)),
            "truncated PQ codes");
    require(in.peek() == EOF, "trailing PQ codes");
  };
  load_codes(dir + "/emb_pq_compressed.bin", codes1_, chunks1_);
  load_codes(dir + "/loc_pq_compressed.bin", codes2_, chunks2_);
  pq1_.load_pq_centroid_bin((dir + "/emb_pq_pivots.bin").c_str(), chunks1_);
  pq2_.load_pq_centroid_bin((dir + "/loc_pq_pivots.bin").c_str(), chunks2_);
  require(pq1_.get_dim() == d1_ && pq2_.get_dim() == d2_,
          "codebook dimension mismatch");
  topology_loc_ = std::make_unique<uint32_t[]>(options_.max_points);
  coordinate_loc_ = std::make_unique<uint32_t[]>(options_.max_points);
  tags_ = std::make_unique<uint64_t[]>(options_.max_points);
  consolidated_ = std::make_unique<std::atomic<bool> []>(options_.max_points);
  for (uint32_t id = 0; id < options_.max_points; ++id) {
    topology_loc_[id] = coordinate_loc_[id] = id;
    consolidated_[id].store(false, std::memory_order_relaxed);
  }
  tag_to_id_.reserve(options_.max_points);
  for (uint32_t id = 0; id < initial_; ++id) {
    tags_[id] = id;
    tag_to_id_.emplace(id, id);
    committed_.set(id);
    accounted_.set(id);
  }
  next_id_.store(initial_, std::memory_order_release);
  std::ifstream centers(dir + "/centers.bin", std::ios::binary);
  require(bool(centers) && get<uint32_t>(centers) == 1 &&
              get<uint32_t>(centers) == d1_ + d2_,
          "missing immutable build centers");
  center1_.resize(d1_);
  center2_.resize(d2_);
  require(
      bool(centers.read(reinterpret_cast<char *>(center1_.data()), d1_ * 4)) &&
          bool(centers.read(reinterpret_cast<char *>(center2_.data()),
                            d2_ * 4)) &&
          centers.peek() == EOF,
      "invalid center file");
  store_ = std::make_unique<PageStore>(
      dir + "/graph.pages", dir + "/coordinates.pages",
      pages(options_.max_points, topo_per_page_),
      pages(options_.max_points, coord_per_page_));
  IOStats stats;
  Context context(*store_, stats);
  for (auto id : initial_entries) {
    const auto &xy = coordinates(id, context);
    entry_frontier_.push_back({ id, l2(xy.data(), center1_.data(), d1_),
                                l2(xy.data() + d1_, center2_.data(), d2_) });
  }

  // One streaming topology scan; no full adjacency is retained by the index.
  for (uint64_t page = 0; page < pages(initial_, topo_per_page_); ++page) {
    Context scan(*store_, stats);
    for (uint32_t id = page * topo_per_page_;
         id < std::min<uint64_t>(initial_, (page + 1) * topo_per_page_); ++id)
      for (auto v : neighbors(id, scan)) {
        require(v < initial_, "initial indegree target out of bounds");
        ledger_.add_committed(v);
      }
  }
}
DualUpdate::~DualUpdate() = default;

struct DualUpdate::Pending {
  DualUpdate *owner;
  AtomicDegreeLedger::Transaction &transaction;
  DegreeCounters &counters;
};
thread_local DualUpdate::Pending *DualUpdate::pending_ = nullptr;
uint32_t DualUpdate::degree_count(uint32_t id) {
  return pending_ && pending_->owner == this ? pending_->transaction.count(id)
                                             : ledger_.count(id);
}
DegreeCounters &DualUpdate::transaction_counters() {
  require(pending_ && pending_->owner == this, "missing degree transaction");
  return pending_->counters;
}
std::vector<uint32_t> DualUpdate::indegree_snapshot() const {
  // Independent auditing requires quiescent writers, as in G1/checkpoints.
  return ledger_.snapshot(allocated());
}
std::vector<uint8_t> DualUpdate::positive_indegree_history() const {
  return ledger_.history(allocated());
}
DegreeCounters DualUpdate::degree_counters() const {
  std::lock_guard<std::mutex> lock(degree_stats_mutex_);
  return degree_stats_;
}
void DualUpdate::degree_delta(uint32_t id, int delta, DegreeChanges &undo) {
  require(id < allocated() && pending_ && pending_->owner == this,
          "degree transaction target");
  undo.push_back({ id, degree_count(id), ledger_.positive(id) });
  pending_->transaction.change(id, delta);
}
void DualUpdate::modify_topology(
    uint64_t page,
    const std::function<void(unsigned char *, DegreeChanges &)> &change,
    IOStats &stats) {
  for (unsigned attempt = 0;; ++attempt) {
    require(attempt < 1024,
            "optimistic topology transaction cannot preserve incoming edges");
    Page buffer;
    auto version = store_->read_for_update(File::topology, page, buffer, stats);
    AtomicDegreeLedger::Transaction transaction(ledger_);
    DegreeCounters counters{};
    DegreeChanges undo;
    Pending pending{ this, transaction, counters };
    auto previous = pending_;
    pending_ = &pending;
    try {
      change(buffer.data(), undo);
    }
    catch (const DegreeConflict &) {
      pending_ = previous;
      ++stats.optimistic_retries;
      std::this_thread::yield();
      continue;
    }
    catch (...) {
      pending_ = previous;
      throw;
    }
    pending_ = previous;
    bool complete = store_->compare_write(
        File::topology, page, version, buffer,
        [&] {
          return transaction.reserve([&](uint32_t v) {
            return !deleted_.test(v);
          });
        },
        [&] {
          transaction.commit();
          std::lock_guard<std::mutex> lock(degree_stats_mutex_);
#define ADD(field) degree_stats_.field += counters.field
          ADD(reverse_reprunes);
          ADD(full_reprunes);
          ADD(protection_events);
          ADD(protected_edges);
          ADD(protected_old_edges);
          ADD(protected_new_edges);
          ADD(reverse_capacity_conflicts);
          ADD(full_before_edges);
          ADD(full_after_edges);
#undef ADD
          for (unsigned d = 0; d <= 40; ++d) {
            degree_stats_.full_after_degree[d] += counters.full_after_degree[d];
            degree_stats_.full_raw_after_degree[d] +=
                counters.full_raw_after_degree[d];
          }
        },
        stats);
    if (complete)
      return;
    ++stats.optimistic_retries;
    std::this_thread::yield();
  }
}
void DualUpdate::replace_neighbors(unsigned char *page, uint32_t id,
                                   const std::vector<uint32_t> &replacement,
                                   DegreeChanges &undo) {
  auto p = page + topology_loc_[id] % topo_per_page_ * record_bytes_;
  uint32_t count;
  std::memcpy(&count, p, 4);
  require(count <= options_.degree, "old indegree row corrupt");
  std::vector<uint32_t> old(count);
  std::memcpy(old.data(), p + 4, count * 4);
  if (committed_.test(id)) {
    while (!accounted_.test(id))
      std::this_thread::yield();
    for (auto v : old)
      if (std::find(replacement.begin(), replacement.end(), v) ==
          replacement.end())
        degree_delta(v, -1, undo);
    for (auto v : replacement)
      if (std::find(old.begin(), old.end(), v) == old.end())
        degree_delta(v, 1, undo);
  }
  put_neighbors(page, id, replacement);
}

std::vector<uint32_t> DualUpdate::neighbors(uint32_t id, Context &context) {
  require(id < allocated(), "topology ID outside allocated range");
  auto loc = topology_loc_[id];
  auto p = context.read(File::topology, loc / topo_per_page_) +
           loc % topo_per_page_ * record_bytes_;
  uint32_t n;
  std::memcpy(&n, p, 4);
  require(n <= options_.degree, "topology degree corrupt");
  std::vector<uint32_t> out(n);
  std::memcpy(out.data(), p + 4, n * 4);
  for (auto v : out)
    require(v < allocated(), "topology references unallocated ID");
  return out;
}
std::vector<uint32_t> DualUpdate::neighbors(uint32_t id, IOStats &stats) {
  Context context(*store_, stats);
  return neighbors(id, context);
}
std::vector<std::vector<uint32_t> > DualUpdate::audit_topology(IOStats &stats) {
  const uint32_t n = allocated();
  std::vector<std::vector<uint32_t> > graph(n);
  for (uint64_t page = 0; page < pages(n, topo_per_page_); ++page) {
    Context context(*store_, stats);
    for (uint32_t id = page * topo_per_page_;
         id < std::min<uint64_t>(n, (page + 1) * topo_per_page_); ++id)
      if (committed_.test(id))
        graph[id] = neighbors(id, context);
  }
  return graph;
}
const std::vector<float> &DualUpdate::coordinates(uint32_t id,
                                                  Context &context) {
#ifdef HVS_PROFILE_PHASES
  ++context.stats.coordinate_calls;
#endif
  require(id < allocated(), "coordinate ID outside allocated range");
  auto saved = context.vectors.find(id);
  if (saved != context.vectors.end()) {
#ifdef HVS_PROFILE_PHASES
    ++context.stats.coordinate_cache_hits;
#endif
    return saved->second;
  }
  auto loc = coordinate_loc_[id];
  auto p = context.read(File::coordinates, loc / coord_per_page_) +
           loc % coord_per_page_ * (d1_ + d2_) * 4;
  std::vector<float> xy(d1_ + d2_);
  std::memcpy(xy.data(), p, xy.size() * 4);
#ifdef HVS_PROFILE_PHASES
  context.stats.coordinate_materialized_bytes += xy.size() * sizeof(float);
#endif
  return context.vectors.emplace(id, std::move(xy)).first->second;
}
std::vector<float> DualUpdate::copy_coordinates_for_audit(uint32_t id,
                                                          IOStats &stats) {
  Context context(*store_, stats);
  return coordinates(id, context);
}
std::vector<uint32_t> DualUpdate::entries() const {
  std::shared_lock<std::shared_mutex> lock(entry_mutex_);
  std::vector<uint32_t> ids;
  for (auto v : entry_frontier_)
    ids.push_back(v.id);
  return ids;
}
bool DualUpdate::consolidated(uint32_t id) const {
  require(id < options_.max_points, "consolidated ID bounds");
  return consolidated_[id].load(std::memory_order_acquire);
}
DistancePair DualUpdate::pq_pair(uint32_t a, uint32_t b) {
  float e, g;
  pq1_.compute_distances_alltoall(codes1_.get() + uint64_t(a) * chunks1_,
                                  codes1_.get() + uint64_t(b) * chunks1_, &e,
                                  1);
  pq2_.compute_distances_alltoall(codes2_.get() + uint64_t(a) * chunks2_,
                                  codes2_.get() + uint64_t(b) * chunks2_, &g,
                                  1);
  return { std::sqrt(std::max(0.f, e)), std::sqrt(std::max(0.f, g)) };
}
DistancePair DualUpdate::exact_pair(uint32_t a, uint32_t b, Context &context) {
  const auto &x = coordinates(a, context), &y = coordinates(b, context);
  return { l2(x.data(), y.data(), d1_),
           l2(x.data() + d1_, y.data() + d1_, d2_) };
}
std::vector<uint32_t> DualUpdate::select(uint32_t source,
                                         std::vector<uint32_t> ids, bool exact,
                                         Context &context) {
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  ids.erase(std::remove(ids.begin(), ids.end(), source), ids.end());
  if (exact) {
    std::vector<uint64_t> pages;
    pages.push_back(coordinate_loc_[source] / coord_per_page_);
    for (auto v : ids)
      pages.push_back(coordinate_loc_[v] / coord_per_page_);
    context.prefetch(File::coordinates, std::move(pages));
  }
  PairDistance distance = [&](uint32_t a, uint32_t b) {
    return exact ? exact_pair(a, b, context) : pq_pair(a, b);
  };
  std::vector<pipeann::NeighborH> pool;
  for (auto id : ids) {
    auto d = distance(source, id);
    pool.emplace_back(id, d.first, d.second, true, 0);
  }
  return prune(std::move(pool), options_.degree, distance);
}

std::vector<pipeann::NeighborH>
DualUpdate::insert_candidates(const float *x, const float *y,
                              Context &context) {
  DualPQScorer score(pq1_, pq2_, codes1_.get(), codes2_.get(), chunks1_,
                     chunks2_, x, y);
  tsl::robin_set<uint64_t> seen;
  std::vector<pipeann::NeighborH> pool;
  auto add = [&](const std::vector<unsigned> &ids) {
    std::vector<unsigned> fresh;
    for (auto id : ids)
      if (id < allocated() && committed_.test(id) && seen.insert(id).second)
        fresh.push_back(id);
    auto ds = score(fresh);
    for (size_t i = 0; i < fresh.size(); ++i)
      pool.emplace_back(fresh[i], ds[i].first, ds[i].second, true, 0);
  };
  add(entries());
  pool = pareto_order(std::move(pool), options_.L_ins);
  for (;;) {
    // Insertion has no query alpha: retain DEG Pareto ranking, but reuse the
    // query engine's beam4 frontier extraction and batched scoring / page I/O.
    auto frontier = take_beam(pool, 0, pool.size());
    if (frontier.empty())
      break;
    std::vector<uint64_t> batch;
    for (auto id : frontier)
      batch.push_back(topology_loc_[id] / topo_per_page_);
    context.prefetch(File::topology, std::move(batch));
    for (auto id : frontier)
      add(neighbors(id, context));
    pool = pareto_order(std::move(pool), options_.L_ins);
  }
  return pool;
}

std::vector<QueryResult> DualUpdate::search(const float *x, const float *y,
                                            float alpha, unsigned k, unsigned L,
                                            unsigned rerank, QueryWork &work) {
  require(std::isfinite(alpha) && alpha >= 0 && alpha <= 1 && k && L >= k &&
              rerank >= k && rerank <= L,
          "invalid query budget/alpha");
  finite_vector(x, d1_);
  finite_vector(y, d2_);
  Context context(*store_, work.io);
  DualPQScorer score(pq1_, pq2_, codes1_.get(), codes2_.get(), chunks1_,
                     chunks2_, x, y);
  auto pool = query_beam(
      entries(), L,
      [&](const std::vector<unsigned> &ids) {
        auto ds = score(ids);
        std::vector<float> fused;
        fused.reserve(ids.size());
        for (auto d : ds)
          fused.push_back(alpha * d.first + (1 - alpha) * d.second);
        work.pq_evaluations += ids.size();
        return fused;
      },
      [&](const std::vector<unsigned> &ids) {
        std::vector<uint64_t> pages;
        for (auto id : ids)
          pages.push_back(topology_loc_[id] / topo_per_page_);
        context.prefetch(File::topology, std::move(pages));
      },
      [&](unsigned id) { return neighbors(id, context); },
      [&](unsigned id) { return id < allocated() && committed_.test(id); },
      work.expanded);
  std::vector<uint32_t> ids;
  for (const auto &v : pool)
    if (!deleted_.test(v.id) && ids.size() < rerank)
      ids.push_back(v.id);
  // Original Engine reranks in ID order with the upstream SIMD L2 function.
  std::sort(ids.begin(), ids.end());
  std::vector<uint64_t> batch;
  for (auto id : ids)
    batch.push_back(coordinate_loc_[id] / coord_per_page_);
  context.prefetch(File::coordinates, std::move(batch));
  std::vector<QueryResult> result;
  for (auto id : ids) {
    auto loc = coordinate_loc_[id];
    const float *xy = reinterpret_cast<const float *>(
        context.read(File::coordinates, loc / coord_per_page_) +
        loc % coord_per_page_ * (d1_ + d2_) * 4);
    float a = query_distance_->compare(x, xy, d1_),
          b = query_distance_->compare(y, xy + d1_, d2_);
    result.push_back(
        { id, tags_[id], alpha * std::sqrt(std::max(0.f, a)) +
                             (1 - alpha) * std::sqrt(std::max(0.f, b)) });
    ++work.exact_evaluations;
  }
  std::sort(result.begin(), result.end(),
            [](const QueryResult &a, const QueryResult &b) {
    return a.distance < b.distance || (a.distance == b.distance && a.id < b.id);
  });
  // Linearize final result filtering with delete(tag), including deletions that
  // occurred during rerank. Return length is explicit when fewer than K
  // survive.
  std::lock_guard<std::mutex> guard(deletion_mutex_);
  result.erase(
      std::remove_if(result.begin(), result.end(), [&](const QueryResult &v) {
        return deleted_.test(v.id) || !committed_.test(v.id);
      }),
      result.end());
  if (result.size() > k)
    result.resize(k);
  return result;
}

void DualUpdate::put_neighbors(unsigned char *page, uint32_t id,
                               const std::vector<uint32_t> &ids) {
  require(ids.size() <= options_.degree, "degree exceeds compact record");
  std::unordered_set<uint32_t> seen;
  for (auto v : ids)
    require(v != id && v < allocated() && seen.insert(v).second,
            "self/duplicate/invalid edge");
  auto p = page + topology_loc_[id] % topo_per_page_ * record_bytes_;
  std::memset(p, 0, record_bytes_);
  auto n = uint32_t(ids.size());
  std::memcpy(p, &n, 4);
  std::memcpy(p + 4, ids.data(), ids.size() * 4);
}

void DualUpdate::update_entries(uint32_t id, const float *x, const float *y,
                                bool publish,
                                const std::vector<uint32_t> *outgoing) {
  Entry added{ id, l2(x, center1_.data(), d1_), l2(y, center2_.data(), d2_) };
  std::unique_lock<std::shared_mutex> guard(entry_mutex_);
  auto candidates = entry_frontier_;
  candidates.erase(
      std::remove_if(candidates.begin(), candidates.end(),
                     [&](const Entry &v) { return deleted_.test(v.id); }),
      candidates.end());
  candidates.push_back(added);
  entry_frontier_ = builder_entry_frontier(std::move(candidates));
  // Publish before releasing the frontier lock, so readers never observe an
  // entry set replaced by a still-uncommitted dominating insertion.
  if (publish) {
    committed_.set(id);
    // No throwing work after publication. Counts understate the graph until
    // this short tail finishes; other sources cannot spend an unseen addition.
    for (auto v : *outgoing)
      ledger_.add_committed(v);
    accounted_.set(id);
  }
}

uint32_t DualUpdate::insert(const float *x, const float *y, uint64_t tag,
                            IOStats &stats) {
#ifdef HVS_PROFILE_PHASES
  auto insert_start = profile_now(), phase_start = insert_start;
  ProfileFinish finish{ stats, insert_start, phase_start };
  auto phase = [&](unsigned i) {
    auto now = profile_now();
    stats.phase_ns[i] += now - phase_start;
    phase_start = now;
  };
#endif
  finite_vector(x, d1_);
  finite_vector(y, d2_);
  std::shared_lock<std::shared_mutex> merging(merge_mutex_);
  uint32_t id;
  {
    std::lock_guard<std::mutex> guard(tag_mutex_);
    require(tag_to_id_.find(tag) == tag_to_id_.end(),
            "tag already allocated (IDs are never reused)");
    id = next_id_.load(std::memory_order_relaxed);
    require(id < options_.max_points, "max-points exhausted");
    tags_[id] = tag;
    tag_to_id_.emplace(tag, id);
    next_id_.store(id + 1, std::memory_order_release);
  }
  // A failed insert consumes its ID and remains uncommitted. No old ID or
  // physical slot is recycled, including after consolidation.
  Context context(*store_, stats);
  auto loc = coordinate_loc_[id];
  store_->modify(File::coordinates, loc / coord_per_page_,
                 [&](unsigned char *page) {
                   auto p = page + loc % coord_per_page_ * (d1_ + d2_) * 4;
                   std::memcpy(p, x, d1_ * 4);
                   std::memcpy(p + d1_ * 4, y, d2_ * 4);
                 },
                 stats);
  pq1_.deflate_vec(x, codes1_.get() + uint64_t(id) * chunks1_);
  pq2_.deflate_vec(y, codes2_.get() + uint64_t(id) * chunks2_);
#ifdef HVS_PROFILE_PHASES
  phase(0);
#endif
  auto candidates = insert_candidates(x, y, context);
#ifdef HVS_PROFILE_PHASES
  phase(1);
#endif
  std::vector<uint32_t> ids;
  for (const auto &v : candidates)
    if (!deleted_.test(v.id) && ids.size() < options_.C_ins)
      ids.push_back(v.id);
  auto selected = select(id, std::move(ids), true, context);
#ifdef HVS_PROFILE_PHASES
  phase(2);
#endif
  modify_topology(topology_loc_[id] / topo_per_page_,
                  [&](unsigned char *page, DegreeChanges &undo) {
                    replace_neighbors(page, id, selected, undo);
                  },
                  stats);

#ifdef HVS_PROFILE_PHASES
  phase(3);
#endif
  std::map<uint64_t, std::vector<uint32_t> > groups;
  for (auto v : selected)
    groups[topology_loc_[v] / topo_per_page_].push_back(v);
  for (const auto &group : groups) {
    modify_topology(
        group.first,
        [&](unsigned char *page, DegreeChanges &undo) {
          for (auto target : group.second) {
            while (committed_.test(target) && !accounted_.test(target))
              std::this_thread::yield();
            auto p =
                page + topology_loc_[target] % topo_per_page_ * record_bytes_;
            uint32_t n;
            std::memcpy(&n, p, 4);
            require(n <= options_.degree, "reverse edge degree corrupt");
            std::vector<uint32_t> old(n);
            std::memcpy(old.data(), p + 4, n * 4);
            if (std::find(old.begin(), old.end(), id) != old.end())
              continue;
            auto candidates = old;
            candidates.push_back(id);
            if (options_.always_reprune || n == options_.degree) {
#ifdef HVS_PROFILE_PHASES
              auto reprune_tick = profile_now();
#endif
              auto ranked = select(target, std::move(candidates),
                                   options_.exact_reprune, context);
#ifdef HVS_PROFILE_PHASES
              stats.reprune_ns += profile_now() - reprune_tick;
#endif
              unsigned raw_degree = ranked.size();
              auto protected_row = protect_last_incoming(
                  old, std::move(ranked), options_.degree,
                  [&](uint32_t v) { return degree_count(v); },
                  [&](uint32_t v) { return !deleted_.test(v); }, id);
              ++transaction_counters().reverse_reprunes;
              auto restored =
                  protected_row.old_restored + protected_row.new_restored;
              transaction_counters().protection_events += restored > 0;
              transaction_counters().protected_edges += restored;
              transaction_counters().protected_old_edges +=
                  protected_row.old_restored;
              transaction_counters().protected_new_edges +=
                  protected_row.new_restored;
              transaction_counters().reverse_capacity_conflicts +=
                  protected_row.capacity_conflict;
              if (n == options_.degree) {
                ++transaction_counters().full_reprunes;
                transaction_counters().full_before_edges += n;
                transaction_counters().full_after_edges +=
                    protected_row.ids.size();
                ++transaction_counters().full_after_degree
                      [protected_row.ids.size()];
                ++transaction_counters().full_raw_after_degree[raw_degree];
              }
              replace_neighbors(page, target, protected_row.ids, undo);
            } else
              replace_neighbors(page, target, candidates, undo);
          }
        },
        stats);
  }
#ifdef HVS_PROFILE_PHASES
  phase(4);
#endif
  require(ledger_.count(id) > 0, "insert cannot commit: no incoming edge after "
                                 "protected reverse insertion");
  update_entries(id, x, y, true, &selected);
#ifdef HVS_PROFILE_PHASES
  phase(5);
#endif
  return id;
}

bool DualUpdate::erase(uint64_t tag) {
  uint32_t id;
  {
    std::lock_guard<std::mutex> guard(tag_mutex_);
    auto found = tag_to_id_.find(tag);
    if (found == tag_to_id_.end())
      return false;
    id = found->second;
  }
  std::lock_guard<std::mutex> guard(deletion_mutex_);
  if (!committed_.test(id) || deleted_.test(id))
    return false;
  deleted_.set(id);
  marked_count_.fetch_add(1, std::memory_order_release);
  return true;
}
bool DualUpdate::consolidation_due() const {
  auto marked = marked_count_.load(std::memory_order_acquire);
  auto merged = merged_count_.load(std::memory_order_acquire);
  uint32_t live = allocated() - marked;
  return double(marked - merged) >=
         std::max(1., double(live) * options_.consolidate_ratio);
}
void DualUpdate::consolidate(IOStats &stats) {
  std::unique_lock<std::shared_mutex> merging(merge_mutex_);
  std::unique_ptr<uint64_t[]> frozen;
  uint32_t frozen_count;
  {
    std::lock_guard<std::mutex> guard(deletion_mutex_);
    frozen = deleted_.snapshot();
    frozen_count = marked_count_.load(std::memory_order_relaxed);
  }
  if (frozen_count == merged_count_.load(std::memory_order_acquire))
    return;
  const uint32_t n = allocated();
  auto dead = [&](uint32_t id) {
    return bool((frozen[id / 64] >> (id % 64)) & 1);
  };
  auto live = [&](uint32_t id) { return committed_.test(id) && !dead(id); };

  // Insertions and other consolidations are paused by merge_mutex_. Readers
  // and new delete marks continue; new marks are outside this frozen batch.
  const auto original = audit_topology(stats);
  const auto initial_degree = ledger_.snapshot(n);
  auto final_degree = initial_degree;
  const auto initial_entries = entries();
  const size_t page_count = pages(n, topo_per_page_);
  std::vector<std::vector<uint32_t> > planned(n), bridge(n);
  std::vector<uint8_t> changed(n), bridged(n);
  auto final_row = [&](uint32_t id)->const std::vector<uint32_t> & {
    return changed[id] ? planned[id] : original[id];
  };
  auto intermediate = [&](uint32_t id)->const std::vector<uint32_t> & {
    return bridged[id] ? bridge[id] : original[id];
  };
  auto account = [](auto &counts, const auto &from, const auto &to) {
    for (auto id : from) {
      require(counts[id] > 0, "consolidation planning indegree underflow");
      --counts[id];
    }
    for (auto id : to)
      ++counts[id];
  };
  for (uint64_t page = 0; page < page_count; ++page) {
    Context context(*store_, stats);
    for (uint32_t id = page * topo_per_page_;
         id < std::min<uint64_t>(n, (page + 1) * topo_per_page_); ++id) {
      if (!committed_.test(id))
        continue;
      if (live(id)) {
        if (std::none_of(original[id].begin(), original[id].end(), dead))
          continue;
        std::vector<uint32_t> candidates;
        for (auto v : original[id]) {
          if (live(v))
            candidates.push_back(v);
          else if (dead(v))
            for (auto next : original[v])
              if (live(next))
                candidates.push_back(next);
        }
        planned[id] =
            select(id, std::move(candidates), options_.exact_reprune, context);
      }
      if (planned[id] != original[id]) {
        changed[id] = true;
        account(final_degree, original[id], planned[id]);
      }
    }
  }

  std::vector<uint32_t> orphans;
  std::vector<uint8_t> orphan(n);
  for (uint32_t id = 0; id < n; ++id)
    if (live(id) && initial_degree[id] && !final_degree[id]) {
      orphans.push_back(id);
      orphan[id] = true;
    }
  // For a lost target, index only the deleted sources that used to reach it
  // and the live predecessors of those sources.
  std::unordered_map<uint32_t, std::vector<uint32_t> > deleted_sources,
      live_predecessors;
  for (uint32_t id = 0; id < n; ++id)
    if (dead(id))
      for (auto target : original[id])
        if (orphan[target]) {
          deleted_sources[target].push_back(id);
          live_predecessors.try_emplace(id);
        }
  if (!live_predecessors.empty())
    for (uint32_t id = 0; id < n; ++id)
      if (live(id))
        for (auto target : original[id]) {
          auto found = live_predecessors.find(target);
          if (found != live_predecessors.end())
            found->second.push_back(id);
        }

  std::vector<Entry> next_entries;
  const bool rebuild =
      std::any_of(initial_entries.begin(), initial_entries.end(), dead);
  std::vector<uint32_t> final_entries = initial_entries;
  if (rebuild) {
    for (uint32_t id = 0; id < n; ++id) {
      if (!live(id))
        continue;
      Context context(*store_, stats);
      const auto &xy = coordinates(id, context);
      next_entries.push_back({ id, l2(xy.data(), center1_.data(), d1_),
                               l2(xy.data() + d1_, center2_.data(), d2_) });
      next_entries = builder_entry_frontier(std::move(next_entries));
    }
    final_entries.clear();
    for (const auto &entry : next_entries)
      final_entries.push_back(entry.id);
  }

  auto bridge_degree = initial_degree;
  auto assign_repair = [&](uint32_t donor, uint32_t target, bool allow_swap) {
    auto staged = intermediate(donor);
    auto final = final_row(donor);
    // Preserve existing edges through the first pass. To add a new edge to
    // a full row, reuse a frozen-dead slot so live indegrees cannot decrease.
    if (std::find(staged.begin(), staged.end(), target) == staged.end()) {
      if (staged.size() == options_.degree) {
        if (!allow_swap)
          return false;
        auto edge = std::find_if(staged.rbegin(), staged.rend(), dead);
        if (edge == staged.rend())
          return false;
        *edge = target;
      } else {
        staged.push_back(target);
      }
    }
    if (final.size() == options_.degree) {
      if (!allow_swap)
        return false;
      auto edge =
          std::find_if(final.rbegin(), final.rend(),
                       [&](uint32_t id) { return final_degree[id] > 1; });
      if (edge == final.rend())
        return false;
      *edge = target;
    } else {
      final.push_back(target);
    }
    if (staged != intermediate(donor)) {
      account(bridge_degree, intermediate(donor), staged);
      bridge[donor] = std::move(staged);
      bridged[donor] = true;
    }
    account(final_degree, final_row(donor), final);
    planned[donor] = std::move(final);
    changed[donor] = true;
    return true;
  };
  uint64_t transferred = 0, searched = 0;
  for (auto target : orphans) {
    for (auto source : deleted_sources[target]) {
      const auto &donors = live_predecessors.at(source);
      for (auto p : donors)
        if (p != target && assign_repair(p, target, true)) {
          ++transferred;
          break;
        }
      if (final_degree[target])
        break;
    }
  }

  auto reachable = [&](const auto &roots, const auto &row, bool only_live) {
    std::vector<uint8_t> seen(n);
    std::vector<uint32_t> fifo;
    auto enqueue = [&](uint32_t id) {
      if (!seen[id] && (only_live ? live(id) : committed_.test(id))) {
        seen[id] = true;
        fifo.push_back(id);
      }
    };
    for (auto id : roots)
      enqueue(id);
    for (size_t i = 0; i < fifo.size(); ++i)
      for (auto next : row(fifo[i]))
        enqueue(next);
    return seen;
  };
  const auto before_reachable =
      reachable(initial_entries,
                [&](uint32_t id)->const auto & { return original[id]; }, false);
  // Consolidation has no user alpha. Navigate with alpha=.5 and choose the
  // closest exact rooted-L2 donor among the eligible search results.
  for (auto target : orphans) {
    if (final_degree[target])
      continue;
    auto has_slot = [&](uint32_t id) {
      const auto &row = intermediate(id);
      return id != target && final_row(id).size() < options_.degree &&
             (row.size() < options_.degree ||
              std::find(row.begin(), row.end(), target) != row.end());
    };
    std::vector<uint32_t> eligible;
    Context context(*store_, stats);
    const auto &xy = coordinates(target, context);
    DualPQScorer score(pq1_, pq2_, codes1_.get(), codes2_.get(), chunks1_,
                       chunks2_, xy.data(), xy.data() + d1_);
    uint64_t expanded = 0;
    query_beam(final_entries,
               std::max<unsigned>(options_.L_ins, final_entries.size()),
               [&](const std::vector<unsigned> &ids) {
                 auto ds = score(ids);
                 std::vector<float> fused;
                 fused.reserve(ids.size());
                 for (size_t i = 0; i < ids.size(); ++i) {
                   if (has_slot(ids[i]))
                     eligible.push_back(ids[i]);
                   fused.push_back(.5f * (ds[i].first + ds[i].second));
                 }
                 return fused;
               },
               [](const auto &) {}, final_row, live, expanded);
    if (eligible.empty()) {
      // The beam is bounded. Search the reachable component when it found
      // no vacancy, then use the exact nearest available source.
      auto seen = reachable(final_entries, final_row, true);
      for (uint32_t id = 0; id < n; ++id)
        if (seen[id] && has_slot(id))
          eligible.push_back(id);
    }
    uint32_t donor = n;
    double best = std::numeric_limits<double>::infinity();
    for (auto id : eligible) {
      // Keep only one donor's coordinates live in the exhaustive fallback.
      Context pair_context(*store_, stats);
      const auto &p = coordinates(id, pair_context);
      double distance = double(l2(xy.data(), p.data(), d1_)) +
                        double(l2(xy.data() + d1_, p.data() + d1_, d2_));
      if (distance < best || (distance == best && id < donor)) {
        best = distance;
        donor = id;
      }
    }
    require(donor < n && assign_repair(donor, target, false),
            "merge plan has no reachable spare source for last incoming edge");
    ++searched;
  }

  // Verify the order before writing anything. Live replacements can depend on
  // an incoming edge supplied by another live page even when total final
  // indegrees are positive. Wake a blocked page only when its blocking count
  // increases. Dead rows are cleared after all live replacements.
  using Rows = std::vector<uint32_t>;
  std::vector<Rows> bridge_pages(page_count), live_pages(page_count),
      dead_pages(page_count);
  for (uint32_t id = 0; id < n; ++id) {
    auto page = topology_loc_[id] / topo_per_page_;
    if (bridged[id])
      bridge_pages[page].push_back(id);
    if (changed[id] && final_row(id) != intermediate(id))
      (dead(id) ? dead_pages : live_pages)[page].push_back(id);
  }
  std::vector<std::vector<std::pair<uint32_t, int> > > changes(page_count);
  std::queue<uint64_t> ready;
  for (uint64_t page = 0; page < page_count; ++page) {
    if (live_pages[page].empty())
      continue;
    std::map<uint32_t, int> net;
    for (auto id : live_pages[page]) {
      for (auto v : intermediate(id))
        --net[v];
      for (auto v : final_row(id))
        ++net[v];
    }
    for (auto item : net)
      if (item.second)
        changes[page].push_back(item);
    ready.push(page);
  }
  std::unordered_map<uint32_t, std::vector<uint64_t> > blocked;
  std::vector<uint64_t> order;
  size_t scheduled = ready.size();
  while (!ready.empty()) {
    auto page = ready.front();
    ready.pop();
    auto &delta = changes[page];
    auto bad = std::find_if(delta.begin(), delta.end(), [&](auto v) {
      return live(v.first) && v.second < 0 &&
             int64_t(bridge_degree[v.first]) + v.second <= 0;
    });
    if (bad != delta.end()) {
      blocked[bad->first].push_back(page);
      continue;
    }
    order.push_back(page);
    for (auto v : delta) {
      auto next = int64_t(bridge_degree[v.first]) + v.second;
      require(next >= 0, "merge schedule indegree underflow");
      bridge_degree[v.first] = uint32_t(next);
      if (v.second > 0) {
        auto pending = blocked.find(v.first);
        if (pending != blocked.end()) {
          for (auto p : pending->second)
            ready.push(p);
          blocked.erase(pending);
        }
      }
    }
  }
  require(order.size() == scheduled,
          "merge plan has cyclic page indegree dependencies before writes");
  for (const auto &ids : dead_pages)
    for (auto id : ids)
      account(bridge_degree, intermediate(id), final_row(id));
  require(bridge_degree == final_degree, "merge plan and schedule disagree");

  const auto after_reachable = reachable(final_entries, final_row, true);
  uint64_t live_count = 0, live_edges = 0, unreachable_before = 0,
           unreachable_after = 0;
  for (uint32_t id = 0; id < n; ++id)
    if (live(id)) {
      require(!initial_degree[id] || final_degree[id],
              "merge plan lost a live incoming edge before writes");
      ++live_count;
      live_edges += final_row(id).size();
      unreachable_before += !before_reachable[id];
      unreachable_after += !after_reachable[id];
    }
  auto write_page = [&](uint64_t page, const Rows &ids, const auto &row) {
    if (ids.empty())
      return;
    modify_topology(page, [&](unsigned char *buffer, DegreeChanges &undo) {
                            for (auto id : ids)
                              replace_neighbors(buffer, id, row(id), undo);
                          },
                    stats);
  };
  for (uint64_t page = 0; page < page_count; ++page)
    write_page(page, bridge_pages[page], intermediate);
  for (auto page : order)
    write_page(page, live_pages[page], final_row);
  if (rebuild) {
    std::unique_lock<std::shared_mutex> guard(entry_mutex_);
    entry_frontier_.swap(next_entries);
  }
  for (uint64_t page = 0; page < page_count; ++page)
    write_page(page, dead_pages[page], final_row);
  for (uint32_t id = 0; id < n; ++id)
    if (dead(id))
      consolidated_[id].store(true, std::memory_order_release);
  merged_count_.store(frozen_count, std::memory_order_release);
  stats.merge_transferred_edges = transferred;
  stats.merge_search_edges = searched;
  stats.merge_live_nodes = live_count;
  stats.merge_live_edges = live_edges;
  stats.merge_unreachable_before = unreachable_before;
  stats.merge_unreachable_after = unreachable_after;
}

void DualUpdate::snapshot(const std::string &path) {
  std::unique_lock<std::shared_mutex> merging(merge_mutex_);
  std::lock_guard<std::mutex> deleting(deletion_mutex_);
  std::shared_lock<std::shared_mutex> entry(entry_mutex_);
  store_->sync();
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  require(bool(out), "create normal-exit snapshot");
  out.write("HVSUPD1", 8);
  put(out, initial_);
  put(out, allocated());
  put(out, options_.max_points);
  put(out, d1_);
  put(out, d2_);
  put(out, chunks1_);
  put(out, chunks2_);
  auto committed = committed_.snapshot(), deleted = deleted_.snapshot();
  for (uint64_t j = 0; j < committed_.words(); ++j) {
    put(out, committed[j]);
    put(out, deleted[j]);
  }
  for (uint32_t id = 0; id < allocated(); ++id) {
    put(out, tags_[id]);
    put(out, uint8_t(consolidated(id)));
    out.write(
        reinterpret_cast<const char *>(codes1_.get() + uint64_t(id) * chunks1_),
        chunks1_);
    out.write(
        reinterpret_cast<const char *>(codes2_.get() + uint64_t(id) * chunks2_),
        chunks2_);
  }
  put(out, uint32_t(entry_frontier_.size()));
  for (auto e : entry_frontier_) {
    put(out, e.id);
    put(out, e.x);
    put(out, e.y);
  }
  out.write(reinterpret_cast<const char *>(center1_.data()), d1_ * 4);
  out.write(reinterpret_cast<const char *>(center2_.data()), d2_ * 4);
  require(bool(out), "snapshot incomplete");
}
} // namespace hvs_update
