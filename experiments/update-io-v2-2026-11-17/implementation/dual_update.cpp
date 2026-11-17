#include "dual_update.h"
#include "entry_frontier.h"
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
void require(bool value, const std::string &message) {
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
      owners.emplace_back(new Page());
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
    fresh.page = std::make_unique<Page>();
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
    auto xy = coordinates(id, context);
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
std::vector<float> DualUpdate::coordinates(uint32_t id, Context &context) {
  require(id < allocated(), "coordinate ID outside allocated range");
  auto saved = context.vectors.find(id);
  if (saved != context.vectors.end())
    return saved->second;
  auto loc = coordinate_loc_[id];
  auto p = context.read(File::coordinates, loc / coord_per_page_) +
           loc % coord_per_page_ * (d1_ + d2_) * 4;
  std::vector<float> xy(d1_ + d2_);
  std::memcpy(xy.data(), p, xy.size() * 4);
  context.vectors.emplace(id, xy);
  return xy;
}
std::vector<float> DualUpdate::coordinates(uint32_t id, IOStats &stats) {
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
  auto x = coordinates(a, context), y = coordinates(b, context);
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
  std::vector<float> t1(chunks1_ * 256), t2(chunks2_ * 256);
  pq1_.populate_chunk_distances(x, t1.data());
  pq2_.populate_chunk_distances(y, t2.data());
  auto distance = [&](uint32_t id) {
    float a = 0, b = 0;
    for (unsigned c = 0; c < chunks1_; ++c)
      a += t1[c * 256 + codes1_[uint64_t(id) * chunks1_ + c]];
    for (unsigned c = 0; c < chunks2_; ++c)
      b += t2[c * 256 + codes2_[uint64_t(id) * chunks2_ + c]];
    return DistancePair(std::sqrt(std::max(0.f, a)),
                        std::sqrt(std::max(0.f, b)));
  };
  std::vector<pipeann::NeighborH> pool;
  std::unordered_set<uint32_t> seen;
  auto add = [&](uint32_t id) {
    if (id < allocated() && committed_.test(id) && seen.insert(id).second) {
      auto d = distance(id);
      pool.emplace_back(id, d.first, d.second, true, 0);
    }
  };
  for (auto id : entries())
    add(id);
  pool = pareto_order(std::move(pool), options_.L_ins);
  // Expand all still-active points in the best current Pareto layer, then
  // re-layer, as in the builder's iterate_to_fixed_point. The explicit cap
  // prevents a wide final layer from exceeding the documented L_ins budget.
  for (;;) {
    int layer = std::numeric_limits<int>::max();
    for (const auto &v : pool)
      if (v.flag)
        layer = std::min(layer, v.layer);
    if (layer == std::numeric_limits<int>::max())
      break;
    std::vector<uint32_t> frontier;
    for (auto &v : pool)
      if (v.flag && v.layer == layer) {
        v.flag = false;
        frontier.push_back(v.id);
      }
    std::vector<uint64_t> batch;
    for (auto id : frontier)
      batch.push_back(topology_loc_[id] / topo_per_page_);
    context.prefetch(File::topology, std::move(batch));
    for (auto id : frontier)
      for (auto v : neighbors(id, context))
        add(v);
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
  std::vector<float> t1(chunks1_ * 256), t2(chunks2_ * 256);
  pq1_.populate_chunk_distances(x, t1.data());
  pq2_.populate_chunk_distances(y, t2.data());
  struct Candidate {
    uint32_t id;
    float distance;
    bool active;
  };
  auto order = [](const Candidate &a, const Candidate &b) {
    return a.distance < b.distance || (a.distance == b.distance && a.id < b.id);
  };
  std::vector<Candidate> pool;
  std::unordered_set<uint32_t> seen;
  auto add = [&](uint32_t id) {
    if (id >= allocated() || !committed_.test(id) || !seen.insert(id).second)
      return;
    float a = 0, b = 0;
    for (unsigned c = 0; c < chunks1_; ++c)
      a += t1[c * 256 + codes1_[uint64_t(id) * chunks1_ + c]];
    for (unsigned c = 0; c < chunks2_; ++c)
      b += t2[c * 256 + codes2_[uint64_t(id) * chunks2_ + c]];
    pool.push_back({ id, alpha * std::sqrt(std::max(0.f, a)) +
                             (1 - alpha) * std::sqrt(std::max(0.f, b)),
                     true });
    ++work.pq_evaluations;
  };
  for (auto id : entries())
    add(id);
  std::sort(pool.begin(), pool.end(), order);
  if (pool.size() > L)
    pool.resize(L);
  for (;;) {
    auto next = std::find_if(pool.begin(), pool.end(),
                             [](const Candidate &v) { return v.active; });
    if (next == pool.end())
      break;
    auto id = next->id;
    next->active = false;
    ++work.expanded;
    for (auto v : neighbors(id, context))
      add(v);
    std::sort(pool.begin(), pool.end(), order);
    if (pool.size() > L)
      pool.resize(L);
  }
  std::vector<uint64_t> batch;
  for (const auto &v : pool)
    if (!deleted_.test(v.id) && batch.size() < rerank)
      batch.push_back(coordinate_loc_[v.id] / coord_per_page_);
  context.prefetch(File::coordinates, std::move(batch));
  std::vector<QueryResult> result;
  for (const auto &v : pool) {
    // Deleted nodes may still route traversal, but never consume exact rerank.
    if (deleted_.test(v.id))
      continue;
    if (result.size() == rerank)
      break;
    auto xy = coordinates(v.id, context);
    float d = alpha * l2(x, xy.data(), d1_) +
              (1 - alpha) * l2(y, xy.data() + d1_, d2_);
    result.push_back({ v.id, tags_[v.id], d });
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
  auto candidates = insert_candidates(x, y, context);
  std::vector<uint32_t> ids;
  for (const auto &v : candidates)
    if (!deleted_.test(v.id) && ids.size() < options_.C_ins)
      ids.push_back(v.id);
  auto selected = select(id, std::move(ids), true, context);
  modify_topology(topology_loc_[id] / topo_per_page_,
                  [&](unsigned char *page, DegreeChanges &undo) {
                    replace_neighbors(page, id, selected, undo);
                  },
                  stats);

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
              auto ranked = select(target, std::move(candidates),
                                   options_.exact_reprune, context);
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
  require(ledger_.count(id) > 0, "insert cannot commit: no incoming edge after "
                                 "protected reverse insertion");
  update_entries(id, x, y, true, &selected);
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
void DualUpdate::rebuild_entries(Context &context) {
  std::vector<Entry> replacement;
  for (uint32_t id = 0; id < allocated(); ++id) {
    if (!committed_.test(id) || deleted_.test(id))
      continue;
    auto xy = coordinates(id, context);
    Entry added{ id, l2(xy.data(), center1_.data(), d1_),
                 l2(xy.data() + d1_, center2_.data(), d2_) };
    replacement.push_back(added);
    replacement = builder_entry_frontier(std::move(replacement));
    // A scan must not turn coordinates or topology into a persistent RAM index.
    if (id % coord_per_page_ == coord_per_page_ - 1) {
      context.vectors.clear();
      context.cache[1].clear();
    }
  }
  std::unique_lock<std::shared_mutex> guard(entry_mutex_);
  entry_frontier_.swap(replacement);
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
  auto dead = [&](uint32_t id) {
    return bool((frozen[id / 64] >> (id % 64)) & 1);
  };
  uint32_t n = allocated();
  // Save only deleted-node neighborhoods needed for bypassing them. This is
  // bounded by the deletion set, not by the full live topology.
  std::unordered_map<uint32_t, std::vector<uint32_t> > deleted_neighbors;
  for (uint32_t id = 0; id < n; ++id) {
    if (!committed_.test(id) || !dead(id) || consolidated(id))
      continue;
    Context context(*store_, stats);
    auto row = neighbors(id, context);
    row.erase(std::remove_if(row.begin(), row.end(), [&](uint32_t v) {
                return !committed_.test(v) || dead(v);
              }),
              row.end());
    deleted_neighbors.emplace(id, std::move(row));
  }
  for (uint64_t page = 0; page < pages(n, topo_per_page_); ++page) {
    Context context(*store_, stats);
    bool affected = false;
    for (uint32_t id = page * topo_per_page_;
         id < std::min<uint64_t>(n, (page + 1) * topo_per_page_); ++id) {
      if (!committed_.test(id))
        continue;
      if (dead(id) && !consolidated(id)) {
        affected = true;
        break;
      }
      if (dead(id))
        continue;
      for (auto v : neighbors(id, context))
        if (dead(v)) {
          affected = true;
          break;
        }
      if (affected)
        break;
    }
    if (!affected)
      continue;
    modify_topology(
        page,
        [&](unsigned char *buffer, DegreeChanges &undo) {
          for (uint32_t id = page * topo_per_page_;
               id < std::min<uint64_t>(n, (page + 1) * topo_per_page_); ++id) {
            if (!committed_.test(id))
              continue;
            if (dead(id)) {
              replace_neighbors(buffer, id, {}, undo);
              continue;
            }
            auto p = buffer + id % topo_per_page_ * record_bytes_;
            uint32_t degree;
            std::memcpy(&degree, p, 4);
            require(degree <= options_.degree, "consolidation degree corrupt");
            std::vector<uint32_t> row(degree), candidates;
            std::memcpy(row.data(), p + 4, degree * 4);
            bool changed = false;
            for (auto v : row) {
              if (dead(v)) {
                changed = true;
                auto found = deleted_neighbors.find(v);
                if (found != deleted_neighbors.end())
                  candidates.insert(candidates.end(), found->second.begin(),
                                    found->second.end());
              } else if (committed_.test(v))
                candidates.push_back(v);
            }
            if (!changed)
              continue;
            auto replacement = select(id, std::move(candidates),
                                      options_.exact_reprune, context);
            replace_neighbors(buffer, id, replacement, undo);
          }
        },
        stats);
  }
  bool rebuild = false;
  for (auto id : entries())
    if (dead(id))
      rebuild = true;
  if (rebuild) {
    Context context(*store_, stats);
    rebuild_entries(context);
  }
  for (uint32_t id = 0; id < n; ++id)
    if (dead(id))
      consolidated_[id].store(true, std::memory_order_release);
  merged_count_.store(frozen_count, std::memory_order_release);
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
