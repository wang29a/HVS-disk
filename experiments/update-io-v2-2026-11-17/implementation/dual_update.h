#pragma once
#include "deg_prune.h"
#include "pq_table.h"
#include "update_store.h"
#include "in_degree_guard.h"
#include "in_degree_ledger.h"
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

namespace hvs_update {
struct Options {
  uint32_t max_points = 0;
  unsigned L_ins = 200, C_ins = 100, degree = 40, update_threads = 4;
  bool exact_reprune = false, always_reprune = false;
  double consolidate_ratio = .05;
  std::string base_graph = "B";
};
struct QueryResult {
  uint32_t id;
  uint64_t tag;
  float distance;
};
struct QueryWork {
  IOStats io;
  uint64_t expanded = 0, pq_evaluations = 0, exact_evaluations = 0;
};

// Only compact 8192-byte pages in original ID order are accepted. Topology
// lives on SSD; only PQ, fixed-capacity state and the small entry set are RAM.
class DualUpdate {
public:
  DualUpdate(const std::string &directory, Options options);
  ~DualUpdate();
  uint32_t insert(const float *x, const float *y, uint64_t tag, IOStats &stats);
  bool erase(uint64_t tag);
  void consolidate(IOStats &stats);
  bool consolidation_due() const;
  std::vector<QueryResult> search(const float *x, const float *y, float alpha,
                                  unsigned k, unsigned L, unsigned rerank,
                                  QueryWork &stats);
  std::vector<uint32_t> neighbors(uint32_t id, IOStats &stats);
  // Offline gate inspection only; normal search/update never retains this
  // complete adjacency. Read each physical page once into a private buffer.
  std::vector<std::vector<uint32_t> > audit_topology(IOStats &stats);
  std::vector<float> coordinates(uint32_t id, IOStats &stats);
  std::vector<uint32_t> entries() const;
  std::vector<uint32_t> indegree_snapshot() const;
  std::vector<uint8_t> positive_indegree_history() const;
  DegreeCounters degree_counters() const;
  uint32_t allocated() const {
    return next_id_.load(std::memory_order_acquire);
  }
  uint32_t initial_points() const { return initial_; }
  uint32_t capacity() const { return options_.max_points; }
  uint32_t dim1() const { return d1_; }
  uint32_t dim2() const { return d2_; }
  bool committed(uint32_t id) const { return committed_.test(id); }
  bool deleted(uint32_t id) const { return deleted_.test(id); }
  bool consolidated(uint32_t id) const;
  void snapshot(const std::string &path);

private:
  struct Context;
  struct Entry {
    uint32_t id;
    float x, y;
  };
  Options options_;
  std::string directory_;
  uint32_t initial_, d1_, d2_, record_bytes_, topo_per_page_, coord_per_page_;
  std::unique_ptr<PageStore> store_;
  AtomicBitmap committed_, deleted_, accounted_;
  std::unique_ptr<std::atomic<bool> []> consolidated_;
  std::unique_ptr<uint32_t[]> topology_loc_, coordinate_loc_;
  std::unique_ptr<uint64_t[]> tags_;
  std::unique_ptr<uint8_t[]> codes1_, codes2_;
  pipeann::FixedChunkPQTable<float> pq1_, pq2_;
  unsigned chunks1_ = 0, chunks2_ = 0;
  std::vector<float> center1_, center2_;
  std::vector<Entry> entry_frontier_;
  std::atomic<uint32_t> next_id_, marked_count_{ 0 }, merged_count_{ 0 };
  mutable std::shared_mutex entry_mutex_;
  std::shared_mutex merge_mutex_;
  mutable std::mutex tag_mutex_, deletion_mutex_;
  std::unordered_map<uint64_t, uint32_t> tag_to_id_;
  // Counts physical edges from committed source records. Soft deletion leaves
  // routing edges in place; consolidation accounts for actual row changes.
  AtomicDegreeLedger ledger_;
  // Telemetry only: never held across pruning, page I/O, or count reservation.
  mutable std::mutex degree_stats_mutex_;
  struct Pending;
  static thread_local Pending *pending_;
  uint32_t degree_count(uint32_t id);
  DegreeCounters &transaction_counters();
  DegreeCounters degree_stats_;
  struct DegreeUndo {
    uint32_t id, count;
    uint8_t positive;
  };
  using DegreeChanges = std::vector<DegreeUndo>;

  static Options resolve(const std::string &directory, Options options);
  std::vector<uint32_t> neighbors(uint32_t id, Context &context);
  std::vector<float> coordinates(uint32_t id, Context &context);
  std::vector<pipeann::NeighborH>
  insert_candidates(const float *x, const float *y, Context &context);
  DistancePair pq_pair(uint32_t a, uint32_t b);
  DistancePair exact_pair(uint32_t a, uint32_t b, Context &context);
  std::vector<uint32_t> select(uint32_t source,
                               std::vector<uint32_t> candidates, bool exact,
                               Context &context);
  void update_entries(uint32_t id, const float *x, const float *y, bool publish,
                      const std::vector<uint32_t> *outgoing = nullptr);
  void rebuild_entries(Context &context);
  void put_neighbors(unsigned char *page, uint32_t id,
                     const std::vector<uint32_t> &neighbors);
  void modify_topology(
      uint64_t page,
      const std::function<void(unsigned char *, DegreeChanges &)> &change,
      IOStats &stats);
  void degree_delta(uint32_t id, int delta, DegreeChanges &undo);
  void replace_neighbors(unsigned char *page, uint32_t id,
                         const std::vector<uint32_t> &neighbors,
                         DegreeChanges &undo);
};
} // namespace hvs_update
