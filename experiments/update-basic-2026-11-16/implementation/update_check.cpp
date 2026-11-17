#include "dual_update.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

using namespace hvs_update;
static void check(bool ok, const std::string &reason) {
  if (!ok)
    throw std::runtime_error(reason);
}
static std::vector<float> matrix(const std::string &path, uint32_t &n,
                                 uint32_t &d) {
  std::ifstream in(path, std::ios::binary);
  in.read(reinterpret_cast<char *>(&n), 4);
  in.read(reinterpret_cast<char *>(&d), 4);
  check(bool(in) && n > 0 && d > 0, "matrix header");
  std::vector<float> data(uint64_t(n) * d);
  in.read(reinterpret_cast<char *>(data.data()), data.size() * 4);
  check(bool(in) && in.peek() == EOF, "matrix length");
  return data;
}
static double validate(DualUpdate &index, bool merged,
                       const std::string &prefix,
                       std::unordered_set<uint32_t> &baseline_unreachable,
                       bool initialize, bool full, std::ofstream &audit,
                       unsigned operation) {
  uint32_t n = index.allocated();
  IOStats stats;
  auto graph = index.audit_topology(stats);
  uint64_t edges = 0;
  std::vector<uint32_t> indegree(n, 0);
  for (uint32_t id = 0; id < n; ++id) {
    if (!index.committed(id))
      continue;
    check(graph[id].size() <= 40, "degree over R at " + std::to_string(id));
    std::unordered_set<uint32_t> seen;
    for (auto v : graph[id]) {
      check(v < n && index.committed(v),
            "edge to uncommitted/unallocated ID at " + std::to_string(id));
      check(v != id && seen.insert(v).second,
            "self/duplicate edge at " + std::to_string(id));
      check(!index.consolidated(v),
            "edge to consolidated ID at " + std::to_string(id));
      if (merged)
        check(!index.deleted(v),
              "post-merge edge to deleted ID at " + std::to_string(id));
      ++indegree[v];
    }
    if (!index.deleted(id))
      edges += graph[id].size();
  }
  auto tracked = index.indegree_snapshot();
  auto positive = index.positive_indegree_history();
  uint32_t zero_live = 0, new_zero_live = 0;
  for (uint32_t id = 0; id < n; ++id) {
    check(tracked[id] == indegree[id],
          "indegree ledger mismatch at " + std::to_string(id));
    if (index.committed(id) && !index.deleted(id) && indegree[id] == 0) {
      ++zero_live;
      if (positive[id] || id >= index.initial_points())
        ++new_zero_live;
    }
  }
  auto frontier = index.entries();
  std::vector<bool> reached(n, false);
  std::vector<uint32_t> queue;
  for (auto id : frontier) {
    check(id < n && index.committed(id), "invalid entry");
    if (!reached[id]) {
      reached[id] = true;
      queue.push_back(id);
    }
  }
  for (uint64_t p = 0; p < queue.size(); ++p)
    for (auto id : graph[queue[p]])
      if (!reached[id]) {
        reached[id] = true;
        queue.push_back(id);
      }
  std::vector<uint32_t> unreachable;
  uint32_t living = 0;
  for (uint32_t id = 0; id < n; ++id)
    if (index.committed(id) && !index.deleted(id)) {
      ++living;
      if (!reached[id])
        unreachable.push_back(id);
    }
  if (initialize) {
    baseline_unreachable.clear();
    baseline_unreachable.insert(unreachable.begin(), unreachable.end());
    std::ofstream out(prefix + "-U0.txt");
    for (auto id : unreachable)
      out << id << '\n';
  }
  std::vector<uint32_t> newly_unreachable, inserted_unreachable, recovered;
  for (auto id : unreachable) {
    if (id >= index.initial_points())
      inserted_unreachable.push_back(id);
    if (!baseline_unreachable.count(id))
      newly_unreachable.push_back(id);
  }
  for (auto id : baseline_unreachable)
    if (!index.deleted(id) && reached[id])
      recovered.push_back(id);
  audit << "{\"op\":" << operation
        << ",\"merged\":" << (merged ? "true" : "false")
        << ",\"allocated\":" << n << ",\"living\":" << living
        << ",\"baseline_U0_size\":" << baseline_unreachable.size()
        << ",\"unreachable_live\":" << unreachable.size()
        << ",\"new_unreachable\":" << newly_unreachable.size()
        << ",\"inserted_unreachable\":" << inserted_unreachable.size()
        << ",\"zero_indegree_live\":" << zero_live
        << ",\"new_zero_indegree_live\":" << new_zero_live
        << ",\"recovered_U0\":[";
  for (size_t j = 0; j < recovered.size(); ++j)
    audit << (j ? "," : "") << recovered[j];
  audit << "]}\n";
  audit.flush();
  if (full || !newly_unreachable.empty() || !inserted_unreachable.empty()) {
    std::ofstream out(prefix + "-unreachable.txt");
    for (auto id : unreachable)
      out << id << '\n';
  }
  check(new_zero_live == 0,
        "live node lost its last incoming edge after operation " +
            std::to_string(operation));
  for (unsigned j = 0; full && j < 5 && living; ++j) {
    uint32_t id = (j * 997) % n;
    auto xy = index.coordinates(id, stats);
    QueryWork work;
    auto results = index.search(xy.data(), xy.data() + index.dim1(),
                                float(j) / 4, 10, 200, 100, work);
    std::unordered_set<uint32_t> returned;
    for (const auto &r : results) {
      check(index.committed(r.id) && !index.deleted(r.id),
            "invalid query result");
      check(returned.insert(r.id).second, "duplicate query result");
      check(std::isfinite(r.distance), "non-finite returned distance");
    }
  }
  double mean = living ? double(edges) / living : 0;
  if (full)
    std::cout << "CHECKPOINT allocated=" << n << " living=" << living
              << " mean_degree=" << mean
              << " U0_size=" << baseline_unreachable.size()
              << " unreachable_live=" << unreachable.size()
              << " new_unreachable=" << newly_unreachable.size()
              << " new_zero_indegree_live=" << new_zero_live
              << " merged=" << merged << std::endl;
  return mean;
}
int main(int argc, char **argv) {
  std::string directory;
  try {
    check(argc == 4, "update_check directory pq|exact operations");
    directory = argv[1];
    Options options;
    check(std::string(argv[2]) == "pq" || std::string(argv[2]) == "exact",
          "reprune-dist");
    options.exact_reprune = std::string(argv[2]) == "exact";
    DualUpdate index(directory, options);
    uint32_t nx, dx, ny, dy;
    auto x = matrix(directory + "/insert-x.bin", nx, dx),
         y = matrix(directory + "/insert-y.bin", ny, dy);
    check(nx == ny && dx == index.dim1() && dy == index.dim2(),
          "insert matrix shape");
    std::ofstream trace(directory + "/operations.jsonl");
    std::ofstream audit(directory + "/structural-audit.jsonl");
    std::unordered_set<uint32_t> baseline_unreachable;
    validate(index, false, directory + "/checkpoint-0", baseline_unreachable,
             true, true, audit, 0);
    std::mt19937 rng(20261116);
    std::vector<uint64_t> living(index.initial_points());
    std::iota(living.begin(), living.end(), 0);
    unsigned inserted = 0, operations = std::stoul(argv[3]);
    bool merged = false;
    IOStats total;
    for (unsigned op = 1; op <= operations; ++op) {
      unsigned choice = rng() % 10;
      IOStats io;
      if (choice < 6) {
        check(inserted < nx, "insufficient fresh Laion points");
        uint64_t tag = uint64_t(index.initial_points()) + inserted;
        auto id = index.insert(x.data() + uint64_t(inserted) * dx,
                               y.data() + uint64_t(inserted) * dy, tag, io);
        living.push_back(tag);
        ++inserted;
        merged = false;
        trace << "{\"op\":" << op << ",\"action\":\"insert\",\"tag\":" << tag
              << ",\"id\":" << id << "}\n";
      } else if (choice < 9 && !living.empty()) {
        size_t at = rng() % living.size();
        auto tag = living[at];
        check(index.erase(tag), "delete live tag failed");
        living[at] = living.back();
        living.pop_back();
        merged = false;
        trace << "{\"op\":" << op << ",\"action\":\"delete\",\"tag\":" << tag
              << "}\n";
      } else {
        index.consolidate(io);
        merged = true;
        trace << "{\"op\":" << op << ",\"action\":\"consolidate\"}\n";
      }
      for (unsigned f = 0; f < 2; ++f) {
        total.read_pages[f] += io.read_pages[f];
        total.write_pages[f] += io.write_pages[f];
      }
      total.page_lock_wait_ns += io.page_lock_wait_ns;
      trace.flush();
      // Revision 3: verify every completed operation, retaining the original
      // every-100 full query checks. U0 exempts initial nodes only.
      validate(index, merged, directory + "/checkpoint-" + std::to_string(op),
               baseline_unreachable, false, op % 100 == 0, audit, op);
      if (index.consolidation_due()) {
        index.consolidate(total);
        merged = true;
        validate(index, true, directory + "/auto-merge-" + std::to_string(op),
                 baseline_unreachable, false, false, audit, op);
      }
    }
    index.consolidate(total);
    validate(index, true, directory + "/checkpoint-final", baseline_unreachable,
             false, true, audit, operations + 1);
    index.snapshot(directory + "/normal-exit.snapshot");
    std::cout << "G1_PASS operations=" << operations << " inserted=" << inserted
              << " reprune=" << argv[2]
              << " topology_read_pages=" << total.read_pages[0]
              << " topology_write_pages=" << total.write_pages[0]
              << " coordinate_read_pages=" << total.read_pages[1]
              << " coordinate_write_pages=" << total.write_pages[1]
              << " page_lock_wait_ns=" << total.page_lock_wait_ns << '\n';
    return 0;
  }
  catch (const std::exception &e) {
    std::cerr << "G1_STOP " << e.what() << '\n';
    if (!directory.empty()) {
      std::ofstream out(directory + "/failure.txt");
      out << e.what() << '\n';
    }
    return 1;
  }
}
