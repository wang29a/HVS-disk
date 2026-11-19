#include "dual_update.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

using namespace hvs_update;
using Clock = std::chrono::steady_clock;

static void check(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}

static double seconds(Clock::time_point at) {
  return std::chrono::duration<double>(Clock::now() - at).count();
}

template <class T>
static std::vector<T> matrix(const std::string &path, unsigned &count,
                             unsigned &dim) {
  std::ifstream f(path, std::ios::binary);
  f.read(reinterpret_cast<char *>(&count), 4);
  f.read(reinterpret_cast<char *>(&dim), 4);
  check(bool(f) && count && dim, "invalid input header");
  std::vector<T> out(uint64_t(count) * dim);
  f.read(reinterpret_cast<char *>(out.data()), out.size() * sizeof(T));
  check(bool(f) && f.peek() == EOF, "invalid input length");
  return out;
}

struct Load {
  std::vector<float> x, y;
  unsigned dx = 0, dy = 0;
  std::vector<unsigned> query_ids;

  explicit Load(const std::string &root) {
    unsigned nx, ny;
    x = matrix<float>(root + "/query-x.bin", nx, dx);
    y = matrix<float>(root + "/query-y.bin", ny, dy);
    check(nx == ny, "query row count");
    std::ifstream cohort(root + "/report-ids.txt");
    unsigned id;
    while (cohort >> id) {
      check(id < nx, "query ID");
      query_ids.push_back(id);
    }
    check(query_ids.size() == 500, "query cohort");
  }

  template <class Work>
  void measure(DualUpdate &index, const std::string &path, unsigned n,
               bool check_deleted, Work work) {
    std::atomic<unsigned> next{ 0 };
    std::atomic<bool> stop{ false };
    std::array<std::vector<double>, 4> timings;
    std::mutex errors;
    std::exception_ptr error;
    auto fail = [&] {
      stop = true;
      std::lock_guard<std::mutex> lock(errors);
      if (!error)
        error = std::current_exception();
    };
    std::vector<std::thread> readers;
    auto tick = Clock::now();
    for (unsigned t = 0; t < timings.size(); ++t)
      readers.emplace_back([&, t] {
        try {
          constexpr float weights[] = { .1f, .3f, .5f, .7f, .9f };
          while (!stop) {
            unsigned k = next++;
            unsigned q = query_ids[(k / 5) % query_ids.size()];
            QueryWork info;
            auto start = Clock::now();
            auto answer = index.search(x.data() + uint64_t(q) * dx,
                                       y.data() + uint64_t(q) * dy,
                                       weights[k % 5], 10, 96, 96, info);
            timings[t].push_back(seconds(start) * 1e6);
            if (check_deleted)
              for (const auto &r : answer)
                check(!index.deleted(r.id), "query returned a deleted node");
          }
        }
        catch (...) {
          fail();
        }
      });
    IOStats io;
    try {
      work(io);
    }
    catch (...) {
      fail();
    }
    const double wall = seconds(tick);
    stop = true;
    for (auto &r : readers)
      r.join();
    const double drain = seconds(tick);
    std::vector<double> latency;
    for (auto &v : timings)
      latency.insert(latency.end(), v.begin(), v.end());
    std::sort(latency.begin(), latency.end());
    const double mean =
        latency.empty() ? 0.
                        : std::accumulate(latency.begin(), latency.end(), 0.) /
                              latency.size();
    const double p95 =
        latency.empty() ? 0. : latency[(95 * latency.size() + 99) / 100 - 1];
    std::ofstream file(path);
    file.precision(17);
    file << "{\"completed\":" << (error ? "false" : "true")
         << ",\"operations\":" << n << ",\"wall_seconds\":" << wall
         << ",\"operations_per_second\":" << (error ? 0. : n / wall)
         << ",\"queries\":" << latency.size() << ",\"query_seconds\":" << drain
         << ",\"query_qps\":" << latency.size() / drain
         << ",\"mean_query_us\":" << mean << ",\"p95_query_us\":" << p95
         << ",\"topology_read_pages\":" << io.read_pages[0]
         << ",\"coordinate_read_pages\":" << io.read_pages[1]
         << ",\"topology_write_pages\":" << io.write_pages[0]
         << ",\"coordinate_write_pages\":" << io.write_pages[1]
         << ",\"page_lock_wait_ns\":" << io.page_lock_wait_ns
         << ",\"optimistic_retries\":" << io.optimistic_retries
         << ",\"degree_reservation_retries\":" << io.degree_reservation_retries
         << ",\"merge_transferred_edges\":" << io.merge_transferred_edges
         << ",\"merge_search_edges\":" << io.merge_search_edges
         << ",\"merge_live_nodes\":" << io.merge_live_nodes
         << ",\"merge_live_edges\":" << io.merge_live_edges
         << ",\"merge_average_degree\":"
         << (io.merge_live_nodes
                 ? double(io.merge_live_edges) / io.merge_live_nodes
                 : 0.)
         << ",\"merge_unreachable_before\":" << io.merge_unreachable_before
         << ",\"merge_unreachable_after\":" << io.merge_unreachable_after
         << "}\n";
    file.close();
    check(bool(file), "cannot save operation measurement");
    if (error)
      std::rethrow_exception(error);
  }
};

static void audit(DualUpdate &index, const std::string &out, unsigned dead,
                  bool merged, bool gate) {
  IOStats stats;
  auto graph = index.audit_topology(stats);
  auto n = index.allocated();
  auto entries = index.entries();
  auto ledger = index.indegree_snapshot();
  auto history = index.positive_indegree_history();
  std::vector<unsigned> recounted(n, 0), frontier;
  std::vector<uint8_t> reached(n);
  std::array<uint64_t, 41> histogram{};
  uint64_t living_edges = 0, physical_edges = 0, dead_refs = 0, dead_rows = 0;
  unsigned actual_dead = 0;
  for (unsigned u = 0; u < n; ++u) {
    check(index.committed(u), "initial node is uncommitted");
    auto row = graph[u];
    check(row.size() <= 40, "degree exceeds R");
    std::sort(row.begin(), row.end());
    check(std::adjacent_find(row.begin(), row.end()) == row.end(),
          "duplicate edge");
    const bool u_dead = index.deleted(u);
    actual_dead += u_dead;
    dead_rows += u_dead && !row.empty();
    if (!u_dead) {
      ++histogram[row.size()];
      living_edges += row.size();
    }
    physical_edges += row.size();
    for (auto v : row) {
      check(v < n && u != v, "invalid edge");
      ++recounted[v];
      dead_refs += index.deleted(v);
    }
  }
  check(actual_dead == dead && recounted == ledger,
        "deletion or indegree ledger mismatch");
  check(!merged || (!dead_refs && !dead_rows),
        "merged topology retains deleted IDs");
  for (auto v : entries) {
    check(v < n && (!merged || !index.deleted(v)), "invalid entry");
    if (!reached[v]) {
      reached[v] = true;
      frontier.push_back(v);
    }
  }
  for (size_t i = 0; i < frontier.size(); ++i)
    for (auto v : graph[frontier[i]])
      if (!reached[v]) {
        reached[v] = true;
        frontier.push_back(v);
      }

  unsigned unreachable = 0, new_zero = 0, zero = 0;
  std::ofstream unseen(out + "/unreachable-tags.txt"),
      rejects(out + "/new-zero-tags.txt");
  for (unsigned id = 0; id < n; ++id) {
    if (index.deleted(id))
      continue;
    if (!reached[id]) {
      ++unreachable;
      unseen << id << '\n';
    }
    if (!recounted[id]) {
      ++zero;
      if (history[id]) {
        ++new_zero;
        rejects << id << '\n';
      }
    }
  }
  unseen.close();
  rejects.close();
  std::vector<unsigned char> meta(8192);
  const auto assign = [&](unsigned offset, uint32_t value) {
    std::memcpy(meta.data() + offset, &value, 4);
  };
  assign(0, n);
  assign(4, index.dim1());
  assign(8, index.dim2());
  assign(12, 40);
  const uint64_t npp = 49;
  std::memcpy(meta.data() + 20, &npp, 8);
  check(32 + 4 * entries.size() <= meta.size(), "too many entries");
  assign(28, entries.size());
  for (unsigned j = 0; j < entries.size(); ++j)
    assign(32 + 4 * j, entries[j]);
  std::ofstream m(out + "/index.meta", std::ios::binary);
  m.write(reinterpret_cast<const char *>(meta.data()), meta.size());
  m.close();
  std::ofstream report(out + "/structure.json");
  report.precision(17);
  report << "{\"initial\":" << n << ",\"deleted\":" << dead
         << ",\"live\":" << n - dead << ",\"unreachable_live\":" << unreachable
         << ",\"zero_indegree_live\":" << zero
         << ",\"new_zero_indegree_live\":" << new_zero
         << ",\"live_source_edges\":" << living_edges
         << ",\"physical_edges\":" << physical_edges
         << ",\"edges_to_deleted\":" << dead_refs
         << ",\"nonempty_deleted_rows\":" << dead_rows
         << ",\"mean_live_source_degree\":" << double(living_edges) / (n - dead)
         << ",\"full_degree_fraction\":" << double(histogram[40]) / (n - dead)
         << ",\"degree_histogram\":[";
  for (unsigned d = 0; d < histogram.size(); ++d)
    report << (d ? "," : "") << histogram[d];
  report << "]}\n";
  report.close();
  check(bool(m) && bool(report) && bool(unseen) && bool(rejects),
        "checkpoint output failed");
  index.snapshot(out + "/state.snapshot");
  check(!gate || new_zero == 0, "new zero indegree among surviving points");
}

int main(int argc, char **argv) {
  try {
    if (argc == 4 && std::string(argv[1]) == "audit") {
      DualUpdate index(argv[2], Options{});
      audit(index, argv[3], 0, false, true);
      return 0;
    }
    check(argc == 9,
          "fixture pq|exact threads C always initial delete-count 1");
    const std::string root = argv[1], mode = argv[2];
    check(mode == "exact" || mode == "pq", "mode");
    Options options;
    options.exact_reprune = mode == "exact";
    const unsigned workers = std::stoul(argv[3]),
                   expected = std::stoul(argv[7]);
    check(workers == 4 && expected && std::string(argv[8]) == "1",
          "operation budget");
    options.update_threads = workers;
    DualUpdate index(root, options);
    check(index.initial_points() == std::stoul(argv[6]), "initial graph count");
    unsigned n, dim;
    auto ids = matrix<unsigned>(root + "/delete-ids.bin", n, dim);
    check(n == expected && dim == 1, "deletion set shape");
    auto sorted = ids;
    std::sort(sorted.begin(), sorted.end());
    check(sorted.back() < index.initial_points() &&
              std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(),
          "invalid deletion IDs");
    Load load(root);
    std::cout << "E2_READY" << std::endl;
    std::string cmd, out;
    check(bool(std::cin >> cmd >> out) && cmd == "delete", "delete command");
    load.measure(index, out + "/delete-summary.json", n, false, [&](IOStats &) {
      std::atomic<unsigned> next{ 0 }, success{ 0 };
      std::vector<std::thread> writers;
      for (unsigned t = 0; t < workers; ++t)
        writers.emplace_back([&] {
          for (auto i = next++; i < n; i = next++)
            if (index.erase(ids[i]))
              ++success;
        });
      for (auto &w : writers)
        w.join();
      check(success == n, "incomplete marking");
    });
    audit(index, out, n, false, true);
    std::cout << "DELETION_READY" << std::endl;
    check(bool(std::cin >> cmd >> out) && cmd == "merge", "merge command");
    try {
      load.measure(index, out + "/merge-summary.json", n, true,
                   [&](IOStats &io) { index.consolidate(io); });
    }
    catch (...) {
      audit(index, out, n, false, false);
      throw;
    }
    audit(index, out, n, true, true);
    std::cout << "MERGE_READY" << std::endl;
    check(bool(std::cin >> cmd) && cmd == "quit", "quit command");
    std::cout << "E1_DRIVER_COMPLETE" << std::endl;
    return 0;
  }
  catch (const std::exception &ex) {
    std::cerr << "E2_FAILED " << ex.what() << std::endl;
    return 1;
  }
}
