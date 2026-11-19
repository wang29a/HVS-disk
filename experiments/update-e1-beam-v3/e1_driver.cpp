#include "dual_update.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <thread>
using namespace hvs_update;
using Clock = std::chrono::steady_clock;
static void check(bool b, const std::string &s) {
  if (!b)
    throw std::runtime_error(s);
}
template <class T> static void put(std::ostream &o, T v) {
  o.write(reinterpret_cast<const char *>(&v), sizeof(v));
  check(bool(o), "binary write");
}
static std::vector<float> matrix(const std::string &p, unsigned &n,
                                 unsigned &d) {
  std::ifstream f(p, std::ios::binary);
  f.read(reinterpret_cast<char *>(&n), 4);
  f.read(reinterpret_cast<char *>(&d), 4);
  check(bool(f) && n && d, "matrix header " + p);
  std::vector<float> v(uint64_t(n) * d);
  f.read(reinterpret_cast<char *>(v.data()), v.size() * 4);
  check(bool(f) && f.peek() == EOF, "matrix length " + p);
  return v;
}
static double percentile(std::vector<double> v, double p) {
  if (v.empty())
    return 0;
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, size_t(std::ceil(p * v.size()) - 1))];
}
static void protection_json(std::ostream &o, const DegreeCounters &before,
                            const DegreeCounters &after) {
  o << "{";
  bool first = true;
  auto field = [&](const char *name, uint64_t b, uint64_t a) {
    check(a >= b, "protection counter regressed");
    if (!first)
      o << ",";
    first = false;
    o << "\"" << name << "\":" << a - b;
  };
#define FIELD(name) field(#name, before.name, after.name)
  FIELD(reverse_reprunes);
  FIELD(full_reprunes);
  FIELD(protection_events);
  FIELD(protected_edges);
  FIELD(protected_old_edges);
  FIELD(protected_new_edges);
  FIELD(reverse_capacity_conflicts);
  FIELD(full_before_edges);
  FIELD(full_after_edges);
#undef FIELD
  for (bool raw : { false, true }) {
    o << ",\"" << (raw ? "full_raw_after_degree_histogram"
                       : "full_after_degree_histogram") << "\":[";
    const auto &b =
        raw ? before.full_raw_after_degree : before.full_after_degree;
    const auto &a = raw ? after.full_raw_after_degree : after.full_after_degree;
    for (unsigned d = 0; d <= 40; ++d) {
      if (d)
        o << ",";
      o << a[d] - b[d];
    }
    o << "]";
  }
  o << ",\"full_degree_before\":40,\"degree_delta_for_bin_d\":\"d-40\"}\n";
}
static uint64_t checkpoint(DualUpdate &index, const std::string &directory,
                           const std::vector<uint32_t> &tags,
                           std::vector<uint32_t> &U0) {
  IOStats work;
  auto graph = index.audit_topology(work);
  auto n = index.allocated();
  std::vector<unsigned char> seen(n, 0);
  std::vector<uint32_t> queue;
  auto entries = index.entries();
  for (auto id : entries) {
    check(id < n && index.committed(id), "entry bounds");
    if (!seen[id]) {
      seen[id] = 1;
      queue.push_back(id);
    }
  }
  uint64_t edges = 0, zero = 0, new_zero = 0;
  std::vector<uint32_t> counted(n, 0);
  std::array<uint64_t, 41> histogram{};
  auto ledger = index.indegree_snapshot();
  auto positive = index.positive_indegree_history();
  for (unsigned id = 0; id < n; ++id) {
    check(index.committed(id) && !index.deleted(id),
          "checkpoint has invisible/deleted ID");
    auto row = graph[id];
    check(row.size() <= 40, "degree exceeds40");
    std::sort(row.begin(), row.end());
    check(std::adjacent_find(row.begin(), row.end()) == row.end(),
          "duplicate edge");
    for (auto v : row) {
      check(v < n && v != id, "invalid/self edge");
      ++counted[v];
    }
    ++histogram[row.size()];
    edges += row.size();
  }
  check(counted == ledger,
        "indegree ledger differs from independent graph recount");
  for (unsigned id = 0; id < n; ++id)
    if (!counted[id]) {
      ++zero;
      if (positive[id] || id >= index.initial_points())
        ++new_zero;
    }
  for (size_t at = 0; at < queue.size(); ++at)
    for (auto v : graph[queue[at]])
      if (!seen[v]) {
        seen[v] = 1;
        queue.push_back(v);
      }
  std::vector<uint32_t> unreachable, newly, inserted;
  for (unsigned id = 0; id < n; ++id)
    if (!seen[id]) {
      unreachable.push_back(tags[id]);
      if (tags[id] >= index.initial_points())
        inserted.push_back(tags[id]);
    }
  std::sort(unreachable.begin(), unreachable.end());
  if (n == index.initial_points())
    U0 = unreachable;
  std::set_difference(unreachable.begin(), unreachable.end(), U0.begin(),
                      U0.end(), std::back_inserter(newly));
  std::ofstream u(directory + "/unreachable-tags.txt");
  for (auto id : unreachable)
    u << id << '\n';
  std::ofstream nu(directory + "/new-unreachable-tags.txt");
  for (auto id : newly)
    nu << id << '\n';
  std::ofstream iu(directory + "/inserted-unreachable-tags.txt");
  for (auto id : inserted)
    iu << id << '\n';
  std::ofstream um(directory + "/unreachable-internal.txt");
  for (unsigned id = 0; id < n; ++id)
    if (!seen[id])
      um << id << '\n';
  std::ofstream mapping(directory + "/id-to-tag.u32", std::ios::binary);
  mapping.write(reinterpret_cast<const char *>(tags.data()), uint64_t(n) * 4);
  std::vector<unsigned char> meta(8192, 0);
  auto assign = [&](unsigned off,
                    uint32_t v) { std::memcpy(meta.data() + off, &v, 4); };
  assign(0, n);
  assign(4, index.dim1());
  assign(8, index.dim2());
  assign(12, 40);
  assign(16, 0);
  uint64_t npp = 49;
  std::memcpy(meta.data() + 20, &npp, 8);
  assign(28, entries.size());
  check(32 + 4 * entries.size() <= 8192, "entry metadata overflow");
  for (unsigned j = 0; j < entries.size(); ++j)
    assign(32 + j * 4, entries[j]);
  std::ofstream mf(directory + "/index.meta", std::ios::binary);
  mf.write(reinterpret_cast<const char *>(meta.data()), meta.size());
  mf.close();
  std::ofstream summary(directory + "/structure.json");
  summary << std::setprecision(17) << "{\"points\":" << n
          << ",\"edges\":" << edges
          << ",\"average_degree\":" << double(edges) / n
          << ",\"unreachable_live\":" << unreachable.size()
          << ",\"initial_U0\":" << U0.size()
          << ",\"new_unreachable\":" << newly.size()
          << ",\"inserted_unreachable\":" << inserted.size()
          << ",\"entries\":" << entries.size()
          << ",\"zero_indegree_live\":" << zero
          << ",\"new_zero_indegree_live\":" << new_zero
          << ",\"full_degree_nodes\":" << histogram[40]
          << ",\"full_degree_fraction\":" << double(histogram[40]) / n
          << ",\"degree_histogram\":[";
  for (unsigned d = 0; d <= 40; ++d) {
    if (d)
      summary << ",";
    summary << histogram[d];
  }
  summary << "]}\n";
  summary.close();
  std::ofstream cumulative(directory + "/protection-cumulative.json");
  protection_json(cumulative, DegreeCounters{}, index.degree_counters());
  check(new_zero == 0, "update made a live node indegree zero");
  index.snapshot(directory + "/normal-exit.snapshot");
  std::cout << "CHECKPOINT_READY " << n << " " << directory << std::endl;
  return edges;
}
int main(int argc, char **argv) {
  try {
    check(
        argc == 9,
        "fixture pq|exact threads C_ins always_reprune initial batch batches");
    std::string root = argv[1];
    Options options;
    options.exact_reprune = std::string(argv[2]) == "exact";
    check(options.exact_reprune || std::string(argv[2]) == "pq",
          "reprune mode");
    unsigned threads = std::stoul(argv[3]), initial = std::stoul(argv[6]),
             batch = std::stoul(argv[7]), batches = std::stoul(argv[8]);
    options.update_threads = threads;
    options.C_ins = std::stoul(argv[4]);
    options.always_reprune = std::stoi(argv[5]);
    check(threads && threads <= 8 && batch && batches, "workload sizes");
    DualUpdate index(root, options);
    check(index.initial_points() == initial, "initial count");
    unsigned nx, dx, ny, dy, nq, dq, nq2, dq2;
    auto x = matrix(root + "/insert-x.bin", nx, dx),
         y = matrix(root + "/insert-y.bin", ny, dy);
    auto qx = matrix(root + "/query-x.bin", nq, dq),
         qy = matrix(root + "/query-y.bin", nq2, dq2);
    check(nx == ny && nx == batch * batches && dx == index.dim1() &&
              dy == index.dim2() && dq == dx && dq2 == dy && nq == nq2,
          "data dimensions");
    std::ifstream qids(root + "/report-ids.txt");
    std::vector<unsigned> query_ids;
    unsigned q;
    while (qids >> q) {
      check(q < nq, "query ID");
      query_ids.push_back(q);
    }
    check(query_ids.size() == 500, "report cohort count");
    std::vector<uint32_t> tags(index.capacity()), U0;
    std::iota(tags.begin(), tags.end(), 0);
    unsigned inserted = 0;
    uint64_t last_edges = 0;
    std::string command, directory;
    unsigned target = 0, L = 0, rerank = 0;
    while (std::cin >> command) {
      if (command == "checkpoint") {
        std::cin >> directory;
        last_edges = checkpoint(index, directory, tags, U0);
      } else if (command == "insert") {
        std::cin >> target >> L >> rerank >> directory;
        check(target > inserted && target <= nx && L >= 10 && rerank >= 10 &&
                  rerank <= L,
              "insert command");
        std::atomic<unsigned> next{ inserted }, done{ 0 }, query_sequence{ 0 };
        std::atomic<bool> stop{ false }, failed{ false };
        std::exception_ptr error;
        std::mutex mu;
        struct Sample {
          unsigned ordinal, id;
          double us;
          IOStats io;
        };
        std::vector<std::vector<Sample> > samples(threads);
        struct Query {
          unsigned sequence, qi;
          float alpha;
          double us;
          uint64_t wait;
        };
        std::vector<std::vector<Query> > query_samples(4);
        auto degree_before = index.degree_counters();
        auto start = Clock::now();
        auto fail = [&] {
          failed = true;
          std::lock_guard<std::mutex> g(mu);
          if (!error)
            error = std::current_exception();
        };
        std::vector<std::thread> readers, writers;
        std::array<float, 5> weights{ { .1f, .3f, .5f, .7f, .9f } };
        for (unsigned t = 0; t < 4; ++t)
          readers.emplace_back([&, t] {
            try {
              while (!stop && !failed) {
                auto seq = query_sequence.fetch_add(1);
                auto qi = query_ids[(seq / 5) % query_ids.size()];
                float a = weights[seq % 5];
                QueryWork work;
                auto tick = Clock::now();
                auto result = index.search(qx.data() + uint64_t(qi) * dx,
                                           qy.data() + uint64_t(qi) * dy, a, 10,
                                           L, rerank, work);
                for (auto r : result)
                  check(r.tag < initial + target, "query returned invalid tag");
                query_samples[t].push_back(
                    { seq, qi, a, std::chrono::duration<double, std::micro>(
                                      Clock::now() - tick).count(),
                      work.io.page_lock_wait_ns });
              }
            }
            catch (...) {
              fail();
            }
          });
        for (unsigned t = 0; t < threads; ++t)
          writers.emplace_back([&, t] {
            try {
              while (!failed) {
                auto ordinal = next.fetch_add(1);
                if (ordinal >= target)
                  break;
                IOStats stats;
                auto tick = Clock::now();
                auto id = index.insert(x.data() + uint64_t(ordinal) * dx,
                                       y.data() + uint64_t(ordinal) * dy,
                                       initial + ordinal, stats);
                tags[id] = initial + ordinal;
                samples[t].push_back(
                    { ordinal, id, std::chrono::duration<double, std::micro>(
                                       Clock::now() - tick).count(),
                      stats });
                auto completed = done.fetch_add(1) + 1;
                if (completed % 1000 == 0) {
                  std::lock_guard<std::mutex> g(mu);
                  std::cout << "INSERT_PROGRESS " << inserted + completed << "/"
                            << target << std::endl;
                }
              }
            }
            catch (...) {
              fail();
            }
          });
        for (auto &t : writers)
          t.join();
        auto update_end = Clock::now();
        stop = true;
        for (auto &t : readers)
          t.join();
        auto query_end = Clock::now();
        if (error)
          std::rethrow_exception(error);
        check(done == target - inserted, "incomplete batch");
        auto degree_after = index.degree_counters();
        auto counts = index.indegree_snapshot();
        uint64_t edges_after =
            std::accumulate(counts.begin(), counts.end(), uint64_t(0));
        std::ofstream protection(directory + "/protection-batch.json");
        protection_json(protection, degree_before, degree_after);
        double seconds =
            std::chrono::duration<double>(update_end - start).count();
        std::ofstream raw(directory + "/insert-samples.jsonl");
        std::vector<double> latency;
        IOStats total;
        for (auto &v : samples)
          for (auto s : v) {
            latency.push_back(s.us);
            for (unsigned f = 0; f < 2; ++f) {
              total.read_pages[f] += s.io.read_pages[f];
              total.write_pages[f] += s.io.write_pages[f];
            }
            total.page_lock_wait_ns += s.io.page_lock_wait_ns;
            raw << std::setprecision(17) << "{\"ordinal\":" << s.ordinal
                << ",\"id\":" << s.id << ",\"latency_us\":" << s.us
                << ",\"topology_read_pages\":" << s.io.read_pages[0]
                << ",\"topology_write_pages\":" << s.io.write_pages[0]
                << ",\"coordinate_read_pages\":" << s.io.read_pages[1]
                << ",\"coordinate_write_pages\":" << s.io.write_pages[1]
                << ",\"page_lock_wait_ns\":" << s.io.page_lock_wait_ns << "}\n";
          }
        std::ofstream queries(directory + "/concurrent-query-samples.jsonl");
        std::vector<double> qlat;
        for (auto &v : query_samples)
          for (auto s : v) {
            qlat.push_back(s.us);
            queries << std::setprecision(17) << "{\"sequence\":" << s.sequence
                    << ",\"query_id\":" << s.qi << ",\"alpha\":" << s.alpha
                    << ",\"latency_us\":" << s.us
                    << ",\"page_lock_wait_ns\":" << s.wait << "}\n";
          }
        std::ofstream summary(directory + "/update-summary.json");
        summary << std::setprecision(17) << "{\"insertions\":" << done
                << ",\"total_edges_before\":" << last_edges
                << ",\"total_edges_after\":" << edges_after
                << ",\"update_threads\":" << threads
                << ",\"query_threads\":4,\"wall_seconds\":" << seconds
                << ",\"insert_qps\":" << done / seconds
                << ",\"mean_insert_us\":"
                << std::accumulate(latency.begin(), latency.end(), 0.) /
                       latency.size()
                << ",\"p95_insert_us\":" << percentile(latency, .95)
                << ",\"topology_read_pages\":" << total.read_pages[0]
                << ",\"topology_write_pages\":" << total.write_pages[0]
                << ",\"coordinate_read_pages\":" << total.read_pages[1]
                << ",\"coordinate_write_pages\":" << total.write_pages[1]
                << ",\"concurrent_queries\":" << qlat.size()
                << ",\"query_seconds_including_drain\":"
                << std::chrono::duration<double>(query_end - start).count()
                << ",\"concurrent_query_qps_including_drain\":"
                << qlat.size() /
                       std::chrono::duration<double>(query_end - start).count()
                << ",\"concurrent_query_mean_us\":"
                << (qlat.empty() ? 0 : std::accumulate(qlat.begin(), qlat.end(),
                                                       0.) /
                                           qlat.size())
                << ",\"concurrent_query_p95_us\":" << percentile(qlat, .95)
                << ",\"query_L\":" << L << ",\"query_rerank\":" << rerank
                << "}\n";
        summary.close();
        protection.close();
        raw.close();
        queries.close();
        check(bool(summary) && bool(protection) && bool(raw) && bool(queries),
              "failed to save update measurement");
        inserted = target;
        std::cout << "INSERT_BATCH_COMPLETE " << target << std::endl;
      } else if (command == "quit") {
        std::cout << "E1_DRIVER_COMPLETE" << std::endl;
        return 0;
      } else
        throw std::runtime_error("unknown command");
    }
    throw std::runtime_error("unexpected controller EOF");
  }
  catch (const std::exception &e) {
    std::cerr << "E1_DRIVER_FAIL " << e.what() << std::endl;
    return 1;
  }
}
