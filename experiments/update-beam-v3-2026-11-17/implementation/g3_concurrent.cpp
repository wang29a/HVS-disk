#include "dual_update.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <deque>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <thread>

using namespace hvs_update;
using Clock = std::chrono::steady_clock;
static void check(bool b, const std::string &s) {
  if (!b)
    throw std::runtime_error(s);
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
struct Event {
  char kind;
  unsigned tag;
};
struct Interval {
  std::vector<unsigned char> must, may;
  size_t sequence;
};
// Test oracle only. The index never uses this journal. Every API operation is
// enclosed in begin/end; uncertainty includes publication inside the API call.
struct Journal {
  std::mutex mutex;
  std::vector<unsigned char> live;
  std::vector<Event> events;
  std::array<Event, 2> pending{ { { '-', 0 }, { '-', 0 } } };
  explicit Journal(unsigned n, unsigned initial = 10000) : live(n, 0) {
    check(initial <= n, "journal capacity");
    std::fill(live.begin(), live.begin() + initial, 1);
  }
  void begin(unsigned worker, Event e) {
    std::lock_guard<std::mutex> g(mutex);
    pending[worker] = e;
    events.push_back(e);
  }
  void end(unsigned worker) {
    std::lock_guard<std::mutex> g(mutex);
    auto e = pending[worker];
    if (e.kind == 'i')
      live[e.tag] = 1;
    if (e.kind == 'd')
      live[e.tag] = 0;
    pending[worker] = { '-', 0 };
  }
  static void uncertain(Interval &s, Event e) {
    if (e.kind == 'i')
      s.may[e.tag] = 1;
    if (e.kind == 'd')
      s.must[e.tag] = 0;
  }
  Interval before() {
    std::lock_guard<std::mutex> g(mutex);
    Interval s{ live, live, events.size() };
    for (auto e : pending)
      uncertain(s, e);
    return s;
  }
  void after(Interval &s) {
    std::lock_guard<std::mutex> g(mutex);
    for (size_t j = s.sequence; j < events.size(); ++j)
      uncertain(s, events[j]);
  }
};
struct Oracle {
  unsigned d1, d2, n;
  std::vector<float> x, y;
  double distance(unsigned id, const float *qx, const float *qy,
                  float a) const {
    double dx = 0, dy = 0;
    for (unsigned j = 0; j < d1; ++j) {
      double v = double(qx[j]) - x[uint64_t(id) * d1 + j];
      dx += v * v;
    }
    for (unsigned j = 0; j < d2; ++j) {
      double v = double(qy[j]) - y[uint64_t(id) * d2 + j];
      dy += v * v;
    }
    return double(a) * std::sqrt(dx) + (1 - double(a)) * std::sqrt(dy);
  }
  std::vector<unsigned> truth(const float *qx, const float *qy, float a,
                              const std::vector<unsigned char> &mask) const {
    std::vector<std::pair<double, unsigned> > pool;
    pool.reserve(n);
    for (unsigned id = 0; id < n; ++id)
      if (mask[id])
        pool.emplace_back(distance(id, qx, qy, a), id);
    check(pool.size() >= 10, "less than ten oracle points");
    std::partial_sort(pool.begin(), pool.begin() + 10, pool.end());
    std::vector<unsigned> out;
    for (unsigned j = 0; j < 10; ++j)
      out.push_back(pool[j].second);
    return out;
  }
  unsigned hits(const std::vector<QueryResult> &returned,
                const std::vector<unsigned> &truth,
                const std::vector<unsigned char> &must) const {
    std::set<uint64_t> seen;
    unsigned hits = 0;
    for (auto r : returned) {
      check(r.tag < n && seen.insert(r.tag).second,
            "invalid/duplicate result tag");
      if (must[r.tag] &&
          std::find(truth.begin(), truth.end(), r.tag) != truth.end())
        ++hits;
    }
    return hits;
  }
};
struct Counters {
  uint64_t queries = 0, hits = 0, lock_ns = 0;
  std::deque<unsigned> window;
  double min_window = 1;
  unsigned windows = 0;
};
static void oracle_self_test() {
  Journal journal(14, 10);
  journal.begin(0, { 'i', 10 });
  journal.begin(1, { 'd', 2 });
  auto interval = journal.before();
  check(interval.may[10] && !interval.must[10] && interval.may[2] &&
            !interval.must[2],
        "in-flight operations missing from interval");
  journal.end(0);
  journal.end(1);
  journal.begin(0, { 'i', 11 });
  journal.end(0);
  journal.after(interval);
  check(interval.may[11] && !interval.must[11],
        "operation inside query interval missing");
  auto next = journal.before();
  check(next.must[10] && next.must[11] && !next.may[2],
        "completed operations not reflected");
  Oracle oracle;
  oracle.n = 14;
  oracle.d1 = oracle.d2 = 1;
  for (unsigned i = 0; i < 14; ++i) {
    oracle.x.push_back(i + 1);
    oracle.y.push_back(i + 1);
  }
  std::vector<unsigned char> may(14, 1), must(14, 1);
  std::array<unsigned, 4> uncertain{ { 0, 3, 8, 12 } };
  for (auto id : uncertain)
    must[id] = 0;
  float query = 0;
  auto may_truth = oracle.truth(&query, &query, .5f, may);
  std::vector<QueryResult> returned;
  for (unsigned i = 4; i < 14; ++i)
    returned.push_back({ i, i, float(i + 1) });
  auto bound = oracle.hits(returned, may_truth, must);
  for (unsigned bits = 0; bits < 16; ++bits) {
    auto live = must;
    for (unsigned j = 0; j < 4; ++j)
      if (bits & (1u << j))
        live[uncertain[j]] = 1;
    auto truth = oracle.truth(&query, &query, .5f, live);
    check(bound <= oracle.hits(returned, truth, live),
          "interval bound exceeds a possible live-state recall");
  }
  std::cout << "G3_ORACLE_SELF_TEST_PASS possible_live_states=16" << std::endl;
}
int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--oracle-self-test") {
      oracle_self_test();
      return 0;
    }
    check(argc == 7 || argc == 8,
          "fixture initial-x initial-y query-x query-y seconds [pq|exact]");
    std::string root = argv[1];
    unsigned seconds = std::stoul(argv[6]);
    check(seconds >= 1, "duration");
    unsigned n, d, n2, d2, ni, di, ni2, di2, nq, dq, nq2, dq2;
    Oracle oracle;
    oracle.x = matrix(argv[2], n, d);
    oracle.y = matrix(argv[3], n2, d2);
    auto ix = matrix(root + "/insert-x.bin", ni, di),
         iy = matrix(root + "/insert-y.bin", ni2, di2);
    auto qx = matrix(argv[4], nq, dq), qy = matrix(argv[5], nq2, dq2);
    check(n == 10000 && n2 == n && ni >= 591 && ni2 == ni && di == d &&
              di2 == d2 && nq2 == nq && dq == d && dq2 == d2,
          "oracle dimensions");
    oracle.x.insert(oracle.x.end(), ix.begin(), ix.end());
    oracle.y.insert(oracle.y.end(), iy.begin(), iy.end());
    oracle.n = n + ni;
    oracle.d1 = d;
    oracle.d2 = d2;
    Options options;
    if (argc == 8) {
      check(std::string(argv[7]) == "pq" || std::string(argv[7]) == "exact",
            "reprune mode");
      options.exact_reprune = std::string(argv[7]) == "exact";
    }
    DualUpdate index(root, options);
    Journal journal(oracle.n);
    auto cohort = [&](const std::string &s) {
      std::ifstream f(root + "/" + s + "-ids.txt");
      std::vector<unsigned> ids;
      unsigned q;
      while (f >> q) {
        check(q < nq, "query ID range");
        ids.push_back(q);
      }
      check(ids.size() == 500, "cohort must have500 IDs");
      return ids;
    };
    auto tune = cohort("tune"), report = cohort("report");
    std::array<float, 3> alphas{ { .1f, .5f, .9f } };
    std::array<unsigned, 3> budgets{};
    std::ofstream calibration(root + "/calibration.jsonl");
    auto live = journal.before().must;
    for (unsigned a = 0; a < 3; ++a) {
      std::vector<std::vector<unsigned> > truths;
      for (auto q : tune)
        truths.push_back(oracle.truth(qx.data() + uint64_t(q) * d,
                                      qy.data() + uint64_t(q) * d2, alphas[a],
                                      live));
      for (unsigned L : { 32u, 64u, 128u, 256u, 512u, 1024u }) {
        uint64_t hits = 0;
        for (unsigned j = 0; j < tune.size(); ++j) {
          auto q = tune[j];
          QueryWork w;
          auto r = index.search(qx.data() + uint64_t(q) * d,
                                qy.data() + uint64_t(q) * d2, alphas[a], 10, L,
                                L, w);
          hits += oracle.hits(r, truths[j], live);
        }
        double recall = double(hits) / (tune.size() * 10);
        calibration << "{\"alpha\":" << alphas[a]
                    << ",\"cohort\":\"tune\",\"L\":" << L
                    << ",\"recall\":" << recall << "}\n";
        calibration.flush();
        if (recall >= .955) {
          budgets[a] = L;
          break;
        }
      }
      check(budgets[a], "static tune cannot reach95.5%");
      uint64_t hits = 0;
      for (auto q : report) {
        auto truth =
            oracle.truth(qx.data() + uint64_t(q) * d,
                         qy.data() + uint64_t(q) * d2, alphas[a], live);
        QueryWork w;
        auto r = index.search(qx.data() + uint64_t(q) * d,
                              qy.data() + uint64_t(q) * d2, alphas[a], 10,
                              budgets[a], budgets[a], w);
        hits += oracle.hits(r, truth, live);
      }
      double recall = double(hits) / (report.size() * 10);
      calibration << "{\"alpha\":" << alphas[a]
                  << ",\"cohort\":\"report\",\"L\":" << budgets[a]
                  << ",\"recall\":" << recall << "}\n";
      calibration.flush();
      check(recall >= .95, "held-out static prerequisite below95%");
    }
    std::vector<Event> plan;
    for (unsigned i = 0; i < 591; ++i)
      plan.push_back({ 'i', 10000 + i });
    std::vector<unsigned> delete_ids(10000);
    std::iota(delete_ids.begin(), delete_ids.end(), 0);
    std::mt19937 rng(20261116);
    std::shuffle(delete_ids.begin(), delete_ids.end(), rng);
    for (unsigned i = 0; i < 309; ++i)
      plan.push_back({ 'd', delete_ids[i] });
    for (unsigned i = 0; i < 100; ++i)
      plan.push_back({ 'm', 0 });
    std::shuffle(plan.begin(), plan.end(), rng);
    std::ofstream raw(root + "/concurrent-queries.jsonl"),
        operations(root + "/concurrent-operations.jsonl");
    std::mutex output, errors;
    std::exception_ptr error;
    std::atomic<bool> stop{ false };
    std::atomic<unsigned> next_query{ 0 }, next_op{ 0 }, completed{ 0 };
    std::array<Counters, 3> counts;
    std::atomic<uint64_t> retries{ 0 }, degree_retries{ 0 }, max_batch{ 0 };
    auto start = Clock::now(), end = start + std::chrono::seconds(seconds);
    auto failure = [&] {
      stop = true;
      std::lock_guard<std::mutex> g(errors);
      if (!error)
        error = std::current_exception();
    };
    std::vector<std::thread> workers;
    for (unsigned t = 0; t < 4; ++t)
      workers.emplace_back([&] {
        try {
          while (!stop && Clock::now() < end) {
            unsigned seq = next_query.fetch_add(1), a = seq % 3;
            unsigned qi = report[(seq / 3) % report.size()];
            auto interval = journal.before();
            QueryWork work;
            auto r = index.search(qx.data() + uint64_t(qi) * d,
                                  qy.data() + uint64_t(qi) * d2, alphas[a], 10,
                                  budgets[a], budgets[a], work);
            journal.after(interval);
            auto truth = oracle.truth(qx.data() + uint64_t(qi) * d,
                                      qy.data() + uint64_t(qi) * d2, alphas[a],
                                      interval.may);
            for (auto v : r)
              check(v.tag < oracle.n && interval.may[v.tag],
                    "returned point absent throughout query interval");
            auto hits = oracle.hits(r, truth, interval.must);
            unsigned uncertain = 0;
            for (unsigned j = 0; j < oracle.n; ++j)
              uncertain += interval.may[j] != interval.must[j];
            std::lock_guard<std::mutex> g(output);
            auto &s = counts[a];
            ++s.queries;
            s.hits += hits;
            s.lock_ns += work.io.page_lock_wait_ns;
            s.window.push_back(hits);
            if (s.window.size() > 500)
              s.window.pop_front();
            if (s.window.size() == 500) {
              double recall =
                  std::accumulate(s.window.begin(), s.window.end(), 0.) / 5000.;
              s.min_window = std::min(s.min_window, recall);
              ++s.windows;
            }
            raw << "{\"sequence\":" << seq << ",\"query_id\":" << qi
                << ",\"alpha\":" << alphas[a] << ",\"guaranteed_hits\":" << hits
                << ",\"uncertain_points\":" << uncertain
                << ",\"lock_wait_ns\":" << work.io.page_lock_wait_ns
                << ",\"seconds\":"
                << std::chrono::duration<double>(Clock::now() - start).count()
                << "}\n";
            if (seq % 100 == 0)
              raw.flush();
          }
        }
        catch (...) {
          failure();
        }
      });
    for (unsigned worker = 0; worker < 2; ++worker)
      workers.emplace_back([&, worker] {
        try {
          while (!stop) {
            auto i = next_op.fetch_add(1);
            if (i >= plan.size())
              break;
            std::this_thread::sleep_until(
                start +
                std::chrono::milliseconds(
                    (i < 32 ? 0 : uint64_t(i) * seconds * 1000 / plan.size())));
            if (stop)
              break;
            auto e = plan[i];
            journal.begin(worker, e);
            IOStats stats;
            if (e.kind == 'i') {
              unsigned j = e.tag - 10000;
              index.insert(ix.data() + uint64_t(j) * d,
                           iy.data() + uint64_t(j) * d2, e.tag, stats);
            } else if (e.kind == 'd')
              check(index.erase(e.tag), "delete failed");
            else
              index.consolidate(stats);
            retries.fetch_add(stats.optimistic_retries);
            degree_retries.fetch_add(stats.degree_reservation_retries);
            auto seen = max_batch.load();
            while (
                seen < stats.max_read_batch &&
                !max_batch.compare_exchange_weak(seen, stats.max_read_batch)) {
            }
            journal.end(worker);
            ++completed;
            std::lock_guard<std::mutex> g(output);
            operations << "{\"operation\":" << i << ",\"kind\":\"" << e.kind
                       << "\",\"tag\":" << e.tag << ",\"seconds\":"
                       << std::chrono::duration<double>(Clock::now() - start)
                              .count() << "}\n";
            operations.flush();
          }
        }
        catch (...) {
          failure();
        }
      });
    for (auto &t : workers)
      t.join();
    if (error)
      std::rethrow_exception(error);
    double elapsed =
        std::chrono::duration<double>(Clock::now() - start).count();
    IOStats audit_io;
    auto graph = index.audit_topology(audit_io);
    std::vector<uint32_t> indegree(index.allocated(), 0);
    for (uint32_t id = 0; id < graph.size(); ++id)
      if (index.committed(id)) {
        check(graph[id].size() <= 40, "post-stress degree");
        std::set<uint32_t> seen;
        for (auto v : graph[id]) {
          check(v < graph.size() && index.committed(v) && v != id &&
                    seen.insert(v).second,
                "post-stress invalid edge");
          ++indegree[v];
        }
      }
    check(indegree == index.indegree_snapshot(),
          "post-stress atomic indegree mismatch");
    auto positive = index.positive_indegree_history();
    for (uint32_t id = 0; id < graph.size(); ++id)
      if (index.committed(id) && !index.deleted(id))
        check(!(positive[id] && indegree[id] == 0),
              "post-stress lost last incoming edge");
    bool pass = completed == plan.size();
    std::ofstream summary(root + "/concurrent-result.json");
    summary << "{\"requested_seconds\":" << seconds
            << ",\"elapsed_seconds\":" << elapsed
            << ",\"update_operations\":" << completed
            << ",\"initial_burst_operations\":32,\"optimistic_retries\":"
            << retries.load()
            << ",\"degree_reservation_retries\":" << degree_retries.load()
            << ",\"max_read_batch\":" << max_batch.load()
            << ",\"post_stress_indegree_verified\":true,\"query_threads\":4,"
               "\"update_threads\":2,\"weights\":[";
    for (unsigned a = 0; a < 3; ++a) {
      auto &s = counts[a];
      double recall = s.queries ? double(s.hits) / (10 * s.queries) : 0;
      pass = pass && s.windows && recall >= .93 && s.min_window >= .93;
      if (a)
        summary << ",";
      summary << "{\"alpha\":" << alphas[a] << ",\"L\":" << budgets[a]
              << ",\"queries\":" << s.queries
              << ",\"recall_lower_bound\":" << recall
              << ",\"min_500_query_window_lower_bound\":" << s.min_window
              << ",\"windows\":" << s.windows
              << ",\"page_lock_wait_ns\":" << s.lock_ns << "}";
    }
    summary << "],\"passed\":" << (pass ? "true" : "false") << "}\n";
    summary.close();
    check(pass, "G3 recall/workload gate failed");
    std::cout << "G3_CONCURRENT_PASS seconds=" << seconds << std::endl;
  }
  catch (const std::exception &e) {
    std::cerr << "G3_CONCURRENT_FAIL " << e.what() << std::endl;
    return 1;
  }
}
