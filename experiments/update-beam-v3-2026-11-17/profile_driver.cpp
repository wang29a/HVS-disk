#include "dual_update.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <thread>
using namespace hvs_update;
using Clock = std::chrono::steady_clock;
static void check(bool b, const char *s) {
  if (!b)
    throw std::runtime_error(s);
}
static std::vector<float> matrix(const std::string &p, unsigned &n, unsigned &d,
                                 unsigned limit = 0) {
  std::ifstream f(p, std::ios::binary);
  f.read(reinterpret_cast<char *>(&n), 4);
  f.read(reinterpret_cast<char *>(&d), 4);
  check(bool(f) && n && d, "matrix header");
  if (limit) {
    check(limit <= n, "matrix prefix");
    n = limit;
  }
  std::vector<float> a(uint64_t(n) * d);
  f.read(reinterpret_cast<char *>(a.data()), a.size() * 4);
  check(bool(f), "matrix body");
  return a;
}
static double percentile(std::vector<double> v, double p) {
  if (v.empty())
    return 0;
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, size_t(p * v.size()))];
}
static void fields(std::ostream &o, const IOStats &s) {
  o << ",\"topology_read_pages\":" << s.read_pages[0]
    << ",\"coordinate_read_pages\":" << s.read_pages[1]
    << ",\"topology_write_pages\":" << s.write_pages[0]
    << ",\"coordinate_write_pages\":" << s.write_pages[1]
    << ",\"page_lock_wait_ns\":" << s.page_lock_wait_ns;
  o << ",\"read_batches\":" << s.read_batches
    << ",\"write_batches\":" << s.write_batches
    << ",\"max_read_batch\":" << s.max_read_batch
    << ",\"optimistic_retries\":" << s.optimistic_retries;
#ifdef HVS_PROFILE_PHASES
  o << ",\"phase_ns\":[";
  for (unsigned i = 0; i < 7; ++i) {
    if (i)
      o << ",";
    o << s.phase_ns[i];
  }
  o << "]";
#define FIELD(x) o << ",\"" #x "\":" << s.x
  FIELD(insert_total_ns);
  FIELD(reprune_ns);
  FIELD(coordinate_calls);
  FIELD(coordinate_cache_hits);
  FIELD(coordinate_materialized_bytes);
#undef FIELD
  for (unsigned f = 0; f < 2; ++f)
    o << ",\"read_io_ns_" << f << "\":" << s.read_io_ns[f] << ",\"write_io_ns_"
      << f << "\":" << s.write_io_ns[f];
#endif
}
int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--clock-cost") {
      const unsigned n = 10000000;
      volatile uint64_t sink = 0;
      auto start = Clock::now();
      for (unsigned i = 0; i < n; ++i)
        sink ^= i;
      auto middle = Clock::now();
      for (unsigned i = 0; i < n; ++i)
        sink ^= Clock::now().time_since_epoch().count();
      auto end = Clock::now();
      double plain =
          std::chrono::duration<double, std::nano>(middle - start).count() / n;
      double clock =
          std::chrono::duration<double, std::nano>(end - middle).count() / n;
      std::cout << std::setprecision(17) << "{\"calls\":" << n
                << ",\"loop_ns_per_iteration\":" << plain
                << ",\"clock_loop_ns_per_call\":" << clock
                << ",\"net_clock_ns_estimate\":" << std::max(0., clock - plain)
                << ",\"sink\":" << sink << "}" << std::endl;
      return 0;
    }
    check(argc == 9,
          "fixture pq|exact writers readers lean|e1 count output snapshot0|1");
    std::string root = argv[1], mode = argv[2], telemetry = argv[5],
                out = argv[7];
    unsigned writers = std::stoul(argv[3]), readers = std::stoul(argv[4]),
             count = std::stoul(argv[6]);
    check((mode == "pq" || mode == "exact") &&
              (telemetry == "lean" || telemetry == "e1") && writers && count,
          "options");
    bool logged = telemetry == "e1", snapshot = std::stoi(argv[8]);
    Options options;
    options.exact_reprune = mode == "exact";
    options.update_threads = writers;
    DualUpdate index(root, options);
    unsigned n, d, n2, d2, nq, dq, nq2, dq2;
    auto x = matrix(root + "/insert-x.bin", n, d, count),
         y = matrix(root + "/insert-y.bin", n2, d2, count);
    auto qx = matrix(root + "/query-x.bin", nq, dq),
         qy = matrix(root + "/query-y.bin", nq2, dq2);
    check(n == n2 && nq == nq2 && d == index.dim1() && d2 == index.dim2() &&
              dq == d && dq2 == d2,
          "dimensions");
    std::vector<unsigned> query_ids;
    std::ifstream ids(root + "/report-ids.txt");
    unsigned qi;
    while (ids >> qi) {
      check(qi < nq, "query id");
      query_ids.push_back(qi);
    }
    check(query_ids.size() == 500, "cohort");
    std::atomic<unsigned> next{ 0 }, done{ 0 }, seq{ 0 };
    std::atomic<bool> stop{ false }, failed{ false };
    std::mutex mu;
    std::exception_ptr error;
    struct Sample {
      unsigned ordinal, id;
      double us;
      IOStats io;
    };
    struct QSample {
      unsigned sequence, qi;
      float alpha;
      double us;
      uint64_t wait;
    };
    std::vector<std::vector<Sample> > samples(writers);
    std::vector<std::vector<QSample> > qs(readers);
    std::vector<uint32_t> tags(index.capacity());
    std::iota(tags.begin(), tags.end(), 0);
    std::vector<unsigned> inserted(writers), queries(readers);
    auto fail = [&] {
      failed = true;
      std::lock_guard<std::mutex> g(mu);
      if (!error)
        error = std::current_exception();
    };
    std::vector<std::thread> rq, wr;
    std::array<float, 5> weights{ { .1f, .3f, .5f, .7f, .9f } };
    auto start = Clock::now();
    for (unsigned t = 0; t < readers; ++t)
      rq.emplace_back([&, t] {
        try {
          while (!stop && !failed) {
            auto number = seq.fetch_add(1);
            auto q = query_ids[(number / 5) % query_ids.size()];
            float a = weights[number % 5];
            QueryWork work;
            auto tick = logged ? Clock::now() : Clock::time_point{};
            auto result =
                index.search(qx.data() + uint64_t(q) * d,
                             qy.data() + uint64_t(q) * d2, a, 10, 96, 96, work);
            if (logged) {
              for (auto r : result)
                check(r.tag < index.initial_points() + count, "query tag");
              qs[t].push_back({ number, q, a,
                                std::chrono::duration<double, std::micro>(
                                    Clock::now() - tick).count(),
                                work.io.page_lock_wait_ns });
            }
            ++queries[t];
          }
        }
        catch (...) {
          fail();
        }
      });
    for (unsigned t = 0; t < writers; ++t)
      wr.emplace_back([&, t] {
        try {
          while (!failed) {
            auto ordinal = next.fetch_add(1);
            if (ordinal >= count)
              break;
            IOStats io;
            auto tick = logged ? Clock::now() : Clock::time_point{};
            auto id = index.insert(x.data() + uint64_t(ordinal) * d,
                                   y.data() + uint64_t(ordinal) * d2,
                                   index.initial_points() + ordinal, io);
            if (logged) {
              tags[id] = index.initial_points() + ordinal;
              samples[t].push_back({ ordinal, id,
                                     std::chrono::duration<double, std::micro>(
                                         Clock::now() - tick).count(),
                                     io });
              auto completed = done.fetch_add(1) + 1;
              if (completed % 1000 == 0) {
                std::lock_guard<std::mutex> g(mu);
                std::cout << "INSERT_PROGRESS " << completed << "/" << count
                          << std::endl;
              }
            }
            ++inserted[t];
          }
        }
        catch (...) {
          fail();
        }
      });
    for (auto &t : wr)
      t.join();
    auto end = Clock::now();
    stop = true;
    for (auto &t : rq)
      t.join();
    auto drained = Clock::now();
    if (error)
      std::rethrow_exception(error);
    check(std::accumulate(inserted.begin(), inserted.end(), 0u) == count,
          "incomplete insertions");
    double seconds = std::chrono::duration<double>(end - start).count();
    std::ofstream raw(out + "/inserts.jsonl");
    std::vector<double> latency;
    for (auto &v : samples)
      for (auto &s : v) {
        latency.push_back(s.us);
        raw << std::setprecision(17) << "{\"ordinal\":" << s.ordinal
            << ",\"id\":" << s.id << ",\"latency_us\":" << s.us;
        fields(raw, s.io);
        raw << "}\n";
      }
    std::ofstream qraw(out + "/queries.jsonl");
    for (auto &v : qs)
      for (auto &s : v)
        qraw << std::setprecision(17) << "{\"sequence\":" << s.sequence
             << ",\"query_id\":" << s.qi << ",\"alpha\":" << s.alpha
             << ",\"latency_us\":" << s.us
             << ",\"page_lock_wait_ns\":" << s.wait << "}\n";
    std::ofstream result(out + "/result.json");
    result << std::setprecision(17) << "{\"count\":" << count
           << ",\"writers\":" << writers << ",\"readers\":" << readers
           << ",\"mode\":\"" << mode << "\",\"telemetry\":\"" << telemetry
           << "\",\"wall_seconds\":" << seconds
           << ",\"insert_qps\":" << count / seconds << ",\"queries\":"
           << std::accumulate(queries.begin(), queries.end(), uint64_t(0))
           << ",\"query_drain_seconds\":"
           << std::chrono::duration<double>(drained - end).count()
           << ",\"mean_insert_us\":"
           << (latency.empty() ? 0 : std::accumulate(latency.begin(),
                                                     latency.end(), 0.) /
                                         latency.size())
           << ",\"p95_insert_us\":" << percentile(latency, .95) << "}\n";
    // Outside the insertion timer: visibility checks, optional snapshot and
    // sync.
    for (unsigned id = index.initial_points(); id < index.allocated(); ++id)
      check(index.committed(id), "uncommitted insert");
    if (snapshot)
      index.snapshot(out + "/normal-exit.snapshot");
    std::cout << "PROFILE_DRIVER_PASS insertions=" << count << std::endl;
  }
  catch (const std::exception &e) {
    std::cerr << "PROFILE_DRIVER_FAIL " << e.what() << std::endl;
    return 1;
  }
}
