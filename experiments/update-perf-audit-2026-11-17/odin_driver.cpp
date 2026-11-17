// External harness for the unmodified OdinANN0.1.1 API.
#include "v2/dynamic_index.h"
#include "linux_aligned_file_reader.h"
#include <algorithm>
#include <numeric>
#include <vector>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>
#include <fcntl.h>
#include <unistd.h>
using Clock = std::chrono::steady_clock;
static void check(bool b, const char *s) {
  if (!b)
    throw std::runtime_error(s);
}
static void sync_index(const std::string &path) {
  int fd = ::open(path.c_str(), O_RDWR);
  check(fd >= 0, "native sync open");
  int rc = ::fsync(fd);
  ::close(fd);
  check(rc == 0, "native fsync");
}
static std::vector<float> matrix(const std::string &p, unsigned &n, unsigned &d,
                                 unsigned limit = 0) {
  std::ifstream f(p, std::ios::binary);
  f.read(reinterpret_cast<char *>(&n), 4);
  f.read(reinterpret_cast<char *>(&d), 4);
  check(bool(f) && n && d, "matrix header");
  if (limit) {
    check(limit <= n, "prefix");
    n = limit;
  }
  std::vector<float> a(uint64_t(n) * d);
  f.read(reinterpret_cast<char *>(a.data()), a.size() * 4);
  check(bool(f), "matrix body");
  return a;
}
int main(int argc, char **argv) {
  try {
    check(
        argc == 8,
        "index-prefix insert-matrix query-matrix writers readers count output");
    std::string prefix = argv[1], out = argv[7];
    unsigned writers = std::stoul(argv[4]), readers = std::stoul(argv[5]),
             count = std::stoul(argv[6]);
    unsigned n, d, nq, dq;
    auto x = matrix(argv[2], n, d, count), q = matrix(argv[3], nq, dq);
    check(d == dq, "dimensions");
    pipeann::Parameters p;
    p.Set<unsigned>("L_disk", 200);
    p.Set<unsigned>("R_disk", 0);
    p.Set<unsigned>("C", 384);
    p.Set<float>("alpha_disk", 1.2f);
    p.Set<unsigned>("beamwidth", 4);
    p.Set<unsigned>("nodes_to_cache", 0);
    p.Set<unsigned>("num_threads", writers + readers);
    auto distance = pipeann::get_distance_function<float>(pipeann::Metric::L2);
    pipeann::DynamicSSDIndex<float, uint32_t> index(
        p, prefix, prefix + "_merge", distance, pipeann::Metric::L2,
        BEAM_SEARCH, false);
    auto initial = index._disk_index->num_points;
    check(index._disk_index->data_dim == d && initial == 1800000,
          "native index identity");
    // The native constructor makes a buffered shadow copy. Complete its writes
    // before starting, as for the dual-space fixture, so copy writeback is
    // excluded.
    sync_index(index._disk_index_prefix_in + "_disk.index");
    std::atomic<unsigned> next{ 0 }, query_seq{ 0 };
    std::atomic<bool> stop{ false }, failed{ false };
    std::exception_ptr error;
    std::mutex mutex;
    std::vector<uint32_t> returned(count);
    std::vector<uint64_t> query_count(readers);
    std::vector<std::thread> rq, wr;
    auto fail = [&] {
      failed = true;
      std::lock_guard<std::mutex> g(mutex);
      if (!error)
        error = std::current_exception();
    };
    auto start = Clock::now();
    for (unsigned t = 0; t < readers; ++t)
      rq.emplace_back([&, t] {
        try {
          while (!stop && !failed) {
            unsigned i = query_seq.fetch_add(1) % nq;
            uint32_t tags[10];
            float distances[10];
            pipeann::QueryStats stats;
            index.search(q.data() + uint64_t(i) * d, 10, 0, 96, 4, tags,
                         distances, &stats, false);
            ++query_count[t];
          }
        }
        catch (...) {
          fail();
        }
      });
    for (unsigned t = 0; t < writers; ++t)
      wr.emplace_back([&] {
        try {
          while (!failed) {
            unsigned i = next.fetch_add(1);
            if (i >= count)
              break;
            returned[i] = index.insert(x.data() + uint64_t(i) * d, initial + i);
          }
        }
        catch (...) {
          fail();
        }
      });
    for (auto &t : wr)
      t.join();
    auto api_end = Clock::now();
    stop = true;
    // A pending background write retains its query buffer until I/O completion.
    // Acquiring every allocated buffer is a public-API drain barrier;
    // queue.empty
    // alone would fail to account for a task already popped by the I/O thread.
    std::vector<pipeann::QueryBuffer<float> *> held;
    for (unsigned i = 0; i < 2 * (writers + readers); ++i)
      held.push_back(index._disk_index->pop_query_buf(nullptr));
    auto drained = Clock::now();
    for (auto b : held)
      index._disk_index->push_query_buf(b);
    for (auto &t : rq)
      t.join();
    auto queries_end = Clock::now();
    if (error)
      std::rethrow_exception(error);
    auto sorted = returned;
    std::sort(sorted.begin(), sorted.end());
    for (unsigned i = 0; i < count; ++i)
      check(sorted[i] == initial + i, "native insertion ID set");
    double api = std::chrono::duration<double>(api_end - start).count(),
           complete = std::chrono::duration<double>(drained - start).count();
    std::ofstream f(out + "/result.json");
    f << std::setprecision(17) << "{\"count\":" << count
      << ",\"writers\":" << writers << ",\"readers\":" << readers
      << ",\"initial\":" << initial << ",\"dimension\":" << d
      << ",\"sector_bytes\":" << SECTOR_LEN << ",\"api_wall_seconds\":" << api
      << ",\"api_insert_qps\":" << count / api
      << ",\"drained_wall_seconds\":" << complete
      << ",\"drained_insert_qps\":" << count / complete
      << ",\"drain_after_api_seconds\":" << complete - api
      << ",\"query_tail_seconds\":"
      << std::chrono::duration<double>(queries_end - api_end).count()
      << ",\"query_count\":"
      << std::accumulate(query_count.begin(), query_count.end(), uint64_t(0))
      << ",\"outstanding_bg_queue_after_barrier\":"
      << index._disk_index->bg_tasks.size()
      << ",\"transient_page_buffers_after_barrier\":" << v2::cache.cache.size()
      << ",\"search_mode\":\"BEAM_SEARCH\",\"L_search\":96,\"beam_width\":4,"
         "\"static_cached_nodes\":0}\n";
    f.close();
    // Persistence fence is outside both API and completed-write timing
    // intervals.
    sync_index(index._disk_index_prefix_in + "_disk.index");
    std::cout << "ODIN_REFERENCE_PASS insertions=" << count << std::endl;
  }
  catch (const std::exception &e) {
    std::cerr << "ODIN_REFERENCE_FAIL " << e.what() << std::endl;
    return 1;
  }
}
