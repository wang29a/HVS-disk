// Driver for two unmodified OdinANN engines. Each gets its own process-global
// page buffer. One application worker drives both engines serially.
#include "v2/dynamic_index.h"
#include "linux_aligned_file_reader.h"
#include "memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <libaio.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>

using Clock = std::chrono::steady_clock;
constexpr unsigned writers = 4, query_threads = 4,
                   lanes = writers + query_threads;
constexpr unsigned dim = 512, coord_bytes = 8192, read_batch = 48;
static_assert(SECTOR_LEN == 4096, "native OdinANN page size must be 4 KiB");

struct Submitted {
  uint64_t reads = 0, bytes = 0;
};
static thread_local Submitted *observed_io = nullptr;

extern "C" int __real_io_submit(io_context_t, long, struct iocb **);
extern "C" int __wrap_io_submit(io_context_t ctx, long count,
                                struct iocb **requests) {
  const int submitted = __real_io_submit(ctx, count, requests);
  if (observed_io)
    for (int i = 0; i < submitted; ++i) {
      if (requests[i]->aio_lio_opcode == IO_CMD_PREAD) {
        ++observed_io->reads;
        observed_io->bytes += requests[i]->u.c.nbytes;
      }
    }
  return submitted;
}

static void check(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}

static double seconds(Clock::time_point t) {
  return std::chrono::duration<double>(Clock::now() - t).count();
}

template <class T> struct Matrix {
  unsigned n = 0, d = 0;
  std::vector<T> data;

  explicit Matrix(const std::string &path, unsigned limit = 0) {
    std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char *>(&n), sizeof(n));
    f.read(reinterpret_cast<char *>(&d), sizeof(d));
    check(f && n && d && limit <= n, "matrix header");
    if (limit)
      n = limit;
    data.resize(uint64_t(n) * d);
    check(bool(f.read(reinterpret_cast<char *>(data.data()),
                      data.size() * sizeof(T))),
          "matrix body");
  }

  const T *row(unsigned i) const { return data.data() + uint64_t(i) * d; }
};

static void transfer(int fd, void *buffer, size_t size, bool outgoing) {
  auto *p = static_cast<char *>(buffer);
  while (size) {
    ssize_t count =
        outgoing ? send(fd, p, size, MSG_NOSIGNAL) : recv(fd, p, size, 0);
    if (count < 0 && errno == EINTR)
      continue;
    check(count > 0, "OdinANN worker channel closed");
    p += count;
    size -= count;
  }
}

template <class T> static void put(int fd, T value) {
  transfer(fd, &value, sizeof(value), true);
}
template <class T> static T take(int fd) {
  T out{};
  transfer(fd, &out, sizeof(out), false);
  return out;
}

enum Op : uint32_t {
  query,
  insert,
  drain,
  flush,
  quit
};
struct Request {
  Op op;
  uint32_t row = 0, depth = 0, candidates = 0;
};
struct Candidates {
  uint64_t pages = 0;
  uint32_t count = 0;
};

static void durable_index(const std::string &prefix) {
  int fd = open((prefix + "_disk.index").c_str(), O_RDWR);
  check(fd >= 0, "native shadow open");
  int rc = fsync(fd);
  close(fd);
  check(rc == 0, "native fsync");
}

static void native(const std::array<int, lanes + 1> &fds,
                   const std::string &prefix, const Matrix<float> &points,
                   const Matrix<float> &queries, unsigned initial,
                   unsigned active_lanes) {
  pipeann::Parameters p;
  p.Set<unsigned>("L_disk", 200);
  p.Set<unsigned>("R_disk", 0);
  p.Set<unsigned>("C", 384);
  p.Set<float>("alpha_disk", 1.2f);
  p.Set<unsigned>("beamwidth", 4);
  p.Set<unsigned>("nodes_to_cache", 0);
  p.Set<unsigned>("num_threads", active_lanes);
  auto distance = std::unique_ptr<pipeann::Distance<float> >(
      pipeann::get_distance_function<float>(pipeann::Metric::L2));
  pipeann::DynamicSSDIndex<float, uint32_t> index(
      p, prefix, prefix + "_merge", distance.get(), pipeann::Metric::L2,
      BEAM_SEARCH, false);
  check(index._disk_index->num_points == initial &&
            index._disk_index->data_dim == dim &&
            index._disk_index_prefix_in == prefix + "_shadow",
        "unexpected native index identity");
  durable_index(index._disk_index_prefix_in);
  std::vector<std::thread> active;
  for (unsigned lane = 0; lane < active_lanes; ++lane) {
    const int fd = fds[lane];
    active.emplace_back([&, fd] {
      try {
        for (;;) {
          const auto req = take<Request>(fd);
          if (req.op == quit)
            return;
          if (req.op == insert) {
            check(req.row < points.n, "insert ordinal");
            put(fd, index.insert(points.row(req.row), initial + req.row));
            continue;
          }
          check(req.op == query && req.row < queries.n && req.depth <= 1600 &&
                    req.candidates <= req.depth && req.candidates,
                "query request");
          std::vector<uint32_t> ids(req.candidates,
                                    std::numeric_limits<uint32_t>::max());
          std::vector<float> distances(req.candidates);
          pipeann::QueryStats stats;
          Submitted actual;
          observed_io = &actual;
          index.search(queries.row(req.row), req.candidates, 0, req.depth, 4,
                       ids.data(), distances.data(), &stats, false);
          observed_io = nullptr;
          ids.erase(std::remove(ids.begin(), ids.end(),
                                std::numeric_limits<uint32_t>::max()),
                    ids.end());
          for (auto id : ids)
            check(id < initial + points.n, "native query tag outside dataset");
          check(actual.reads <= stats.n_ios &&
                    actual.bytes == SECTOR_LEN * actual.reads,
                "unexpected native AIO submission counters");
          put(fd,
              Candidates{ actual.reads, static_cast<uint32_t>(ids.size()) });
          transfer(fd, ids.data(), ids.size() * sizeof(uint32_t), true);
        }
      }
      catch (const std::exception &error) {
        std::cerr << "OdinANN worker failed: " << error.what() << std::endl;
        _exit(2);
      }
    });
  }
  const auto control = fds[lanes];
  put(control, 0);
  for (;;) {
    const auto req = take<Request>(control);
    if (req.op == quit)
      break;
    if (req.op == drain) {
      std::vector<pipeann::QueryBuffer<float> *> held(2 * active_lanes);
      for (auto &buffer : held)
        buffer = index._disk_index->pop_query_buf(nullptr);
      check(index._disk_index->bg_tasks.size() == 0 &&
                v2::cache.cache.size() == 0,
            "native write drain incomplete");
      for (auto buffer : held)
        index._disk_index->push_query_buf(buffer);
    } else {
      check(req.op == flush, "control request");
      durable_index(index._disk_index_prefix_in);
    }
    put(control, 0);
  }
  for (auto &thread : active)
    thread.join();
  // The fixed upstream version owns detached background workers until exit.
  _exit(0);
}

class Pair {
public:
  Pair(const std::array<std::string, 2> &prefixes,
       const std::array<Matrix<float>, 2> &points,
       const std::array<Matrix<float>, 2> &queries, unsigned initial,
       unsigned active_lanes) {
    std::array<std::array<std::array<int, 2>, lanes + 1>, 2> pairs{};
    for (auto &side : pairs)
      for (auto &p : side)
        check(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, p.data()) == 0,
              "socketpair");
    const pid_t parent = getpid();
    for (unsigned side = 0; side < 2; ++side) {
      pids[side] = fork();
      check(pids[side] >= 0, "native worker fork");
      if (pids[side] == 0) {
        if (prctl(PR_SET_PDEATHSIG, SIGTERM) < 0 || getppid() != parent)
          _exit(2);
        for (unsigned s = 0; s < 2; ++s)
          for (unsigned lane = 0; lane <= lanes; ++lane) {
            const auto &p = pairs[s][lane];
            close(p[0]);
            if (s == side)
              channels[s][lane] = p[1];
            else
              close(p[1]);
          }
        try {
          native(channels[side], prefixes[side], points[side], queries[side],
                 initial, active_lanes);
        }
        catch (const std::exception &error) {
          std::cerr << "OdinANN initialization/control failed: " << error.what()
                    << std::endl;
          _exit(2);
        }
      }
    }
    for (unsigned side = 0; side < 2; ++side)
      for (unsigned lane = 0; lane <= lanes; ++lane) {
        channels[side][lane] = pairs[side][lane][0];
        close(pairs[side][lane][1]);
      }
    for (auto &side : channels)
      check(take<int>(side[lanes]) == 0, "native init reply");
  }

  ~Pair() {
    for (unsigned side = 0; side < 2; ++side) {
      for (int fd : channels[side])
        close(fd);
      if (pids[side] > 0) {
        kill(pids[side], SIGTERM);
        waitpid(pids[side], nullptr, 0);
      }
    }
  }

  void finish() {
    for (auto &side : channels)
      for (auto fd : side)
        put(fd, Request{ quit });
    for (auto &pid : pids) {
      int code = 0;
      check(waitpid(pid, &code, 0) == pid, "wait for native worker");
      pid = 0;
      check(WIFEXITED(code) && WEXITSTATUS(code) == 0,
            "native worker exit status");
    }
  }

  Memory memory() {
    Memory sample;
    sample.add(getpid());
    for (auto pid : pids)
      sample.add(pid);
    return sample;
  }

  void control(Op op) {
    for (auto &side : channels)
      put(side[lanes], Request{ op });
    for (auto &side : channels)
      check(take<int>(side[lanes]) == 0, "native control reply");
  }

  std::array<int, 2> add(unsigned ordinal, unsigned lane) {
    std::array<int, 2> result{};
    for (unsigned s = 0; s < 2; ++s) {
      auto fd = channels[s][lane];
      put(fd, Request{ insert, ordinal });
      result[s] = take<int>(fd);
      check(result[s] >= 0, "native insertion failed");
    }
    return result;
  }

  uint64_t search(unsigned side, unsigned lane, unsigned row, unsigned depth,
                  unsigned budget, std::vector<uint32_t> &out) {
    auto fd = channels[side][lane];
    put(fd, Request{ query, row, depth, budget });
    auto result = take<Candidates>(fd);
    check(result.count <= budget, "native candidate reply");
    auto old = out.size();
    out.resize(old + result.count);
    transfer(fd, out.data() + old, result.count * sizeof(uint32_t), false);
    return result.pages;
  }

private:
  std::array<std::array<int, lanes + 1>, 2> channels{};
  std::array<pid_t, 2> pids{};
};

struct SearchResult {
  std::array<uint32_t, 10> ids{};
  std::array<uint64_t, 2> native_pages{};
  uint64_t coordinate_pages = 0, reranked = 0;
  double latency_us = 0;
};

class CoordinateFile {
public:
  explicit CoordinateFile(const std::string &name, unsigned expected)
      : fd(open(name.c_str(), O_RDONLY | O_DIRECT)) {
    struct stat metadata;
    check(fd >= 0 && fstat(fd, &metadata) == 0 &&
              uint64_t(metadata.st_size) >=
                  uint64_t(expected) * 2 * dim * sizeof(float),
          "aligned coordinate sidecar");
  }
  ~CoordinateFile() { close(fd); }

  struct Scratch {
    Scratch() {
      void *p = nullptr;
      check(posix_memalign(&p, coord_bytes, read_batch * coord_bytes) == 0,
            "coordinate allocation");
      data.reset(static_cast<char *>(p));
    }
    std::unique_ptr<char, decltype(&std::free)> data{ nullptr, std::free };
    std::vector<IORequest> requests;
  };

  void rerank(std::vector<uint32_t> &ids, const float *x, const float *y,
              double alpha, Scratch &scratch, SearchResult &out) {
    check(ids.size() >= out.ids.size(),
          "fewer than ten paired visible candidates");
    alpha = static_cast<float>(alpha);
    out.reranked = ids.size();
    std::vector<std::pair<double, uint32_t> > scores;
    scores.reserve(ids.size());
    for (size_t i = 0; i < ids.size();) {
      auto &requests = scratch.requests;
      requests.clear();
      const size_t first = i;
      uint64_t prior = std::numeric_limits<uint64_t>::max();
      for (; i < ids.size(); ++i) {
        uint64_t page = ids[i] / 2;
        if (page == prior)
          continue;
        if (requests.size() == read_batch)
          break;
        requests.emplace_back(
            page * coord_bytes, coord_bytes,
            scratch.data.get() + requests.size() * coord_bytes, 0, 0);
        prior = page;
      }
      Submitted actual;
      observed_io = &actual;
      reader.read_fd(fd, requests, reader.get_ctx());
      observed_io = nullptr;
      check(actual.reads == requests.size() &&
                actual.bytes == actual.reads * coord_bytes,
            "coordinate AIO submission mismatch");
      out.coordinate_pages += actual.reads;
      size_t slot = 0;
      for (size_t j = first; j < i; ++j) {
        while (requests[slot].offset / coord_bytes != ids[j] / 2)
          ++slot;
        const float *xy = reinterpret_cast<const float *>(
            scratch.data.get() + slot * coord_bytes +
            (ids[j] % 2) * (2 * dim * sizeof(float)));
        double dx = 0, dy = 0;
        for (unsigned d = 0; d < dim; ++d) {
          const double vx = double(x[d]) - xy[d],
                       vy = double(y[d]) - xy[d + dim];
          dx += vx * vx;
          dy += vy * vy;
        }
        scores.emplace_back(
            alpha * std::sqrt(dx) + (1.0 - alpha) * std::sqrt(dy), ids[j]);
      }
    }
    std::partial_sort(scores.begin(), scores.begin() + out.ids.size(),
                      scores.end());
    for (size_t i = 0; i < out.ids.size(); ++i)
      out.ids[i] = scores[i].second;
  }

private:
  int fd;
  LinuxAlignedFileReader reader;
};

class Benchmark {
public:
  Benchmark(Pair &pair, CoordinateFile &coordinates,
            const std::array<Matrix<float>, 2> &queries, unsigned initial,
            unsigned capacity)
      : pair(pair), coordinates(coordinates), queries(queries),
        initial(initial), capacity(capacity),
        published(new std::atomic<bool>[capacity]) {
    for (unsigned i = 0; i < capacity; ++i)
      published[i].store(false, std::memory_order_relaxed);
  }

  SearchResult query_pair(unsigned lane, unsigned row, unsigned depth,
                          const std::array<unsigned, 2> &k, double alpha,
                          CoordinateFile::Scratch &scratch) {
    auto tick = Clock::now();
    SearchResult result;
    std::vector<uint32_t> ids;
    ids.reserve(k[0] + k[1]);
    for (unsigned s = 0; s < 2; ++s)
      result.native_pages[s] = pair.search(s, lane, row, depth, k[s], ids);
    ids.erase(std::remove_if(ids.begin(), ids.end(), [&](uint32_t id) {
                return id >= initial &&
                       !published[id - initial].load(std::memory_order_acquire);
              }),
              ids.end());
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    coordinates.rerank(ids, queries[0].row(row), queries[1].row(row), alpha,
                       scratch, result);
    result.latency_us = seconds(tick) * 1e6;
    return result;
  }

  void update(unsigned count, unsigned readers, const std::string &out) {
    check(count && count <= capacity - done &&
              (readers == 0 || readers == query_threads),
          "invalid update workload");
    const auto before = pair.memory();
    std::atomic<unsigned> next{ 0 }, seq{ 0 };
    std::atomic<bool> stop{ false }, failed{ false };
    std::array<unsigned, query_threads> queries_done{};
    std::array<std::vector<int>, 2> ids{ std::vector<int>(count),
                                         std::vector<int>(count) };
    std::vector<std::thread> updates, probes;
    std::mutex errors;
    std::exception_ptr failure;
    auto fail = [&] {
      failed = true;
      std::lock_guard<std::mutex> lock(errors);
      if (!failure)
        failure = std::current_exception();
    };
    auto start = Clock::now();
    for (unsigned t = 0; t < readers; ++t)
      probes.emplace_back([&, t] {
        try {
          CoordinateFile::Scratch scratch;
          constexpr double weights[] = { .1, .3, .5, .7, .9 };
          while (!stop && !failed) {
            unsigned sequence = seq++;
            query_pair(writers + t, report[(sequence / 5) % report.size()], 96,
                       { 96, 96 }, weights[sequence % 5], scratch);
            ++queries_done[t];
          }
        }
        catch (...) {
          fail();
        }
      });
    for (unsigned t = 0; t < writers; ++t)
      updates.emplace_back([&, t] {
        try {
          for (;;) {
            unsigned ordinal = next++;
            if (ordinal >= count || failed)
              break;
            auto result = pair.add(done + ordinal, t);
            ids[0][ordinal] = result[0];
            ids[1][ordinal] = result[1];
            published[done + ordinal].store(true, std::memory_order_release);
          }
        }
        catch (...) {
          fail();
        }
      });
    for (auto &t : updates)
      t.join();
    auto api = seconds(start);
    stop = true;
    for (auto &t : probes)
      t.join();
    pair.control(drain);
    auto complete = seconds(start);
    if (failure)
      std::rethrow_exception(failure);
    for (auto &index : ids) {
      std::sort(index.begin(), index.end());
      for (unsigned i = 0; i < count; ++i)
        check(index[i] == int(initial + done + i),
              "native insertion ID coverage");
    }
    done += count;
    auto after = pair.memory();
    pair.control(flush);
    std::ofstream f(out);
    f << std::setprecision(17) << "{\"initial\":" << initial
      << ",\"count\":" << count << ",\"inserted_total\":" << done
      << ",\"writers\":" << writers << ",\"readers\":" << readers
      << ",\"api_wall_seconds\":" << api
      << ",\"api_insert_qps\":" << count / api
      << ",\"drained_wall_seconds\":" << complete
      << ",\"insert_qps\":" << count / complete << ",\"query_count\":"
      << std::accumulate(queries_done.begin(), queries_done.end(), 0u)
      << ",\"loaded_memory\":";
    before.write(f);
    f << ",\"settled_memory\":";
    after.write(f);
    f << "}\n";
    f.close();
    check(bool(f), "update output");
  }

  void measure(double alpha, unsigned depth, const std::array<unsigned, 2> &k,
               const std::vector<unsigned> &cohort,
               const Matrix<uint32_t> &truth, const std::string &output) {
    check(!cohort.empty() && truth.n == queries[0].n && truth.d == 10 && k[0] &&
              k[1] && std::max(k[0], k[1]) <= depth && depth <= 1600,
          "invalid query sweep");
    std::vector<SearchResult> outcomes(cohort.size());
    std::array<std::exception_ptr, query_threads> errors{};
    std::vector<std::thread> team;
    std::atomic<size_t> next{ 0 };
    auto wall = Clock::now();
    for (unsigned j = 0; j < query_threads; ++j)
      team.emplace_back([&, j] {
        try {
          CoordinateFile::Scratch scratch;
          for (size_t i = next++; i < cohort.size(); i = next++)
            outcomes[i] =
                query_pair(writers + j, cohort[i], depth, k, alpha, scratch);
        }
        catch (...) {
          errors[j] = std::current_exception();
        }
      });
    for (auto &thread : team)
      thread.join();
    auto elapsed = seconds(wall);
    for (auto &error : errors)
      if (error)
        std::rethrow_exception(error);
    std::array<uint64_t, 2> pages{};
    uint64_t coordinate_pages = 0, reranked = 0, hits = 0;
    std::vector<double> latency;
    std::ofstream f(output);
    f << std::setprecision(17) << "{\"alpha\":" << alpha << ",\"L\":" << depth
      << ",\"kx\":" << k[0] << ",\"ky\":" << k[1] << ",\"queries\":[";
    for (size_t i = 0; i < cohort.size(); ++i) {
      auto &r = outcomes[i];
      unsigned found = 0;
      for (auto id : r.ids)
        found += std::find(truth.row(cohort[i]), truth.row(cohort[i]) + 10,
                           id) != truth.row(cohort[i]) + 10;
      hits += found;
      for (unsigned s = 0; s < 2; ++s)
        pages[s] += r.native_pages[s];
      coordinate_pages += r.coordinate_pages;
      reranked += r.reranked;
      latency.push_back(r.latency_us);
      if (i)
        f << ',';
      f << "{\"id\":" << cohort[i] << ",\"hits\":" << found
        << ",\"latency_us\":" << r.latency_us
        << ",\"x_pages4\":" << r.native_pages[0]
        << ",\"y_pages4\":" << r.native_pages[1]
        << ",\"coordinate_pages8\":" << r.coordinate_pages << ",\"ids\":[";
      for (size_t j = 0; j < r.ids.size(); ++j)
        f << (j ? "," : "") << r.ids[j];
      f << "]}";
    }
    const double n = cohort.size();
    std::sort(latency.begin(), latency.end());
    f << "],\"recall\":" << hits / (10 * n) << ",\"latency_us\":"
      << std::accumulate(latency.begin(), latency.end(), 0.) / n
      << ",\"p95_us\":" << latency[(95 * latency.size() + 99) / 100 - 1]
      << ",\"qps\":" << n / elapsed << ",\"x_pages4\":" << pages[0] / n
      << ",\"y_pages4\":" << pages[1] / n
      << ",\"coordinate_pages8\":" << coordinate_pages / n
      << ",\"read_requests\":" << (pages[0] + pages[1] + coordinate_pages) / n
      << ",\"read_KiB\":"
      << (4 * (pages[0] + pages[1]) + 8 * coordinate_pages) / n
      << ",\"reranked\":" << reranked / n << "}\n";
    f.close();
    check(bool(f), "query output");
  }

  std::vector<unsigned> query_ids(const std::string &path) {
    std::ifstream f(path);
    std::vector<unsigned> ids;
    unsigned id;
    while (f >> id) {
      check(id < queries[0].n, "cohort ID");
      ids.push_back(id);
    }
    check(f.eof() && !ids.empty(), "cohort file");
    return ids;
  }

  std::vector<unsigned> report;

private:
  Pair &pair;
  CoordinateFile &coordinates;
  const std::array<Matrix<float>, 2> &queries;
  unsigned initial, capacity, done = 0;
  std::unique_ptr<std::atomic<bool> []> published;
};

int main(int argc, char **argv) {
  try {
    check(argc == 10, "image-prefix text-prefix fixture coords initial count "
                      "report-ids ready-memory readers");
    const std::string fixture = argv[3];
    const unsigned initial = std::stoul(argv[5]), count = std::stoul(argv[6]);
    const unsigned readers = std::stoul(argv[9]);
    check(readers == 0 || readers == query_threads, "reader count");
    const std::array<Matrix<float>, 2> points{
      Matrix<float>(fixture + "/insert-x.bin", count),
      Matrix<float>(fixture + "/insert-y.bin", count)
    };
    const std::array<Matrix<float>, 2> queries{
      Matrix<float>(fixture + "/query-x.bin"),
      Matrix<float>(fixture + "/query-y.bin")
    };
    check(count && queries[0].n == queries[1].n && queries[0].d == dim &&
              queries[1].d == dim && points[0].d == dim && points[1].d == dim,
          "dual-space fixture dimensions");
    Pair native_pair({ argv[1], argv[2] }, points, queries, initial,
                     writers + readers);
    CoordinateFile coordinates(argv[4], initial + count);
    Benchmark benchmark(native_pair, coordinates, queries, initial, count);
    benchmark.report = benchmark.query_ids(argv[7]);
    {
      std::ofstream out(argv[8]);
      native_pair.memory().write(out);
      out << '\n';
      out.close();
      check(bool(out), "ready memory output");
    }
    std::cout << "DUAL_ODIN_READY" << std::endl;
    for (std::string action; std::cin >> action;) {
      if (action == "quit") {
        native_pair.finish();
        std::cout << "DUAL_ODIN_DONE" << std::endl;
        return 0;
      }
      if (action == "update") {
        unsigned n, active_readers;
        std::string output;
        check(bool(std::cin >> n >> active_readers >> output) &&
                  active_readers == readers,
              "update command");
        benchmark.update(n, readers, output);
      } else {
        check(action == "measure" && readers == query_threads,
              "unknown command");
        double alpha;
        unsigned depth, kx, ky;
        std::string ids, truth, output;
        check(bool(std::cin >> alpha >> depth >> kx >> ky >> ids >> truth >>
                   output),
              "measure command");
        benchmark.measure(alpha, depth, { kx, ky }, benchmark.query_ids(ids),
                          Matrix<uint32_t>(truth), output);
      }
      std::cout << "DUAL_ODIN_RESULT" << std::endl;
    }
    throw std::runtime_error("lost driver control stream");
  }
  catch (const std::exception &error) {
    std::cerr << "DUAL_ODIN_FAIL " << error.what() << std::endl;
    return 1;
  }
}
