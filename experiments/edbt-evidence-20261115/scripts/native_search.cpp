#include "native_counts.h"
#include "measured_reader.h"
#include "ssd_index.h"
#include "utils.h"
#include <omp.h>
#include <sys/resource.h>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <chrono>
#include <cmath>
#include <array>
using Clock = std::chrono::steady_clock;
static double us(Clock::time_point p) {
  return std::chrono::duration<double, std::micro>(Clock::now() - p).count();
}
struct Result {
  std::array<unsigned, 10> ids{};
  std::array<float, 10> distances{};
  pipeann::QueryStats stats;
  native_measure::Counts counts;
  io_measure::Observation io;
  double latency = 0;
  unsigned hits = 0;
};
int main(int argc, char **argv) {
  try {
    if (argc < 12)
      throw std::runtime_error(
          "prefix qx qy alpha gt ids output threads seconds L...");
    unsigned threads = std::stoul(argv[9]);
    double duration = std::stod(argv[10]);
    if (!threads || threads > 32 || duration < 0 || duration > 30)
      throw std::runtime_error("run limits");
    omp_set_num_threads(threads);
    float *qx = nullptr, *qy = nullptr, *alpha = nullptr;
    unsigned *gt = nullptr;
    size_t q, d, q2, d2, nqa, da, nqg, dg;
    pipeann::load_bin<float>(argv[2], qx, q, d);
    pipeann::load_bin<float>(argv[3], qy, q2, d2);
    pipeann::load_bin<float>(argv[4], alpha, nqa, da);
    pipeann::load_bin<unsigned>(argv[5], gt, nqg, dg);
    if (q != q2 || q != nqa || q != nqg || da != 1 || dg != 10)
      throw std::runtime_error("input dimensions");
    std::vector<unsigned> selected;
    std::ifstream idf(argv[6]);
    unsigned id;
    while (idf >> id) {
      if (id >= q)
        throw std::runtime_error("query id");
      selected.push_back(id);
    }
    if (selected.empty())
      throw std::runtime_error("empty cohort");
    auto measured = std::make_shared<io_measure::Reader>();
    std::shared_ptr<AlignedFileReader> reader = measured;
    measured->strategy = 0;
    // Match native CLI process-lifetime ownership; detached update workers
    // make stack destruction inappropriate for this read-only benchmark.
    auto &index = *new pipeann::SSDIndex<float>(pipeann::Metric::L2, reader,
                                                BEAM_SEARCH, true);
    if (index.load(argv[1], threads, false, false))
      throw std::runtime_error("native load");
    std::ofstream out(argv[7]);
    out << std::setprecision(17) << "[";
    // argv[8] is a reserved trace/audit mode; emitted ids and distances are
    // always available.
    for (int k = 11; k < argc; ++k) {
      unsigned L = std::stoul(argv[k]);
      if (L < 10 || L > 20000)
        throw std::runtime_error("L range");
      std::vector<Result> answers(selected.size());
      std::vector<double> times;
      std::string failure;
      auto batch = [&](bool timed) {
#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t j = 0; j < int64_t(selected.size()); ++j)
          try {
            unsigned qi = selected[j];
            Result r;
            measured->begin(0);
            auto tick = Clock::now();
            index.beam_search(qx + qi * d, qy + qi * d2, alpha[qi], 10, 0, L,
                              r.ids.data(), r.distances.data(), 4, &r.stats,
                              nullptr, false);
            r.latency = us(tick);
            r.counts = native_measure::counts;
            r.io = measured->end();
            for (unsigned nb : r.ids)
              for (unsigned z = 0; z < 10; ++z)
                if (nb == gt[qi * 10 + z]) {
                  ++r.hits;
                  break;
                }
            if (r.io.coupled_requested != uint64_t(r.stats.n_ios))
              throw std::runtime_error("reader/search request mismatch");
            if (r.counts.pq_candidates &&
                (r.counts.pq_spaces != 2 * r.counts.pq_candidates ||
                 r.counts.exact_spaces != 2 * r.counts.expanded ||
                 r.counts.requested != uint64_t(r.stats.n_ios) ||
                 r.counts.requested_bytes != uint64_t(r.stats.read_size) ||
                 r.counts.pq_spaces != uint64_t(r.stats.n_cmps)))
              throw std::runtime_error("native explicit counter mismatch");
            if (r.io.coupled_backend_bytes !=
                8192 * r.io.coupled_backend_requests)
              throw std::runtime_error("backend page bytes");
            answers[j] = r;
          }
        catch (const std::exception &e) {
#pragma omp critical
          failure = e.what();
        }
        if (!failure.empty())
          throw std::runtime_error(failure);
        if (timed)
          for (const auto &r : answers)
            times.push_back(r.latency);
      };
      if (duration)
        batch(false);
      auto start = Clock::now();
      double epoch = std::chrono::duration<double>(
          std::chrono::system_clock::now().time_since_epoch()).count();
      do {
        batch(true);
      } while (us(start) / 1e6 < duration);
      double seconds = us(start) / 1e6;
      std::sort(times.begin(), times.end());
      double hits = 0, pqc = 0, pqs = 0, ex = 0, expanded = 0, req = 0,
             back = 0, bytes = 0, wait = 0;
      for (const auto &r : answers) {
        hits += r.hits;
        pqc += r.counts.pq_candidates;
        pqs += r.counts.pq_spaces;
        ex += r.counts.exact_spaces;
        expanded += r.counts.expanded;
        req += r.io.coupled_requested;
        back += r.io.coupled_backend_requests;
        bytes += r.io.coupled_backend_bytes;
        wait += r.io.coupled_reader_us;
      }
      double n = answers.size();
      struct rusage usage {};
      getrusage(RUSAGE_SELF, &usage);
      if (k != 11)
        out << ",";
      out << "{\"L\":" << L << ",\"threads\":" << threads
          << ",\"start_unix\":" << epoch << ",\"seconds\":" << seconds
          << ",\"recall\":" << hits / n / 10 << ",\"latency_us\":"
          << std::accumulate(times.begin(), times.end(), 0.) / times.size()
          << ",\"qps\":" << times.size() / seconds << ",\"p95_us\":"
          << times[std::min(times.size() - 1,
                            size_t(std::ceil(.95 * times.size()) - 1))]
          << ",\"pq_candidates\":" << pqc / n
          << ",\"pq_space_evals\":" << pqs / n
          << ",\"exact_space_evals\":" << ex / n
          << ",\"expanded\":" << expanded / n
          << ",\"coupled_requested\":" << req / n
          << ",\"coupled_backend_pages\":" << back / n
          << ",\"coupled_backend_bytes\":" << bytes / n
          << ",\"coupled_reader_us\":" << wait / n
          << ",\"peak_rss_kib\":" << usage.ru_maxrss << ",\"queries\":[";
      for (size_t j = 0; j < answers.size(); ++j) {
        if (j)
          out << ",";
        auto &r = answers[j];
        out << "{\"query_id\":" << selected[j] << ",\"ids\":[";
        for (unsigned z = 0; z < 10; ++z) {
          if (z)
            out << ",";
          out << r.ids[z];
        }
        out << "],\"distances\":[";
        for (unsigned z = 0; z < 10; ++z) {
          if (z)
            out << ",";
          out << r.distances[z];
        }
        out << "]}";
      }
      out << "]}";
      out.flush();
    }
    out << "]\n";
    out.flush();
    out.close();
    std::exit(0);
  }
  catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
