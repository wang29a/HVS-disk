#include <algorithm>
#include <array>
#include <fstream>
#include "measured_reader.h"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <omp.h>
#include <stdexcept>
#include <vector>
#include "aux_utils.h"
#include "linux_aligned_file_reader.h"
#include "ssd_index.h"
#include "utils.h"

// Measurement-only driver: links the unchanged d2a39e2 search library.
// Each timed parallel loop contains many cyclic replays of the original
// queries.
int main(int argc, char **argv) {
  try {
    if (argc < 15 || std::string(argv[1]) != "float")
      throw std::runtime_error(
          "Use original dual-space search arguments; float only");
    const std::string prefix = argv[2];
    const unsigned threads = std::stoul(argv[3]), beam = std::stoul(argv[4]);
    const unsigned k = std::stoul(argv[9]), mode = std::stoul(argv[11]);
    const unsigned mem_l = std::stoul(argv[12]),
                   strategy = std::stoul(argv[13]);
    if (std::string(argv[10]) != "l2" || (mode != 0 && mode != 6) ||
        mem_l != 0 || !threads || !k)
      throw std::runtime_error(
          "Expected L2, mode 0/6, mem_L=0, positive thread/K");
    double seconds = std::getenv("HVS_POINT_SECONDS")
                         ? std::stod(std::getenv("HVS_POINT_SECONDS"))
                         : 5.0;
    const bool one_pass = std::getenv("HVS_ONE_PASS") != nullptr;
    if (!std::isfinite(seconds) || seconds <= 0)
      throw std::runtime_error("Invalid point duration");
    float *qe = nullptr, *ql = nullptr, *alpha = nullptr, *gt_dist = nullptr;
    unsigned *gt = nullptr;
    uint32_t *tags = nullptr;
    size_t q, q2, aq, de, dl, ad, gq, gk;
    pipeann::load_bin<float>(argv[5], qe, q, de);
    pipeann::load_bin<float>(argv[6], ql, q2, dl);
    pipeann::load_bin<float>(argv[7], alpha, aq, ad);
    pipeann::load_truthset(argv[8], gt, gt_dist, gq, gk, &tags);
    if (!q || q != q2 || q != aq || q != gq || ad != 1 || gk < k)
      throw std::runtime_error("Query/alpha/GT mismatch");
    for (size_t i = 0; i < q; ++i)
      if (!std::isfinite(alpha[i]) || alpha[i] < 0 || alpha[i] > 1)
        throw std::runtime_error("Invalid query alpha");
    omp_set_num_threads(threads);
    const bool observe = std::getenv("PHASE_DISABLE_OBSERVER") == nullptr;
    auto measured = std::make_shared<io_measure::Reader>(observe);
    std::shared_ptr<AlignedFileReader> reader = measured;
    static_cast<LinuxAlignedFileReader *>(reader.get())->strategy = strategy;
    std::unique_ptr<pipeann::SSDIndex<float> > index(
        new pipeann::SSDIndex<float>(pipeann::Metric::L2, reader,
                                     SearchMode(mode), true));
    if (index->load(prefix.c_str(), threads, mode != 0, mode != 0) != 0)
      throw std::runtime_error("Index load failed");

    std::vector<pipeann::QueryStats> stats;
    std::vector<uint32_t> results;
    std::vector<float> distances;
    std::vector<io_measure::Observation> observations;
    std::vector<double> service;
    constexpr size_t NFIELDS = 25;
    const char *names[NFIELDS] = {
      "legacy_topology_wait_us",       "legacy_coordinate_read_us",
      "legacy_exact_distance_us",      "external_service_us",
      "reader_total_us",               "reader_coupled_us",
      "reader_coupled_backend_us",     "reader_topology_us",
      "reader_topology_submit_us",     "reader_topology_wait_us",
      "reader_topology_blocking_us",   "reader_coordinate_us",
      "reader_coordinate_submit_us",   "reader_coordinate_wait_us",
      "reader_coordinate_blocking_us", "coupled_requested",
      "coupled_backend_requests",      "topology_backend_requests",
      "coordinate_backend_requests",   "coupled_backend_bytes",
      "topology_backend_bytes",        "coordinate_backend_bytes",
      "non_reader_service_us",         "normalized_candidate_evaluations",
      "legacy_io_phase_residual_us"
    };
    auto run = [&](size_t samples, unsigned search_l) {
      stats.assign(samples, pipeann::QueryStats{});
      results.assign(samples * k, std::numeric_limits<uint32_t>::max());
      distances.resize(samples * k);
      observations.resize(samples);
      service.resize(samples);
      const auto start = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
      for (int64_t i = 0; i < static_cast<int64_t>(samples); ++i) {
        const size_t qi = i % q;
        auto call = io_measure::Clock::now();
        measured->begin(mode);
        if (mode == 0)
          index->beam_search(qe + qi * de, ql + qi * dl, alpha[qi], k, mem_l,
                             search_l, results.data() + i * k,
                             distances.data() + i * k, beam, stats.data() + i,
                             nullptr, false);
        else
          index->pq_page_rerank_search(qe + qi * de, ql + qi * dl, alpha[qi], k,
                                       mem_l, search_l, results.data() + i * k,
                                       distances.data() + i * k, beam,
                                       stats.data() + i);
        observations[i] = measured->end();
        service[i] = io_measure::elapsed(call);
        const auto &o = observations[i];
        if (o.total_us() > service[i] + .01)
          throw std::runtime_error("reader time exceeds query service time");
        if (observe) {
          if (mode == 0) {
            if (o.coupled_requested != uint64_t(stats[i].n_ios) ||
                o.coupled_backend_requests > o.coupled_requested ||
                o.topology_backend_requests || o.coordinate_backend_requests)
              throw std::runtime_error("coupled counter boundary mismatch");
          } else if (o.topology_backend_requests != uint64_t(stats[i].n_ios) ||
                     o.coordinate_backend_requests !=
                         uint64_t(stats[i].rerank_ios) ||
                     o.coupled_requested)
            throw std::runtime_error("separated counter boundary mismatch");
        }
      }
      return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                           start).count();
    };

    for (int arg = 14; arg < argc; ++arg) {
      const unsigned search_l = std::stoul(argv[arg]);
      if (search_l < k)
        throw std::runtime_error("L below K");
      const size_t pilot =
          std::min(q, std::max(size_t(256), size_t(threads) * 8));
      double elapsed = run(pilot, search_l); // Excluded warm-up/calibration.
      size_t samples =
          one_pass ? q : std::max(q, size_t(std::ceil(seconds * 1.15 / elapsed *
                                                      pilot / q)) *
                                         q);
      if (samples > 2000000)
        throw std::runtime_error("Calibration sample cap exceeded");
      auto unix_start = std::chrono::system_clock::now();
      unsigned attempts = 0;
      do {
        unix_start = std::chrono::system_clock::now();
        elapsed = run(samples, search_l);
        ++attempts;
        if (one_pass || elapsed >= seconds)
          break;
        samples = std::max(
            samples + q,
            size_t(std::ceil(seconds * 1.15 / elapsed * samples / q)) * q);
        if (samples > 2000000 || attempts >= 5)
          throw std::runtime_error("Cannot achieve requested steady interval");
      } while (true);

      if (const char *dump = std::getenv("PHASE_DUMP_PREFIX")) {
        if (!one_pass)
          throw std::runtime_error("dump requires one pass");
        std::string stem = std::string(dump) + ".L" + std::to_string(search_l);
        std::ofstream oi(stem + ".ids", std::ios::binary),
            od(stem + ".dists", std::ios::binary);
        oi.write((char *)results.data(), results.size() * sizeof(uint32_t));
        od.write((char *)distances.data(), distances.size() * sizeof(float));
        if (!oi || !od)
          throw std::runtime_error("dump write");
      }
      double sums[8] = {};
      std::array<double, NFIELDS> extra{};
      double bucket_sum[5][6] = {};
      std::vector<double> latency(samples);
      for (size_t i = 0; i < samples; ++i) {
        const size_t qi = i % q;
        const auto &s = stats[i];
        const double vals[] = {
          s.total_us,   s.n_hops,       s.n_cmps, s.n_ios,
          s.rerank_ios, s.n_cache_hits, s.cpu_us, s.io_us
        };
        for (int j = 0; j < 8; ++j)
          sums[j] += vals[j];
        const auto &o = observations[i];
        const double phase[] = { s.io_us1,
                                 s.io_us2,
                                 s.accuracy_compute_us,
                                 service[i],
                                 o.total_us(),
                                 o.coupled_reader_us,
                                 o.coupled_backend_us,
                                 o.topology_us(),
                                 o.topology_submit_us,
                                 o.topology_wait_us,
                                 o.topology_blocking_us,
                                 o.coordinate_us(),
                                 o.coordinate_submit_us,
                                 o.coordinate_wait_us,
                                 o.coordinate_blocking_us,
                                 double(o.coupled_requested),
                                 double(o.coupled_backend_requests),
                                 double(o.topology_backend_requests),
                                 double(o.coordinate_backend_requests),
                                 double(o.coupled_backend_bytes),
                                 double(o.topology_backend_bytes),
                                 double(o.coordinate_backend_bytes),
                                 service[i] - o.total_us(),
                                 s.n_cmps / (mode == 0 ? 2.0 : 1.0),
                                 s.io_us - s.io_us1 - s.io_us2 };
        static_assert(sizeof(phase) / sizeof(double) == NFIELDS,
                      "phase fields");
        for (size_t j = 0; j < NFIELDS; j++)
          extra[j] += phase[j];
        latency[i] = s.total_us;
        unsigned hits = 0;
        for (unsigned j = 0; j < k; ++j) {
          const uint32_t id = results[i * k + j];
          if (id == std::numeric_limits<uint32_t>::max())
            throw std::runtime_error("Incomplete results");
          for (unsigned other = 0; other < j; ++other)
            if (id == results[i * k + other])
              throw std::runtime_error("Duplicate result ID");
          hits +=
              std::find(gt + qi * gk, gt + qi * gk + k, id) != gt + qi * gk + k;
        }
        const int bucket = std::min(4, int(alpha[qi] * 5));
        bucket_sum[bucket][0] += 1;
        bucket_sum[bucket][1] += hits;
        bucket_sum[bucket][2] += s.total_us;
        bucket_sum[bucket][3] += s.n_ios;
        bucket_sum[bucket][4] += s.rerank_ios;
        bucket_sum[bucket][5] += s.n_cmps;
      }
      std::sort(latency.begin(), latency.end());
      auto percentile = [&](double p) {
        return latency[std::min(samples - 1, size_t(p * samples))];
      };
      double hits = 0;
      for (int b = 0; b < 5; ++b)
        hits += bucket_sum[b][1];
      std::cout << std::setprecision(12) << "HVS_POINT {\"L\":" << search_l
                << ",\"threads\":" << threads << ",\"mode\":" << mode
                << ",\"strategy\":" << strategy << ",\"unique_queries\":" << q
                << ",\"samples\":" << samples << ",\"seconds\":" << elapsed
                << ",\"qps\":" << samples / elapsed
                << ",\"recall\":" << hits * 100 / (samples * k)
                << ",\"latency_us\":" << sums[0] / samples
                << ",\"p95_us\":" << percentile(.95)
                << ",\"p99_us\":" << percentile(.99)
                << ",\"p999_us\":" << percentile(.999)
                << ",\"hops\":" << sums[1] / samples
                << ",\"comparisons\":" << sums[2] / samples
                << ",\"topology_ios\":" << sums[3] / samples
                << ",\"rerank_ios\":" << sums[4] / samples
                << ",\"cache_hits\":" << sums[5] / samples
                << ",\"cpu_us\":" << sums[6] / samples
                << ",\"io_us\":" << sums[7] / samples
                << ",\"attempts\":" << attempts << ",\"start_unix\":"
                << std::chrono::duration<double>(unix_start.time_since_epoch())
                       .count();
      for (size_t j = 0; j < NFIELDS; j++)
        std::cout << ",\"" << names[j] << "\":" << extra[j] / samples;
      std::cout << ",\"observer_enabled\":" << (observe ? "true" : "false")
                << ",\"alpha_buckets\":[";
      for (int b = 0; b < 5; ++b) {
        if (b)
          std::cout << ',';
        const double count = bucket_sum[b][0];
        std::cout << "{\"bucket\":" << b << ",\"samples\":" << count;
        if (count)
          std::cout << ",\"recall\":" << bucket_sum[b][1] * 100 / (count * k)
                    << ",\"latency_us\":" << bucket_sum[b][2] / count
                    << ",\"topology_ios\":" << bucket_sum[b][3] / count
                    << ",\"rerank_ios\":" << bucket_sum[b][4] / count
                    << ",\"comparisons\":" << bucket_sum[b][5] / count;
        std::cout << '}';
      }
      std::cout << "]}" << std::endl;
    }
    std::cout << "HVS_IO_PHASE_COMPLETE" << std::endl;
    std::exit(0); // Match the original CLI's process-lifetime cleanup policy.
  }
  catch (const std::exception &e) {
    std::cerr << "IO_PHASE_ERROR: " << e.what() << std::endl;
    return 2;
  }
}
