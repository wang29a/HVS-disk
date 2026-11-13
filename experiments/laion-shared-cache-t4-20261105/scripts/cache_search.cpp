#include "shared_topo_reader.h"
#include "aux_utils.h"
#include "ssd_index.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <omp.h>

int main(int argc, char **argv) {
  try {
    if (argc < 15 || std::string(argv[1]) != "float")
      throw std::runtime_error("native dual-space float arguments required");
    unsigned threads = std::stoul(argv[3]), beam = std::stoul(argv[4]);
    unsigned k = std::stoul(argv[9]), mode = std::stoul(argv[11]);
    unsigned mem_l = std::stoul(argv[12]), strategy = std::stoul(argv[13]);
    if (threads != 4 || mode != 6 || mem_l || strategy != 17 || k != 10 ||
        beam != 4)
      throw std::runtime_error(
          "this experiment fixes T4/K10/beam4/mode6/mem0/strategy17");
    uint64_t cache_bytes =
        std::getenv("HVS_SHARED_CACHE_BYTES")
            ? std::stoull(std::getenv("HVS_SHARED_CACHE_BYTES"))
            : 0;
    double seconds = std::getenv("HVS_POINT_SECONDS")
                         ? std::stod(std::getenv("HVS_POINT_SECONDS"))
                         : 10;
    bool one_pass = std::getenv("HVS_ONE_PASS");
    if (!std::isfinite(seconds) || seconds <= 0)
      throw std::runtime_error("invalid point duration");
    float *q1 = nullptr, *q2 = nullptr, *alpha = nullptr, *gt_dist = nullptr;
    unsigned *gt = nullptr;
    uint32_t *tags = nullptr;
    size_t nq, nq2, na, d1, d2, da, ngt, gk;
    pipeann::load_bin<float>(argv[5], q1, nq, d1);
    pipeann::load_bin<float>(argv[6], q2, nq2, d2);
    pipeann::load_bin<float>(argv[7], alpha, na, da);
    pipeann::load_truthset(argv[8], gt, gt_dist, ngt, gk, &tags);
    if (nq != 1000 || nq != nq2 || nq != na || nq != ngt || da != 1 || gk < k)
      throw std::runtime_error("query/GT mismatch");
    for (size_t i = 0; i < nq; ++i)
      if (alpha[i] != .5f)
        throw std::runtime_error("Laion alpha must equal 0.5");
    omp_set_num_threads(threads);
    auto reader = std::make_shared<SharedTopoReader>();
    reader->strategy = strategy;
    std::shared_ptr<AlignedFileReader> native_reader = reader;
    std::unique_ptr<pipeann::SSDIndex<float> > index(
        new pipeann::SSDIndex<float>(pipeann::Metric::L2, native_reader,
                                     SearchMode(mode), true));
    if (index->load(argv[2], threads, true, true))
      throw std::runtime_error("native index load failed");
    reader->configure(cache_bytes);
    std::vector<pipeann::QueryStats> stats;
    std::vector<SharedTopoReader::Observation> observations;
    std::vector<uint32_t> ids;
    std::vector<float> distances;
    std::vector<double> external;
    auto run = [&](size_t samples, unsigned L) {
      stats.assign(samples, pipeann::QueryStats{});
      observations.resize(samples);
      external.resize(samples);
      ids.assign(samples * k, std::numeric_limits<uint32_t>::max());
      distances.resize(samples * k);
      auto begin = SharedTopoReader::Clock::now();
#pragma omp parallel for schedule(dynamic, 1)
      for (int64_t i = 0; i < int64_t(samples); ++i) {
        size_t qi = i % nq;
        auto call = SharedTopoReader::Clock::now();
        reader->begin();
        index->pq_page_rerank_search(q1 + qi * d1, q2 + qi * d2, alpha[qi], k,
                                     0, L, ids.data() + i * k,
                                     distances.data() + i * k, beam, &stats[i]);
        observations[i] = reader->end();
        external[i] = std::chrono::duration<double, std::micro>(
            SharedTopoReader::Clock::now() - call).count();
        if (observations[i].requested != uint64_t(stats[i].n_ios) ||
            observations[i].coordinate_pages != uint64_t(stats[i].rerank_ios))
          throw std::runtime_error("native versus reader counters differ");
      }
      return std::chrono::duration<double>(SharedTopoReader::Clock::now() -
                                           begin).count();
    };
    for (int a = 14; a < argc; ++a) {
      unsigned L = std::stoul(argv[a]);
      if (L < k || L > 400)
        throw std::runtime_error("L outside short-test bounds");
      reader->clear_cache();
      double duration = run(nq, L); // Full 1,000-query warm-up, excluded.
      size_t samples =
          one_pass
              ? nq
              : std::max(nq, size_t(std::ceil(seconds * 1.1 / duration)) * nq);
      unsigned attempts = 0;
      double unix_start = 0;
      do {
        if (samples > 2000000 || ++attempts > 5)
          throw std::runtime_error("calibration limit");
        unix_start = std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        duration = run(samples, L);
        if (one_pass || duration >= seconds)
          break;
        samples = std::max(
            samples + nq,
            size_t(std::ceil(seconds * 1.1 / duration * samples / nq)) * nq);
      } while (true);
      if (const char *prefix = std::getenv("HVS_DUMP_PREFIX")) {
        if (!one_pass)
          throw std::runtime_error("dump requires one pass");
        std::string base = std::string(prefix) + ".L" + std::to_string(L);
        std::ofstream oi(base + ".ids", std::ios::binary),
            od(base + ".dists", std::ios::binary);
        oi.write((char *)ids.data(), ids.size() * sizeof(uint32_t));
        od.write((char *)distances.data(), distances.size() * sizeof(float));
        if (!oi || !od)
          throw std::runtime_error("dump failed");
      }
      double internal = 0, service = 0, cmps = 0, logical = 0, rerank = 0,
             local_hits = 0;
      double requested = 0, hits = 0, backend = 0, coords = 0, topo_us = 0,
             coord_us = 0;
      uint64_t correct = 0;
      for (size_t i = 0; i < samples; ++i) {
        auto &s = stats[i];
        auto &o = observations[i];
        size_t qi = i % nq;
        internal += s.total_us;
        service += external[i];
        cmps += s.n_cmps;
        logical += s.n_ios;
        rerank += s.rerank_ios;
        local_hits += s.n_cache_hits;
        requested += o.requested;
        hits += o.hits;
        backend += o.backend_pages;
        coords += o.coordinate_pages;
        topo_us += o.topology_reader_us;
        coord_us += o.coordinate_reader_us;
        for (unsigned j = 0; j < k; ++j) {
          auto id = ids[i * k + j];
          if (id == std::numeric_limits<uint32_t>::max())
            throw std::runtime_error("missing result");
          for (unsigned z = 0; z < j; ++z)
            if (ids[i * k + z] == id)
              throw std::runtime_error("duplicate result");
          correct +=
              std::find(gt + qi * gk, gt + qi * gk + k, id) != gt + qi * gk + k;
        }
      }
      std::sort(external.begin(), external.end());
      std::cout << std::setprecision(12) << "HVS_POINT {\"L\":" << L
                << ",\"threads\":4,\"cache_bytes\":" << cache_bytes
                << ",\"samples\":" << samples << ",\"unique_queries\":" << nq
                << ",\"seconds\":" << duration
                << ",\"start_unix\":" << unix_start
                << ",\"qps\":" << samples / duration
                << ",\"recall\":" << correct * 100. / (samples * k)
                << ",\"latency_us\":" << internal / samples
                << ",\"external_service_us\":" << service / samples
                << ",\"external_p95_us\":" << external[size_t(.95 * samples)]
                << ",\"external_p99_us\":" << external[size_t(.99 * samples)]
                << ",\"comparisons\":" << cmps / samples
                << ",\"topology_ios\":" << logical / samples
                << ",\"rerank_ios\":" << rerank / samples
                << ",\"query_local_hits\":" << local_hits / samples
                << ",\"shared_requested\":" << requested / samples
                << ",\"shared_hits\":" << hits / samples
                << ",\"topology_backend_pages\":" << backend / samples
                << ",\"coordinate_backend_pages\":" << coords / samples
                << ",\"backend_bytes\":" << (backend + coords) * 8192 / samples
                << ",\"reader_topology_us\":" << topo_us / samples
                << ",\"reader_coordinate_us\":" << coord_us / samples << "}"
                << std::endl;
    }
    std::cout << "HVS_CACHE_COMPLETE" << std::endl;
    std::exit(0);
  }
  catch (const std::exception &e) {
    std::cerr << "CACHE_SEARCH_ERROR: " << e.what() << std::endl;
    return 2;
  }
}
