#include "ssd_index.h"
#include "layout_store.h"
#include <cmath>
#include <iomanip>
#include <numeric>
#include <random>
#include <sys/resource.h>

using namespace layout_exp;
inline float metric_score(float a, float b, float alpha, bool single_x,
                          bool concat) {
  a = std::max(0.0f, a);
  b = std::max(0.0f, b);
  if (concat)
    return std::sqrt(a + b);
  return single_x ? std::sqrt(a)
                  : alpha * std::sqrt(a) + (1 - alpha) * std::sqrt(b);
}
struct Stats {
  double latency = 0, prep = 0, coarse = 0, rerank = 0, topo_wait = 0,
         coord_wait = 0, candidate_cpu = 0, exact_us = 0, sort_us = 0;
  U topo_submitted = 0, coord_submitted = 0, topo_hits = 0, coord_hits = 0;
  U topo_distinct = 0, coord_distinct = 0;
  U topo_shared_hits = 0, coord_shared_hits = 0;
  double topo_shared_us = 0, coord_shared_us = 0;
  U edges = 0, alpha_reject = 0, duplicate_reject = 0, pq = 0, expanded = 0,
    inserted = 0, reranked = 0;
  U pq_space_evals = 0, exact_space_evals = 0;
  U gt_discovered = 0, gt_candidates = 0, gt_rerank = 0, candidate_count = 0;
  U navigation_exact_evals = 0;
  double pq_lookup_us = 0;
  Counters io;
};
struct Answer {
  std::array<unsigned, 10> ids;
  std::array<float, 10> dists;
  Stats stats;
};
struct Scratch {
  float *a, *b = nullptr, *qa, *qb = nullptr;
  std::vector<float> table1, table2, de, dl;
  std::vector<uint8_t> codes1, codes2;
  Scratch(U d1, U d2, U chunks, U r)
      : table1(2 * chunks * 256), table2(d2 ? chunks * 256 : 0),
        de(std::max(U(2048), r)), dl(d2 ? std::max(U(2048), r) : 0),
        codes1(std::max(U(2048), r) * chunks * 2),
        codes2(d2 ? std::max(U(2048), r) * chunks : 0) {
    if (posix_memalign((void **)&a, 64, (d1 + d2 + 8) * 4) ||
        posix_memalign((void **)&qa, 64, (d1 + d2 + 8) * 4) ||
        (d2 && (posix_memalign((void **)&b, 64, (d2 + 8) * 4) ||
                posix_memalign((void **)&qb, 64, (d2 + 8) * 4))))
      throw std::runtime_error("scratch");
    memset(a, 0, (d1 + d2 + 8) * 4);
    memset(qa, 0, (d1 + d2 + 8) * 4);
    if (d2) {
      memset(b, 0, (d2 + 8) * 4);
      memset(qb, 0, (d2 + 8) * 4);
    }
  }
  ~Scratch() {
    free(a);
    free(b);
    free(qa);
    free(qb);
  }
};
struct Engine {
  Source &s;
  const bool exact_navigation = std::getenv("METRIC_NAV_EXACT");
  const bool concat_metric = std::getenv("METRIC_CONCAT");
  const unsigned rerank_limit = std::getenv("METRIC_RERANK")
                                    ? std::stoul(std::getenv("METRIC_RERANK"))
                                    : 6400;
  pipeann::FixedChunkPQTable<float> pq1, pq2;
  std::vector<uint8_t> code1, code2;
  U chunks = 0;
  const bool only_x;
  const bool bypass_alpha;
  U interval_slots;
  std::vector<int8_t> sidecar;
  const int coordinate_mask;
  std::unique_ptr<pipeann::Distance<float> > distance;
  Engine(Source &src, const std::string &prefix, bool first_only,
         int coord_fields, bool bypass)
      : s(src), only_x(first_only), bypass_alpha(bypass), interval_slots(s.M),
        coordinate_mask(coord_fields) {
    if (const char *path = std::getenv("LAYOUT_ALPHA_SIDECAR")) {
      const char *slots = std::getenv("LAYOUT_ALPHA_SLOTS");
      if (!slots || s.M || bypass_alpha)
        throw std::runtime_error("sidecar requires filtered compact topology");
      interval_slots = std::stoull(slots);
      if (!interval_slots || interval_slots > 64)
        throw std::runtime_error("sidecar interval width");
      sidecar.resize(s.n * s.R * 2 * interval_slots);
      std::ifstream input(path, std::ios::binary);
      if (!input.read(reinterpret_cast<char *>(sidecar.data()),
                      sidecar.size()) ||
          input.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("sidecar size");
    }
    size_t n, c;
    const std::string x_prefix = prefix + (s.d2 ? "emb" : "");
    pipeann::load_bin<uint8_t>(x_prefix + "_pq_compressed.bin", code1, n, c);
    if (n != s.n || c != 64)
      throw std::runtime_error("X PQ identity");
    chunks = c;
    pq1.load_pq_centroid_bin((x_prefix + "_pq_pivots.bin").c_str(), chunks);
    if (pq1.ndims != s.d1)
      throw std::runtime_error("X PQ dimensions");
    if (s.d2) {
      size_t n2, c2;
      pipeann::load_bin<uint8_t>(prefix + "loc_pq_compressed.bin", code2, n2,
                                 c2);
      pq2.load_pq_centroid_bin((prefix + "loc_pq_pivots.bin").c_str(), chunks);
      if (n2 != n || c2 != c || pq2.ndims != s.d2)
        throw std::runtime_error("Y PQ identity");
    }
    if (concat_metric) {
      if (!s.d2)
        throw std::runtime_error("concat requires two physical source fields");
      std::vector<uint8_t> joined(s.n * chunks * 2);
      for (U i = 0; i < s.n; ++i) {
        memcpy(joined.data() + i * chunks * 2, code1.data() + i * chunks,
               chunks);
        memcpy(joined.data() + i * chunks * 2 + chunks,
               code2.data() + i * chunks, chunks);
      }
      code1.swap(joined);
      std::vector<uint8_t>().swap(code2);
    }
    distance.reset(pipeann::get_distance_function<float>(pipeann::Metric::L2));
  }
  float fuse(float a, float b, float alpha) const {
    return metric_score(a, b, alpha, only_x, concat_metric);
  }
  void lookup(const std::vector<unsigned> &ids, Scratch &w, Stats &stats,
              Cache &cache) {
    if (exact_navigation) {
      if (cache.store.resident.empty())
        throw std::runtime_error("exact navigation requires DRAM backing");
      for (U j = 0; j < ids.size(); ++j) {
        U k1 = cache.store.key(ids[j], 2);
        const auto *x = cache.store.resident[k1 % 8].data() + (k1 / 8) * PAGE +
                        cache.store.offset(ids[j], 2);
        if (concat_metric) {
          memcpy(w.a, x, (s.d1 + s.d2) * 4);
          w.de[j] = distance->compare(w.qa, w.a, s.d1 + s.d2);
          w.dl[j] = 0;
          continue;
        }
        memcpy(w.a, x, s.d1 * 4);
        w.de[j] = distance->compare(w.qa, w.a, s.d1);
        if (!only_x) {
          U k2 = cache.store.key(ids[j], 4);
          const auto *y = cache.store.resident[k2 % 8].data() +
                          (k2 / 8) * PAGE + cache.store.offset(ids[j], 4);
          memcpy(w.b, y, s.d2 * 4);
          w.dl[j] = distance->compare(w.qb, w.b, s.d2);
        }
      }
      stats.navigation_exact_evals +=
          ids.size() * ((only_x || concat_metric) ? 1 : 2);
      return;
    }
    auto start = phase_now();
    U navigation_chunks = concat_metric ? chunks * 2 : chunks;
    aggregate_coords(ids.data(), ids.size(), code1.data(), navigation_chunks,
                     w.codes1.data());
    if (!only_x && !concat_metric)
      aggregate_coords(ids.data(), ids.size(), code2.data(), chunks,
                       w.codes2.data());
    pq_dist_lookup(w.codes1.data(), ids.size(), navigation_chunks,
                   w.table1.data(), w.de.data());
    if (!only_x && !concat_metric)
      pq_dist_lookup(w.codes2.data(), ids.size(), chunks, w.table2.data(),
                     w.dl.data());
    if (concat_metric)
      std::fill(w.dl.begin(), w.dl.begin() + ids.size(), 0.0f);
    stats.pq_lookup_us += phase_us(start);
    stats.pq_space_evals += ids.size() * ((only_x || concat_metric) ? 1 : 2);
  }
  Answer search(const float *q1, const float *q2, float alpha, unsigned L,
                Cache &cache, Scratch &w, Trace *trace = nullptr,
                const unsigned *truth = nullptr) {
    cache.reset();
    Answer result;
    auto begin = phase_now(), stage = begin;
    memcpy(w.qa, q1, s.d1 * 4);
    if (!exact_navigation)
      pq1.populate_chunk_distances(w.qa, w.table1.data());
    if (!only_x) {
      memcpy(w.qb, q2, s.d2 * 4);
      if (!exact_navigation)
        pq2.populate_chunk_distances(w.qb, w.table2.data());
    }
    if (concat_metric) {
      memcpy(w.qa + s.d1, q2, s.d2 * 4);
      if (!exact_navigation)
        memcpy(w.table1.data() + chunks * 256, w.table2.data(),
               chunks * 256 * sizeof(float));
    }
    result.stats.prep = phase_us(stage);
    std::vector<pipeann::Neighbor> ret(
        std::max(U(L) * 10, s.entries.size() + 1));
    tsl::robin_set<U> visited;
    unsigned cur = 0;
    lookup(s.entries, w, result.stats, cache);
    for (U i = 0; i < s.entries.size(); i++) {
      ret[cur].id = s.entries[i];
      ret[cur].distance = fuse(w.de[i], only_x ? 0 : w.dl[i], alpha);
      ret[cur++].flag = true;
      visited.insert(s.entries[i]);
    }
    std::sort(ret.begin(), ret.begin() + cur);
    unsigned k = 0;
    stage = phase_now();
    while (k < cur) {
      unsigned nk = cur, marker = k, seen = 0;
      std::vector<unsigned> frontier;
      while (marker < cur && frontier.size() < 4 && seen < 4) {
        if (ret[marker].flag) {
          seen++;
          frontier.push_back(ret[marker].id);
          ret[marker].flag = false;
        }
        marker++;
      }
      if (trace)
        trace->batches.push_back(frontier);
      cache.fetch(frontier, 1);
      result.stats.expanded += frontier.size();
      auto candidate_begin = phase_now();
      for (auto id : frontier) {
        const auto *p = cache.field(id, 1);
        unsigned nn = read32(p);
        if (nn > s.R)
          throw std::runtime_error("degree");
        std::vector<unsigned> nbrs;
        for (unsigned m = 0; m < nn; m++) {
          auto edge = p + 4 + m * (4 + 2 * s.M);
          unsigned neighbor = read32(edge);
          bool active = bypass_alpha || interval_slots == 0;
          result.stats.edges++;
          const int8_t *intervals =
              sidecar.empty()
                  ? reinterpret_cast<const int8_t *>(edge + 4)
                  : sidecar.data() + (id * s.R + m) * 2 * interval_slots;
          for (U z = 0; !bypass_alpha && z < interval_slots; z++) {
            int lo = intervals[2 * z], hi = intervals[2 * z + 1];
            if (lo < 0 || lo > 100 || hi < 0 || hi > 100)
              break;
            if (lo == 0 && hi == 0)
              break;
            float a100 = alpha * 100.f;
            if (a100 >= lo && a100 <= hi) {
              active = true;
              break;
            }
            if (a100 < lo)
              break;
          }
          if (!active) {
            result.stats.alpha_reject++;
            continue;
          }
          if (visited.find(neighbor) != visited.end()) {
            result.stats.duplicate_reject++;
            continue;
          }
          nbrs.push_back(neighbor);
          visited.insert(neighbor);
        }
        if (!nbrs.empty()) {
          lookup(nbrs, w, result.stats, cache);
          if (!exact_navigation)
            result.stats.pq += nbrs.size();
          for (U j = 0; j < nbrs.size(); j++) {
            float d = fuse(w.de[j], only_x ? 0 : w.dl[j], alpha);
            if (cur < L || d < ret[cur - 1].distance) {
              auto pos = pipeann::InsertIntoPool(
                  ret.data(), cur, pipeann::Neighbor(nbrs[j], d, true));
              result.stats.inserted++;
              if (cur < L)
                cur++;
              if (pos < nk)
                nk = pos;
            }
          }
        }
      }
      if (nk <= k)
        k = nk;
      else
        k++;
      result.stats.candidate_cpu += phase_us(candidate_begin);
    }
    result.stats.coarse = phase_us(stage);
    result.stats.topo_wait = cache.count.wait_us;
    stage = phase_now();
    result.stats.candidate_count = cur;
    if (truth) {
      for (U j = 0; j < 10; ++j) {
        result.stats.gt_discovered += visited.find(truth[j]) != visited.end();
        for (unsigned i = 0; i < cur; ++i)
          result.stats.gt_candidates += ret[i].id == truth[j];
      }
    }
    if (exact_navigation) {
      if (cur < 10)
        throw std::runtime_error("exact navigation insufficient candidates");
      for (int i = 0; i < 10; ++i) {
        result.ids[i] = ret[i].id;
        result.dists[i] = ret[i].distance;
      }
      if (trace)
        trace->coords.clear();
      result.stats.topo_distinct = cache.ever.size();
      result.stats.topo_submitted = cache.count.submitted;
      result.stats.topo_hits = cache.count.hits;
      result.stats.io = cache.count;
      result.stats.latency = phase_us(begin);
      return result;
    }
    std::vector<unsigned> ids;
    for (unsigned i = 0; i < std::min(cur, std::min(L, rerank_limit)); i++)
      ids.push_back(ret[i].id);
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    if (trace)
      trace->coords = ids;
    result.stats.topo_distinct = cache.ever.size();
    result.stats.topo_submitted = cache.count.submitted;
    result.stats.topo_hits = cache.count.hits;
    result.stats.topo_shared_hits = cache.count.shared_hits;
    result.stats.topo_shared_us = cache.count.shared_us;
    result.stats.reranked = ids.size();
    if (truth) {
      for (U j = 0; j < 10; ++j) {
        result.stats.gt_rerank +=
            std::binary_search(ids.begin(), ids.end(), truth[j]);
      }
    }
    std::vector<std::pair<unsigned, float> > exact;
    for (U first = 0; first < ids.size(); first += 256) {
      std::vector<unsigned> batch(ids.begin() + first,
                                  ids.begin() +
                                      std::min(U(ids.size()), first + 256));
      cache.fetch(batch, coordinate_mask);
      auto exact_begin = phase_now();
      for (auto id : batch) {
        if (concat_metric) {
          memcpy(w.a, cache.field(id, 2), (s.d1 + s.d2) * 4);
          exact.emplace_back(
              id, std::sqrt(std::max(
                      0.0f, distance->compare(w.qa, w.a, s.d1 + s.d2))));
          continue;
        }
        memcpy(w.a, cache.field(id, 2), s.d1 * 4);
        if (!only_x)
          memcpy(w.b, cache.field(id, 4), s.d2 * 4);
        const float a = distance->compare(w.qa, w.a, s.d1);
        float score = std::sqrt(a);
        if (!only_x) {
          const float b = distance->compare(w.qb, w.b, s.d2);
          score = fuse(a, b, alpha);
        }
        exact.emplace_back(id, score);
      }
      result.stats.exact_us += phase_us(exact_begin);
    }
    result.stats.exact_space_evals =
        ids.size() * ((only_x || concat_metric) ? 1 : 2);
    result.stats.coord_distinct =
        cache.ever.size() - result.stats.topo_distinct;
    result.stats.coord_wait = cache.count.wait_us - result.stats.topo_wait;
    result.stats.coord_submitted =
        cache.count.submitted - result.stats.topo_submitted;
    result.stats.coord_hits = cache.count.hits - result.stats.topo_hits;
    result.stats.coord_shared_hits =
        cache.count.shared_hits - result.stats.topo_shared_hits;
    result.stats.coord_shared_us =
        cache.count.shared_us - result.stats.topo_shared_us;
    auto sort_begin = phase_now();
    std::sort(exact.begin(), exact.end(),
              [](const auto &a, const auto &b) { return a.second < b.second; });
    if (exact.size() < 10)
      throw std::runtime_error("incomplete top10");
    for (int i = 0; i < 10; i++) {
      result.ids[i] = exact[i].first;
      result.dists[i] = exact[i].second;
    }
    result.stats.sort_us = phase_us(sort_begin);
    result.stats.rerank = phase_us(stage);
    result.stats.io = cache.count;
    result.stats.latency = phase_us(begin);
    return result;
  }
};
template <class T> std::vector<T> matrix(const std::string &path, U &n, U &d) {
  std::vector<T> v;
  size_t nn, dd;
  pipeann::load_bin<T>(path, v, nn, dd);
  n = nn;
  d = dd;
  return v;
}
inline void emit_point(std::ostream &o, const std::vector<Answer> &answers,
                       double seconds, unsigned L, unsigned threads, U q,
                       const std::vector<unsigned> *gt, double start_unix,
                       U topology_resident_bytes = 0) {
  std::vector<double> lat;
  double total = 0, prep = 0, coarse = 0, rerank = 0, tw = 0, cw = 0, pcpu = 0,
         ex = 0, sort = 0;
  U hits = 0, edges = 0, reject = 0, dup = 0, pq = 0, expanded = 0,
    inserted = 0, rc = 0, ts = 0, cs = 0, th = 0, ch = 0;
  U tsh = 0, csh = 0, pe = 0, ee = 0;
  U topology_memory_nodes = 0;
  double lookup_us = 0;
  U discovered = 0, candidates = 0, rerank_gt = 0, pool_count = 0,
    nav_exact = 0, topo_distinct = 0, coord_distinct = 0, evictions = 0;
  double tcu = 0, ccu = 0;
  Counters c;
  for (U i = 0; i < answers.size(); i++) {
    auto &a = answers[i];
    auto &s = a.stats;
    topology_memory_nodes += s.io.topology_memory_nodes;
    lat.push_back(s.latency);
    total += s.latency;
    prep += s.prep;
    coarse += s.coarse;
    rerank += s.rerank;
    edges += s.edges;
    reject += s.alpha_reject;
    dup += s.duplicate_reject;
    pq += s.pq;
    pe += s.pq_space_evals;
    topo_distinct += s.topo_distinct;
    coord_distinct += s.coord_distinct;
    evictions += s.io.evictions;
    discovered += s.gt_discovered;
    candidates += s.gt_candidates;
    rerank_gt += s.gt_rerank;
    pool_count += s.candidate_count;
    nav_exact += s.navigation_exact_evals;
    ee += s.exact_space_evals;
    lookup_us += s.pq_lookup_us;
    expanded += s.expanded;
    inserted += s.inserted;
    rc += s.reranked;
    ts += s.topo_submitted;
    cs += s.coord_submitted;
    th += s.topo_hits;
    ch += s.coord_hits;
    tsh += s.topo_shared_hits;
    csh += s.coord_shared_hits;
    tcu += s.topo_shared_us;
    ccu += s.coord_shared_us;
    c.requested += s.io.requested;
    c.hits += s.io.hits;
    c.submitted += s.io.submitted;
    c.bytes += s.io.bytes;
    c.useful += s.io.useful;
    c.wait_us += s.io.wait_us;
    tw += s.topo_wait;
    cw += s.coord_wait;
    pcpu += s.candidate_cpu;
    ex += s.exact_us;
    sort += s.sort_us;
    if (gt)
      for (auto id : a.ids)
        hits += std::find(gt->begin() + (i % q) * 10,
                          gt->begin() + (i % q + 1) * 10,
                          id) != gt->begin() + (i % q + 1) * 10;
  }
  std::sort(lat.begin(), lat.end());
  double n = answers.size();
  struct rusage r {};
  getrusage(RUSAGE_SELF, &r);
  o << std::setprecision(12) << "{\"L\":" << L << ",\"threads\":" << threads
    << ",\"navigation\":\""
    << (std::getenv("METRIC_NAV_EXACT") ? "exact" : "pq") << "\""
    << ",\"distance_target\":\""
    << (std::getenv("METRIC_CONCAT") ? "concat_l2" : "weighted_l2") << "\""
    << ",\"medium\":\"" << (std::getenv("LAYOUT_DRAM") ? "DRAM" : "SSD") << "\""
    << ",\"topology_medium\":\""
    << (topology_resident_bytes ? "DRAM"
                                : (std::getenv("LAYOUT_DRAM") ? "DRAM" : "SSD"))
    << "\""
    << ",\"coordinate_medium\":\""
    << (std::getenv("LAYOUT_DRAM") ? "DRAM" : "SSD") << "\""
    << ",\"topology_resident_bytes\":" << topology_resident_bytes
    << ",\"topology_memory_node_reads\":" << topology_memory_nodes / n
    << ",\"rerank_limit\":"
    << (std::getenv("METRIC_NAV_EXACT")
            ? 0
            : (std::getenv("METRIC_RERANK")
                   ? std::stoul(std::getenv("METRIC_RERANK"))
                   : L)) << ",\"queries\":" << q
    << ",\"samples\":" << answers.size() << ",\"seconds\":" << seconds
    << ",\"start_unix\":" << start_unix << ",\"qps\":" << n / seconds
    << ",\"recall\":" << (gt ? 100. * hits / n / 10 : -1)
    << ",\"latency_us\":" << total / n << ",\"p50_us\":" << lat[lat.size() / 2]
    << ",\"p95_us\":" << lat[std::min(lat.size() - 1, U(.95 * n))]
    << ",\"p99_us\":" << lat[std::min(lat.size() - 1, U(.99 * n))]
    << ",\"prep_us\":" << prep / n << ",\"coarse_us\":" << coarse / n
    << ",\"rerank_us\":" << rerank / n
    << ",\"topology_submit_wait_us\":" << tw / n
    << ",\"coordinate_submit_wait_us\":" << cw / n
    << ",\"candidate_cpu_us\":" << pcpu / n << ",\"exact_us\":" << ex / n
    << ",\"sort_us\":" << sort / n
    << ",\"unclassified_us\":" << (total - prep - coarse - rerank) / n
    << ",\"inspected_edges\":" << edges / n
    << ",\"alpha_rejected_edges\":" << reject / n
    << ",\"duplicate_rejected_edges\":" << dup / n
    << ",\"pq_comparisons\":" << pq / n << ",\"pq_space_evals\":" << pe / n
    << ",\"exact_space_evals\":" << ee / n
    << ",\"gt_discovered_pct\":" << 10. * discovered / n
    << ",\"gt_candidates_pct\":" << 10. * candidates / n
    << ",\"gt_rerank_pct\":" << 10. * rerank_gt / n
    << ",\"candidate_count\":" << pool_count / n
    << ",\"navigation_exact_space_evals\":" << nav_exact / n
    << ",\"pq_lookup_us\":" << lookup_us / n
    << ",\"expanded_nodes\":" << expanded / n
    << ",\"candidate_insertions\":" << inserted / n
    << ",\"reranked_nodes\":" << rc / n
    << ",\"topology_distinct_pages\":" << topo_distinct / n
    << ",\"coordinate_distinct_pages\":" << coord_distinct / n
    << ",\"cache_evictions\":" << evictions
    << ",\"topology_submitted_pages\":" << ts / n
    << ",\"coordinate_submitted_pages\":" << cs / n
    << ",\"topology_page_hits\":" << th / n
    << ",\"coordinate_page_hits\":" << ch / n
    << ",\"topology_shared_hits\":" << tsh / n
    << ",\"coordinate_shared_hits\":" << csh / n
    << ",\"topology_shared_cache_us\":" << tcu / n
    << ",\"coordinate_shared_cache_us\":" << ccu / n
    << ",\"requested_pages\":" << c.requested / n
    << ",\"page_hits\":" << c.hits / n
    << ",\"submitted_pages\":" << c.submitted / n
    << ",\"submitted_bytes\":" << c.bytes / n
    << ",\"logical_field_bytes\":" << c.useful / n
    << ",\"read_submit_wait_us\":" << c.wait_us / n
    << ",\"rss_kib\":" << r.ru_maxrss << "}";
}
void per_query(const std::string &path, const std::vector<Answer> &a,
               const std::vector<float> &alpha, U q) {
  std::ofstream out(path);
  out << std::setprecision(12) << "[";
  for (U j = 0; j < q; j++) {
    if (j)
      out << ',';
    double lat = 0;
    U edges = 0, reject = 0, dup = 0, pq = 0, expanded = 0, inserted = 0,
      rerank = 0, requested = 0, hits = 0, submitted = 0;
    U n = 0;
    for (U i = j; i < a.size(); i += q) {
      auto &s = a[i].stats;
      n++;
      lat += s.latency;
      edges += s.edges;
      reject += s.alpha_reject;
      dup += s.duplicate_reject;
      pq += s.pq;
      expanded += s.expanded;
      inserted += s.inserted;
      rerank += s.reranked;
      requested += s.io.requested;
      hits += s.io.hits;
      submitted += s.io.submitted;
    }
    double v = n;
    out << "{\"query_id\":" << j << ",\"alpha\":" << alpha[j]
        << ",\"samples\":" << n << ",\"latency_us\":" << lat / v
        << ",\"inspected_edges\":" << edges / v
        << ",\"alpha_rejected_edges\":" << reject / v
        << ",\"duplicate_rejected_edges\":" << dup / v
        << ",\"pq_comparisons\":" << pq / v
        << ",\"expanded_nodes\":" << expanded / v
        << ",\"candidate_insertions\":" << inserted / v
        << ",\"reranked_nodes\":" << rerank / v
        << ",\"requested_pages\":" << requested / v
        << ",\"page_hits\":" << hits / v
        << ",\"submitted_pages\":" << submitted / v << "}";
  }
  out << "]\n";
  if (!out)
    throw std::runtime_error("per-query output");
}
void run_queries(Source &s, Store &store, int argc, char **argv) {
  std::string prefix = argv[8];
  const std::string command = argv[1];
  const bool endpoint = command == "endpoint" || command == "compare_x";
  const bool compare = command == "compare_x";
  const bool single = s.d2 == 0;
  const std::string layout = argv[6];
  if (std::getenv("METRIC_CONCAT") && layout != "TXY" && layout != "T_XY")
    throw std::runtime_error(
        "concat kernel requires co-located contiguous coordinate fields");
  const char *first_only = std::getenv("LAYOUT_ENDPOINT_X_ONLY");
  const bool only_x =
      single || (endpoint && first_only && std::string(first_only) == "1");
  const bool x_layout = layout == "S_T_X";
  if (endpoint && (layout != "T_XY" && !x_layout))
    throw std::runtime_error("endpoint layout");
  if (endpoint && x_layout && !only_x)
    throw std::runtime_error("X-only layout requires X-only computation");
  if (single && (!compare || !x_layout || s.M != 0))
    throw std::runtime_error(
        "single-space graph requires X-only layout and no alpha metadata");
  U q, d, qb = 0, db = 0;
  auto q1 = matrix<float>(argv[9], q, d);
  std::vector<float> q2;
  if (!single) {
    q2 = matrix<float>(argv[10], qb, db);
    if (q != qb || db != s.d2)
      throw std::runtime_error("second query matrix identity");
  }
  std::vector<float> alpha(q, 1.0f);
  std::vector<unsigned> gt;
  if (!endpoint) {
    U qa, da;
    alpha = matrix<float>(argv[11], qa, da);
    if (qa != q || da != 1)
      throw std::runtime_error("weight identity");
  }
  if (!endpoint || compare) {
    U ng, dg;
    gt = matrix<unsigned>(argv[12], ng, dg);
    if (ng != q || dg != 10)
      throw std::runtime_error("GT identity");
  }
  const auto *truth = gt.empty() ? nullptr : &gt;
  unsigned threads = std::stoul(argv[13]);
  double duration = std::stod(argv[14]);
  bool one = std::getenv("LAYOUT_ONE_PASS");
  if (threads < 1 || threads > 64 || !q || d != s.d1)
    throw std::runtime_error("query identity");
  omp_set_num_threads(threads);
  const bool bypass = std::getenv("LAYOUT_BYPASS_ALPHA") != nullptr;
  if (std::getenv("METRIC_RERANK") &&
      std::stoul(std::getenv("METRIC_RERANK")) < 10)
    throw std::runtime_error("rerank depth must cover top10");
  const bool diagnostic = std::getenv("LAYOUT_GT_DIAGNOSTIC") != nullptr;
  if ((bypass && (single || !s.M || (endpoint && (!compare || !x_layout)))) ||
      (diagnostic && (!truth || !one)))
    throw std::runtime_error("invalid ablation or diagnostic configuration");
  Engine engine(s, prefix, only_x, x_layout ? 2 : 6, bypass);
  const char *shared_size = std::getenv("LAYOUT_SHARED_CACHE_BYTES");
  U shared_bytes = shared_size ? std::stoull(shared_size) : 0;
  if (shared_bytes % PAGE)
    throw std::runtime_error("shared cache must use complete physical pages");
  auto shared = shared_bytes
                    ? std::make_unique<SharedPages>(shared_bytes / PAGE)
                    : nullptr;
  std::vector<std::unique_ptr<Cache> > cache;
  std::vector<std::unique_ptr<Scratch> > scratch;
  for (unsigned t = 0; t < threads; t++) {
    cache.emplace_back(new Cache(store, shared.get()));
    scratch.emplace_back(new Scratch(s.d1, s.d2, engine.chunks, s.R));
  }
  std::vector<Answer> answers;
  std::vector<Trace> trace(q);
  bool trace_mode = std::getenv("LAYOUT_TRACE");
  if (trace_mode && !one)
    throw std::runtime_error("trace requires one pass");
  double point_start = 0;
  auto run = [&](U samples, unsigned L, bool record) {
    answers.resize(samples);
    point_start = std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    auto begin = Clock::now();
#pragma omp parallel for schedule(dynamic,1)
    for (int64_t i = 0; i < int64_t(samples); i++) {
      auto t = omp_get_thread_num();
      U j = i % q;
      auto call = Clock::now();
      answers[i] = engine.search(
          q1.data() + j * d, single ? nullptr : q2.data() + j * db, alpha[j], L,
          *cache[t], *scratch[t], record && trace_mode ? &trace[j] : nullptr,
          diagnostic && record ? gt.data() + 10 * j : nullptr);
      answers[i].stats.latency = us(call);
    }
    return us(begin) / 1e6;
  };
  std::ofstream output(argv[7]);
  output << "[";
  bool first = true;
  for (int arg = 15; arg < argc; arg++) {
    unsigned L = std::stoul(argv[arg]);
    if (L < 10 || L > 6400)
      throw std::runtime_error("L range");
    if (trace_mode)
      for (auto &t : trace)
        t = Trace{};
    if (shared)
      shared->clear();
    double sec = run(q, L, false);
    U samples = one ? q : std::max(q, U(std::ceil(duration * 1.1 / sec)) * q);
    unsigned tries = 0;
    do {
      if (samples > 2000000)
        throw std::runtime_error("sample cap");
      sec = run(samples, L, trace_mode || diagnostic);
      if (one || sec >= duration)
        break;
      samples = std::max(samples + q,
                         U(std::ceil(duration * 1.1 / sec * samples / q)) * q);
      if (++tries > 5)
        throw std::runtime_error("calibration failed");
    } while (true);
    if (trace_mode)
      save_trace(std::string(std::getenv("LAYOUT_TRACE")) + ".L" +
                     std::to_string(L),
                 trace);
    if (const char *dump = std::getenv("LAYOUT_DUMP")) {
      if (!one)
        throw std::runtime_error("dump requires one pass");
      std::ofstream ids(std::string(dump) + ".L" + std::to_string(L) + ".ids",
                        std::ios::binary),
          dist(std::string(dump) + ".L" + std::to_string(L) + ".dists",
               std::ios::binary);
      for (auto &a : answers) {
        ids.write((char *)a.ids.data(), 40);
        dist.write((char *)a.dists.data(), 40);
      }
    }
    if (!first)
      output << ",";
    first = false;
    emit_point(output, answers, sec, L, threads, q, truth, point_start,
               store.topology.size());
    output.flush();
    per_query(std::string(argv[7]) + ".L" + std::to_string(L) + ".queries.json",
              answers, alpha, q);
    std::cout << "LAYOUT_POINT ";
    emit_point(std::cout, answers, sec, L, threads, q, truth, point_start,
               store.topology.size());
    std::cout << std::endl;
  }
  output << "]\n";
  if (!output)
    throw std::runtime_error("query output");
}
void run_replay(Store &store, char **argv) {
  auto trace = load_trace(argv[8]);
  unsigned threads = std::stoul(argv[9]);
  double duration = std::stod(argv[10]);
  if (!threads || threads > 64 || trace.size() != 1000)
    throw std::runtime_error("replay identity");
  omp_set_num_threads(threads);
  std::vector<std::unique_ptr<Cache> > c;
  for (unsigned t = 0; t < threads; t++)
    c.emplace_back(new Cache(store));
  std::vector<Answer> a;
  double point_start = 0;
  auto run = [&](U n) {
    a.resize(n);
    point_start = std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    auto begin = Clock::now();
#pragma omp parallel for schedule(dynamic,1)
    for (int64_t i = 0; i < int64_t(n); i++) {
      auto start = Clock::now();
      auto &cache = *c[omp_get_thread_num()];
      cache.reset();
      const auto &t = trace[i % trace.size()];
      for (auto &batch : t.batches)
        cache.fetch(batch, 1);
      int mask = std::getenv("LAYOUT_REPLAY_MASK")
                     ? std::stoi(std::getenv("LAYOUT_REPLAY_MASK"))
                     : 6;
      for (U first = 0; first < t.coords.size(); first += 256) {
        std::vector<unsigned> ids(
            t.coords.begin() + first,
            t.coords.begin() + std::min(U(t.coords.size()), first + 256));
        cache.fetch(ids, mask);
      }
      a[i].stats = Stats{};
      a[i].stats.latency = us(start);
      a[i].stats.io = cache.count;
    }
    return us(begin) / 1e6;
  };
  double sec = run(trace.size());
  U samples = std::max(U(trace.size()),
                       U(std::ceil(duration * 1.1 / sec)) * trace.size());
  do {
    if (samples > 2000000)
      throw std::runtime_error("sample cap");
    sec = run(samples);
    if (sec >= duration)
      break;
    samples += trace.size();
  } while (true);
  std::ofstream o(argv[7]);
  emit_point(o, a, sec, 0, threads, trace.size(), nullptr, point_start);
  o << '\n';
}
void run_patch(Source &s, Store &store, char **argv) {
  U seed = std::stoull(argv[8]), n = std::stoull(argv[9]);
  int mask = std::stoi(argv[10]);
  bool hot = std::string(argv[11]) == "hot";
  std::mt19937_64 rng(seed);
  std::vector<unsigned> ids(n);
  for (auto &id : ids)
    id =
        hot && rng() % 10 < 9 ? rng() % std::max(U(1), s.n / 100) : rng() % s.n;
  std::unordered_map<U, std::vector<uint8_t> > before, expected;
  std::vector<std::array<std::vector<uint8_t>, 3> > payload(n);
  std::vector<std::vector<U> > keys(n);
  auto read_page = [&](U key, uint8_t *buf) {
    if (pread(store.all[key % 8].fd, buf, PAGE, (key / 8) * PAGE) != PAGE)
      throw std::runtime_error("patch read");
  };
  auto write_page = [&](U key, uint8_t *buf) {
    if (pwrite(store.all[key % 8].fd, buf, PAGE, (key / 8) * PAGE) != PAGE)
      throw std::runtime_error("patch write");
  };
  uint8_t *aligned = nullptr;
  if (posix_memalign((void **)&aligned, PAGE, PAGE))
    throw std::runtime_error("patch alignment");
  U logical = 0;
  for (U i = 0; i < n; i++) {
    for (int k = 0; k < 3; k++)
      if (mask & (1 << k)) {
        U bytes = s.bytes(1 << k);
        logical += bytes;
        payload[i][k]
            .assign(s.field(ids[i], 1 << k), s.field(ids[i], 1 << k) + bytes);
        if (k) {
          for (U j = 0; j < bytes; j += 4)
            payload[i][k][j] ^= uint8_t(1 + (i % 3));
        } else if (read32(payload[i][k].data()) && !s.M) {
          unsigned replacement = (read32(payload[i][k].data() + 4) + 1) % s.n;
          memcpy(payload[i][k].data() + 4, &replacement, 4);
        } else if (read32(payload[i][k].data())) {
          auto &v = payload[i][k];
          if (v[9] < 100)
            v[9]++;
          else if (v[8])
            v[8]--;
          else
            v[9] = 99;
        }
        U key = store.key(ids[i], 1 << k);
        keys[i].push_back(key);
        if (!before.count(key)) {
          read_page(key, aligned);
          before[key] = std::vector<uint8_t>(aligned, aligned + PAGE);
          expected[key] = before[key];
        }
      }
    std::sort(keys[i].begin(), keys[i].end());
    keys[i].erase(std::unique(keys[i].begin(), keys[i].end()), keys[i].end());
    for (int k = 0; k < 3; k++)
      if (mask & (1 << k))
        memcpy(expected[store.key(ids[i], 1 << k)].data() +
                   store.offset(ids[i], 1 << k),
               payload[i][k].data(), payload[i][k].size());
  }
  bool coalesce = std::getenv("LAYOUT_PATCH_COALESCE") &&
                  std::string(std::getenv("LAYOUT_PATCH_COALESCE")) == "1";
  U batch_size = std::getenv("LAYOUT_PATCH_BATCH")
                     ? std::stoull(std::getenv("LAYOUT_PATCH_BATCH"))
                     : 128;
  if (!batch_size || batch_size > 128 || !n)
    throw std::runtime_error("patch batch limit");
  uint8_t *buffer = nullptr;
  if (posix_memalign((void **)&buffer, PAGE, 256 * PAGE))
    throw std::runtime_error("patch buffer");
  std::vector<double> lat;
  U reads = 0, writes = 0, max_pages = 0;
  double sync = 0;
  auto start = Clock::now();
  for (U first = 0; first < n; first += batch_size) {
    U last = std::min(n, first + batch_size);
    auto begin = Clock::now();
    std::unordered_set<U> dirty;
    if (coalesce) {
      std::vector<U> pages;
      for (U i = first; i < last; i++)
        pages.insert(pages.end(), keys[i].begin(), keys[i].end());
      std::sort(pages.begin(), pages.end());
      pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
      if (pages.size() > 256)
        throw std::runtime_error("patch page limit");
      max_pages = std::max(max_pages, U(pages.size()));
      for (U j = 0; j < pages.size(); j++) {
        read_page(pages[j], buffer + j * PAGE);
        reads++;
      }
      for (U i = first; i < last; i++)
        for (int k = 0; k < 3; k++)
          if (mask & (1 << k)) {
            U key = store.key(ids[i], 1 << k);
            U j = std::lower_bound(pages.begin(), pages.end(), key) -
                  pages.begin();
            memcpy(buffer + j * PAGE + store.offset(ids[i], 1 << k),
                   payload[i][k].data(), payload[i][k].size());
          }
      for (U j = 0; j < pages.size(); j++) {
        write_page(pages[j], buffer + j * PAGE);
        writes++;
        dirty.insert(pages[j] % 8);
      }
    } else {
      for (U i = first; i < last; i++)
        for (auto key : keys[i]) {
          read_page(key, aligned);
          reads++;
          for (int k = 0; k < 3; k++)
            if ((mask & (1 << k)) && store.key(ids[i], 1 << k) == key)
              memcpy(aligned + store.offset(ids[i], 1 << k),
                     payload[i][k].data(), payload[i][k].size());
          write_page(key, aligned);
          writes++;
          dirty.insert(key % 8);
        }
    }
    auto t = Clock::now();
    for (auto g : dirty)
      if (fdatasync(store.all[g].fd))
        throw std::runtime_error("patch sync");
    sync += us(t);
    lat.push_back(us(begin));
  }
  double elapsed = us(start) / 1e6;
  free(buffer);
  for (auto &p : expected) {
    read_page(p.first, aligned);
    if (memcmp(aligned, p.second.data(), PAGE))
      throw std::runtime_error("patch contents");
  }
  for (auto &p : before) {
    memcpy(aligned, p.second.data(), PAGE);
    write_page(p.first, aligned);
  }
  for (auto &g : store.all)
    if (fdatasync(g.fd))
      throw std::runtime_error("restore sync");
  for (auto &p : before) {
    read_page(p.first, aligned);
    if (memcmp(aligned, p.second.data(), PAGE))
      throw std::runtime_error("snapshot restore");
  }
  free(aligned);
  std::sort(lat.begin(), lat.end());
  std::ofstream o(argv[7]);
  o << std::setprecision(12) << "{\"status\":\"PASS\",\"operations\":" << n
    << ",\"seed\":" << seed << ",\"field_mask\":" << mask
    << ",\"hot\":" << (hot ? "true" : "false") << ",\"seconds\":" << elapsed
    << ",\"operations_per_second\":" << n / elapsed
    << ",\"logical_modified_bytes\":" << logical
    << ",\"submitted_read_bytes\":" << reads * PAGE
    << ",\"submitted_write_bytes\":" << writes * PAGE
    << ",\"coalesce\":" << (coalesce ? "true" : "false")
    << ",\"p50_batch_commit_us\":" << lat[lat.size() / 2]
    << ",\"p95_batch_commit_us\":" << lat[U(.95 * lat.size())]
    << ",\"p99_batch_commit_us\":" << lat[U(.99 * lat.size())]
    << ",\"amortized_us_per_operation\":" << elapsed * 1e6 / n
    << ",\"fdatasync_batch_operations\":" << batch_size
    << ",\"fdatasync_total_us\":" << sync << ",\"batch_count\":" << lat.size()
    << ",\"fixed_buffer_bytes\":" << 256 * PAGE
    << ",\"max_coalesced_pages\":" << max_pages
    << ",\"modified_page_mismatches\":0,\"restored_page_mismatches\":0,"
       "\"durability\":\"same per-batch fdatasync and acknowledgement; no "
       "WAL\","
       "\"scope\":\"single-writer field patch probe, not dynamic ANN "
       "updates\"}\n";
}
int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "selftest") {
      if (!(metric_score(1, 9, .5, false, false) <
            metric_score(4.84, 4.84, .5, false, false)) ||
          !(metric_score(1, 9, .5, false, true) >
            metric_score(4.84, 4.84, .5, false, true)) ||
          metric_score(25, 0, 1, true, false) != 5)
        throw std::runtime_error("metric fusion contract");
      alignas(64) float zero[32] = {}, one[32] = {}, three[32] = {};
      one[0] = 1;
      three[0] = 3;
      auto kernel = std::unique_ptr<pipeann::Distance<float> >(
          pipeann::get_distance_function<float>(pipeann::Metric::L2));
      if (kernel->compare(zero, one, 32) != 1 ||
          kernel->compare(zero, three, 32) != 9)
        throw std::runtime_error("raw kernel must remain squared L2");
      std::vector<uint8_t> codes(128, 0);
      std::vector<float> table(128 * 256, 0);
      for (unsigned chunk = 0; chunk < 128; ++chunk)
        table[chunk * 256] = chunk % 5;
      float combined, first, second;
      pq_dist_lookup(codes.data(), 1, 128, table.data(), &combined);
      pq_dist_lookup(codes.data(), 1, 64, table.data(), &first);
      pq_dist_lookup(codes.data() + 64, 1, 64, table.data() + 64 * 256,
                     &second);
      if (combined != first + second)
        throw std::runtime_error("128-chunk concat ADC");
      std::cout << "PASS actual metric fusion, squared kernel, and 128-chunk "
                   "single-vector ADC" << std::endl;
      return 0;
    }
    if (argc < 8)
      throw std::runtime_error(
          "command meta topo coord directory layout output [parameters]");
    Source source(argv[2], argv[3], argv[4]);
    std::string command = argv[1];
    if (command == "build")
      build_and_validate(source, argv[6], argv[5], argv[7]);
    else {
      Store store(source, argv[6], argv[5]);
      if (command == "query" || command == "endpoint" || command == "compare_x")
        run_queries(source, store, argc, argv);
      else if (command == "replay")
        run_replay(store, argv);
      else if (command == "patch")
        run_patch(source, store, argv);
      else
        throw std::runtime_error("command");
    }
    std::cout << "LAYOUT_OPERATION_COMPLETE" << std::endl;
  }
  catch (const std::exception &e) {
    std::cerr << "LAYOUT_ERROR " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
