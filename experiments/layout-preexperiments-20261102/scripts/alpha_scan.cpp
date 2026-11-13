#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

using U = uint64_t;
constexpr U PAGE = 8192;
struct Map {
  int fd;
  U size;
  const uint8_t *p;
  explicit Map(const char *name) {
    fd = open(name, O_RDONLY);
    struct stat s {};
    if (fd < 0 || fstat(fd, &s))
      throw std::runtime_error("open");
    size = s.st_size;
    auto v = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (v == MAP_FAILED)
      throw std::runtime_error("mmap");
    p = (uint8_t *)v;
  }
  ~Map() {
    munmap((void *)p, size);
    close(fd);
  }
};
uint32_t u32(const uint8_t *p) {
  uint32_t v;
  memcpy(&v, p, 4);
  return v;
}
U u64(const uint8_t *p) {
  U v;
  memcpy(&v, p, 8);
  return v;
}
struct Stats {
  U edges = 0, used = 0, canonical = 0, actual_hits = 0, rounded_hits = 0,
    semantic_diff = 0;
  U empty_valid_bytes = 0, unused_neighbor_bytes = 0, bad = 0,
    overlap_edges = 0;
  std::array<U, 17> stored{}, components{}, node_max{};
  std::array<U, 101> widths{}, gaps{}, starts{}, ends{};
  std::array<U, 1001> node_filter{};
};
template <size_t N> void emit(std::ostream &o, const std::array<U, N> &a) {
  o << '[';
  for (size_t i = 0; i < N; i++) {
    if (i)
      o << ',';
    o << a[i];
  }
  o << ']';
}
using Interval = std::pair<int, int>;
std::vector<Interval> normalize(std::vector<Interval> a) {
  std::sort(a.begin(), a.end());
  std::vector<Interval> b;
  for (auto x : a) {
    if (b.empty() || x.first > b.back().second)
      b.push_back(x);
    else
      b.back().second = std::max(b.back().second, x.second);
  }
  return b;
}
int main(int argc, char **argv) {
  try {
    if (argc != 6)
      throw std::runtime_error("alpha_scan meta topology alpha threads output");
    Map meta(argv[1]), topo(argv[2]), query(argv[3]);
    U n = u32(meta.p), R = u32(meta.p + 12), M = u32(meta.p + 16),
      npp = u64(meta.p + 20), q = u32(query.p);
    U width = 4 + R * (4 + 2 * M);
    if (meta.size != PAGE || M == 0 || M > 16 || R > 512 ||
        npp != PAGE / width || query.size != 8 + q * 4 ||
        u32(query.p + 4) != 1 || !q ||
        topo.size != ((n + npp - 1) / npp) * PAGE)
      throw std::runtime_error("layout mismatch");
    int threads = std::stoi(argv[4]);
    if (threads < 1 || threads > 64)
      throw std::runtime_error("thread cap");
    omp_set_num_threads(threads);
    std::vector<float> alphas(q), rounded(q);
    for (U i = 0; i < q; i++) {
      float a;
      memcpy(&a, query.p + 8 + 4 * i, 4);
      if (!std::isfinite(a) || a < 0 || a > 1)
        throw std::runtime_error("alpha");
      volatile float scaled = a * 100.f;
      alphas[i] = scaled;
      rounded[i] = std::nearbyint(scaled);
    }
    U counts[101][101]{}, rcounts[101][101]{};
    for (int l = 0; l <= 100; l++)
      for (int h = l; h <= 100; h++)
        for (U i = 0; i < q; i++) {
          counts[l][h] += alphas[i] >= l && alphas[i] <= h;
          rcounts[l][h] += rounded[i] >= l && rounded[i] <= h;
        }
    std::vector<Stats> work(threads);
#pragma omp parallel for schedule(static)
    for (U id = 0; id < n; id++) {
      auto &s = work[omp_get_thread_num()];
      auto p = topo.p + (id / npp) * PAGE + (id % npp) * width;
      U d = u32(p), node_hits = 0;
      int maxc = 0;
      if (d > R) {
        s.bad++;
        continue;
      }
      s.unused_neighbor_bytes += (R - d) * (4 + 2 * M);
      for (U e = 0; e < d; e++) {
        auto edge = p + 4 + e * (4 + 2 * M);
        if (u32(edge) >= n)
          s.bad++;
        std::vector<Interval> raw, reader;
        int lower = 0;
        bool stopped = false;
        for (U k = 0; k < M; k++) {
          int l = (int8_t)edge[4 + 2 * k], h = (int8_t)edge[5 + 2 * k];
          if (l == 0 && h == 0) {
            stopped = true;
            continue;
          }
          if (l < 0 || h < l || h > 100 || stopped) {
            s.bad++;
            continue;
          }
          raw.emplace_back(l, h);
          s.starts[l]++;
          s.ends[h]++;
          lower = std::max(lower, l);
          if (lower <= h)
            reader.emplace_back(lower, h);
        }
        auto canonical = normalize(raw), effective = normalize(reader);
        U ah = 0, rh = 0, ch = 0;
        int w = 0;
        for (auto x : canonical) {
          w += x.second - x.first;
          ch += counts[x.first][x.second];
          rh += rcounts[x.first][x.second];
        }
        for (auto x : effective)
          ah += counts[x.first][x.second];
        if (ah != ch)
          s.semantic_diff++;
        if (canonical.size() < raw.size())
          s.overlap_edges++;
        for (size_t k = 1; k < canonical.size(); k++)
          s.gaps[canonical[k].first - canonical[k - 1].second]++;
        s.edges++;
        s.used += raw.size();
        s.canonical += canonical.size();
        s.stored[raw.size()]++;
        s.components[canonical.size()]++;
        s.widths[w]++;
        s.actual_hits += ah;
        s.rounded_hits += rh;
        node_hits += ah;
        s.empty_valid_bytes += (M - raw.size()) * 2;
        maxc = std::max(maxc, (int)canonical.size());
      }
      s.node_max[maxc]++;
      int bin =
          d ? std::min(1000, int(1000. * (1. - double(node_hits) / q / d))) : 0;
      s.node_filter[bin]++;
    }
    Stats all;
    for (auto &s : work) {
      all.edges += s.edges;
      all.used += s.used;
      all.canonical += s.canonical;
      all.actual_hits += s.actual_hits;
      all.rounded_hits += s.rounded_hits;
      all.semantic_diff += s.semantic_diff;
      all.empty_valid_bytes += s.empty_valid_bytes;
      all.unused_neighbor_bytes += s.unused_neighbor_bytes;
      all.bad += s.bad;
      all.overlap_edges += s.overlap_edges;
#define ADD(name)                                                              \
  for (size_t k = 0; k < all.name.size(); k++)                                 \
  all.name[k] += s.name[k]
      ADD(stored);
      ADD(components);
      ADD(node_max);
      ADD(widths);
      ADD(gaps);
      ADD(starts);
      ADD(ends);
      ADD(node_filter);
#undef ADD
    }
    U padding_bad = 0;
#pragma omp parallel for reduction(+ : padding_bad)
    for (U pg = 0; pg < topo.size / PAGE; pg++) {
      U used = std::min(npp, n - pg * npp) * width;
      for (U j = used; j < PAGE; j++)
        padding_bad += topo.p[pg * PAGE + j] != 0;
    }
    std::ofstream o(argv[5]);
    o.precision(17);
    o << "{\"nodes\":" << n << ",\"edges\":" << all.edges << ",\"degree\":" << R
      << ",\"slots\":" << M << ",\"record_bytes\":" << width
      << ",\"nodes_per_page\":" << npp << ",\"queries\":" << q
      << ",\"threads\":" << threads << ",\"invalid_records\":" << all.bad
      << ",\"nonzero_page_padding\":" << padding_bad
      << ",\"allocated_pairs\":" << n *R *M << ",\"used_pairs\":" << all.used
      << ",\"canonical_components\":" << all.canonical
      << ",\"empty_slots_within_valid_edges_bytes\":" << all.empty_valid_bytes
      << ",\"unused_neighbor_record_bytes\":" << all.unused_neighbor_bytes
      << ",\"page_padding_bytes\":" << topo.size - n *width
      << ",\"runtime_filter_probability\":"
      << 1. - double(all.actual_hits) / all.edges / q
      << ",\"historical_rounded_filter_probability\":"
      << 1. - double(all.rounded_hits) / all.edges / q
      << ",\"reader_vs_canonical_disagreement_edges\":" << all.semantic_diff
      << ",\"overlapping_or_touching_edges\":" << all.overlap_edges;
#define OUTPUT(name)                                                           \
  o << ",\"" #name "\":";                                                      \
  emit(o, all.name)
    OUTPUT(stored);
    OUTPUT(components);
    OUTPUT(node_max);
    OUTPUT(widths);
    OUTPUT(gaps);
    OUTPUT(starts);
    OUTPUT(ends);
    OUTPUT(node_filter);
#undef OUTPUT
    o << "}\n";
    if (!o)
      throw std::runtime_error("output failed");
    if (all.bad || padding_bad)
      throw std::runtime_error("validation failed; inspect output");
    std::cout << "ALPHA_SCAN_PASS nodes=" << n << " edges=" << all.edges
              << std::endl;
  }
  catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}
