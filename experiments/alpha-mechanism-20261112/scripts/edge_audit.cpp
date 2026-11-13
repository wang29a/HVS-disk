// Read-only alpha activity and first-discovery provenance from existing traces.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

template <class T> T get(const uint8_t *p) {
  T x;
  std::memcpy(&x, p, sizeof x);
  return x;
}
uint32_t next(std::ifstream &in) {
  uint32_t x;
  in.read(reinterpret_cast<char *>(&x), 4);
  if (!in)
    throw std::runtime_error("truncated input");
  return x;
}
struct Map {
  int fd;
  size_t bytes;
  const uint8_t *p;
  explicit Map(const std::string &path) {
    fd = open(path.c_str(), O_RDONLY);
    struct stat s {};
    if (fd < 0 || fstat(fd, &s))
      throw std::runtime_error("map open");
    bytes = s.st_size;
    p = static_cast<const uint8_t *>(
        mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0));
    if (p == MAP_FAILED)
      throw std::runtime_error("map failed");
  }
  ~Map() {
    munmap(const_cast<uint8_t *>(p), bytes);
    close(fd);
  }
};
struct Graph {
  Map meta, topo;
  uint32_t n, R, M, entries;
  uint64_t npp, width;
  Graph(const char *m, const char *t) : meta(m), topo(t) {
    if (meta.bytes != 8192)
      throw std::runtime_error("meta size");
    n = get<uint32_t>(meta.p);
    R = get<uint32_t>(meta.p + 12);
    M = get<uint32_t>(meta.p + 16);
    npp = get<uint64_t>(meta.p + 20);
    entries = get<uint32_t>(meta.p + 28);
    width = 4 + uint64_t(R) * (4 + 2 * M);
    if (!n || !M || M > 64 || !R || !npp || npp != 8192 / width || !entries ||
        32 + uint64_t(entries) * 4 > 8192 ||
        topo.bytes != ((uint64_t(n) + npp - 1) / npp) * 8192)
      throw std::runtime_error("graph format");
  }
  const uint8_t *node(uint32_t id) const {
    if (id >= n)
      throw std::runtime_error("node ID");
    return topo.p + (id / npp) * 8192 + (id % npp) * width;
  }
  std::vector<std::pair<int, int> > ranges(const uint8_t *edge) const {
    std::vector<std::pair<int, int> > out;
    for (uint32_t j = 0; j < M; ++j) {
      int a = edge[4 + j * 2], b = edge[5 + j * 2];
      if (a == 0 && b == 0)
        break;
      if (a > 100 || b > 100 || a > b ||
          (!out.empty() && a <= out.back().second))
        throw std::runtime_error("invalid/overlapping interval");
      out.emplace_back(a, b);
    }
    return out;
  }
  bool active(const uint8_t *edge, float alpha) const {
    float weight = alpha * 100.f;
    for (auto r : ranges(edge))
      if (weight >= r.first && weight <= r.second)
        return true;
    return false;
  }
};
void activity(Graph &g, const char *path) {
  std::array<int64_t, 102> difference{};
  uint64_t edges = 0, pairs = 0;
  for (uint32_t id = 0; id < g.n; ++id) {
    const uint8_t *p = g.node(id);
    uint32_t d = get<uint32_t>(p);
    if (d > g.R)
      throw std::runtime_error("degree");
    for (uint32_t e = 0; e < d; ++e) {
      const uint8_t *edge = p + 4 + e * (4 + 2 * g.M);
      if (get<uint32_t>(edge) >= g.n)
        throw std::runtime_error("neighbor ID");
      auto ranges = g.ranges(edge);
      if (ranges.empty())
        throw std::runtime_error("edge has no active range");
      ++edges;
      for (auto r : ranges) {
        ++difference[r.first];
        --difference[r.second + 1];
        ++pairs;
      }
    }
  }
  std::ofstream out(path);
  out << "{\"nodes\":" << g.n << ",\"edges\":" << edges
      << ",\"stored_nonempty_pairs\":" << pairs << ",\"full_slots\":" << g.M
      << ",\"nodes_per_page\":" << g.npp << ",\"activity\":[";
  int64_t active = 0;
  for (int a = 0; a <= 100; ++a) {
    active += difference[a];
    if (active < 0 || uint64_t(active) > edges)
      throw std::runtime_error("activity count");
    if (a)
      out << ',';
    out << "{\"alpha100\":" << a << ",\"active_edges\":" << active
        << ",\"total_edges\":" << edges << '}';
  }
  out << "]}\n";
  if (!out)
    throw std::runtime_error("output");
}
void provenance(Graph &g, const char *trace_path, const char *alpha_path,
                const char *ids_path, const char *gt_path,
                const char *out_path) {
  std::ifstream trace(trace_path, std::ios::binary),
      weights(alpha_path, std::ios::binary), ids(ids_path, std::ios::binary),
      gt(gt_path, std::ios::binary);
  if (next(trace) != 0x48565331)
    throw std::runtime_error("trace magic");
  uint32_t q = next(trace);
  if (next(weights) != q || next(weights) != 1 || next(gt) != q ||
      next(gt) != 10)
    throw std::runtime_error("workload shape");
  std::ofstream out(out_path);
  out << "[";
  for (uint32_t j = 0; j < q; ++j) {
    float alpha;
    weights.read(reinterpret_cast<char *>(&alpha), 4);
    if (!weights || !std::isfinite(alpha) || alpha < 0 || alpha > 1)
      throw std::runtime_error("alpha");
    std::unordered_map<uint32_t, bool> first;
    std::unordered_set<uint32_t> entry_ids;
    for (uint32_t i = 0; i < g.entries; ++i) {
      uint32_t id = get<uint32_t>(g.meta.p + 32 + i * 4);
      first.emplace(id, false);
      entry_ids.insert(id);
    }
    uint64_t expanded = 0, edges = 0, inactive = 0, new_edges = 0,
             new_inactive = 0;
    uint32_t batches = next(trace);
    for (uint32_t b = 0; b < batches; ++b) {
      uint32_t size = next(trace);
      for (uint32_t k = 0; k < size; ++k) {
        uint32_t id = next(trace);
        const uint8_t *p = g.node(id);
        uint32_t degree = get<uint32_t>(p);
        ++expanded;
        if (degree > g.R)
          throw std::runtime_error("trace degree");
        for (uint32_t e = 0; e < degree; ++e) {
          const uint8_t *edge = p + 4 + e * (4 + 2 * g.M);
          uint32_t to = get<uint32_t>(edge);
          if (to >= g.n)
            throw std::runtime_error("edge ID");
          bool blocked = !g.active(edge, alpha);
          ++edges;
          inactive += blocked;
          if (first.emplace(to, blocked).second) {
            ++new_edges;
            new_inactive += blocked;
          }
        }
      }
    }
    uint32_t nc = next(trace), candidate_inactive = 0, candidate_nonentry = 0;
    for (uint32_t c = 0; c < nc; ++c) {
      uint32_t id = next(trace);
      auto it = first.find(id);
      if (it == first.end())
        throw std::runtime_error("candidate not discovered");
      if (!entry_ids.count(id)) {
        ++candidate_nonentry;
        candidate_inactive += it->second;
      }
    }
    std::unordered_set<uint32_t> truth;
    for (uint32_t k = 0; k < 10; ++k)
      truth.insert(next(gt));
    uint32_t returned_inactive = 0, returned_nonentry = 0, true_inactive = 0,
             true_nonentry = 0, true_hits = 0;
    for (uint32_t k = 0; k < 10; ++k) {
      uint32_t id = next(ids);
      auto it = first.find(id);
      if (it == first.end())
        throw std::runtime_error("returned ID not discovered");
      bool is_true = truth.count(id);
      true_hits += is_true;
      if (!entry_ids.count(id)) {
        ++returned_nonentry;
        returned_inactive += it->second;
        if (is_true) {
          ++true_nonentry;
          true_inactive += it->second;
        }
      }
    }
    if (j)
      out << ',';
    out << "{\"query_id\":" << j << ",\"alpha\":" << alpha
        << ",\"expanded_nodes\":" << expanded
        << ",\"inspected_edges\":" << edges
        << ",\"inactive_inspected_edges\":" << inactive
        << ",\"first_discovery_edges\":" << new_edges
        << ",\"inactive_first_discovery_edges\":" << new_inactive
        << ",\"rerank_candidates\":" << nc
        << ",\"candidate_nonentry\":" << candidate_nonentry
        << ",\"candidate_inactive_parent\":" << candidate_inactive
        << ",\"returned_nonentry\":" << returned_nonentry
        << ",\"returned_inactive_parent\":" << returned_inactive
        << ",\"true_hits\":" << true_hits
        << ",\"true_nonentry\":" << true_nonentry
        << ",\"true_inactive_parent\":" << true_inactive << '}';
  }
  if (trace.peek() != std::char_traits<char>::eof() ||
      weights.peek() != std::char_traits<char>::eof() ||
      ids.peek() != std::char_traits<char>::eof() ||
      gt.peek() != std::char_traits<char>::eof())
    throw std::runtime_error("input tail");
  out << "]\n";
  if (!out)
    throw std::runtime_error("output");
}
int main(int argc, char **argv) {
  try {
    if (argc != 5 && argc != 9)
      throw std::runtime_error(
          "static meta topo out | trace meta topo trace alpha ids gt out");
    Graph g(argv[2], argv[3]);
    if (std::string(argv[1]) == "static" && argc == 5)
      activity(g, argv[4]);
    else if (std::string(argv[1]) == "trace" && argc == 9)
      provenance(g, argv[4], argv[5], argv[6], argv[7], argv[8]);
    else
      throw std::runtime_error("mode");
  }
  catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
