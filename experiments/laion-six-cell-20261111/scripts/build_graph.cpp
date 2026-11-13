#include "index.h"
#include "parameters.h"
#include "layout_store.h"
#include <cmath>
#include <iomanip>
#include <omp.h>

using namespace layout_exp;
using Intervals = std::vector<std::pair<float, float> >;
using Quantized = std::vector<std::pair<int8_t, int8_t> >;

void durable(const std::string &path) {
  int fd = ::open(path.c_str(), O_RDWR);
  if (fd < 0 || ::fsync(fd))
    throw std::runtime_error("checkpoint fsync");
  ::close(fd);
}

Quantized quantize(const Intervals &input) {
  Quantized out;
  for (auto p : input) {
    if (!std::isfinite(p.first) || !std::isfinite(p.second) || p.first < 0 ||
        p.second > 1 || p.first > p.second)
      throw std::runtime_error("invalid floating interval");
    int lo = int(std::floor(double(p.first) * 100));
    int hi = int(std::ceil(double(p.second) * 100));
    // (0,0) is the existing disk reader's unused-slot sentinel.
    if (!lo && !hi)
      hi = 1;
    out.emplace_back(lo, hi);
  }
  std::sort(out.begin(), out.end());
  Quantized merged;
  for (auto p : out) {
    if (!merged.empty() && p.first <= merged.back().second)
      merged.back().second = std::max(merged.back().second, p.second);
    else
      merged.push_back(p);
  }
  return merged;
}

template <class T> void write(std::ofstream &out, const T &value) {
  out.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

// Save the exact float source graph before quantization. Unlike mem.index,
// this artifact preserves pre-serialization ranges and can be audited again.
template <class Graph, class Range>
void save_graph(const Graph &graph, Range ranges, unsigned d1, unsigned d2,
                const std::vector<unsigned> &entries, const std::string &dir) {
  const U n = graph.size();
  const unsigned R = 40;
  unsigned M = 0;
  U edges = 0, pairs = 0, truncation_shrunk_edges = 0,
    quantized_lost_points = 0;
  double truncation_lost_width = 0;
  std::ofstream raw(dir + "/float-graph.bin", std::ios::binary);
  write(raw, n);
  write(raw, d1);
  write(raw, d2);
  for (U i = 0; i < n; ++i) {
    unsigned degree = graph[i].size();
    if (degree > R)
      throw std::runtime_error("degree exceeds construction budget");
    write(raw, degree);
    for (const auto &edge : graph[i]) {
      const unsigned id = ranges.id(edge);
      if (id >= n || id == i)
        throw std::runtime_error("invalid graph neighbor");
      const auto &original = ranges.intervals(edge);
      const auto compact = quantize(original);
      M = std::max(M, unsigned(compact.size()));
      ++edges;
      pairs += original.size();
      write(raw, id);
      unsigned count = original.size();
      write(raw, count);
      bool shrunk = false;
      for (auto p : original) {
        write(raw, p.first);
        write(raw, p.second);
        const double old_hi = int(p.second * 100.f) / 100.0;
        shrunk |= old_hi + 1e-10 < p.second;
        truncation_lost_width +=
            std::max(0.0, double(p.second) - std::max(double(p.first), old_hi));
        // Continuous containment below also implies containment of every
        // integer-alpha point; avoid 101 redundant tests per interval on
        // the 10.6M-node source graph.
        bool contained = false;
        for (auto q : compact)
          contained |=
              q.first / 100.0 <= p.first && q.second / 100.0 >= p.second;
        if (!contained)
          throw std::runtime_error("quantization narrowed float coverage");
      }
      truncation_shrunk_edges += shrunk;
    }
  }
  raw.close();
  if (!raw || quantized_lost_points)
    throw std::runtime_error("float audit/save failed");
  durable(dir + "/float-graph.bin");
  for (unsigned slots :
       (M && !std::getenv("HVS_FULL_ONLY") ? std::vector<unsigned>{ M, 0 }
                                           : std::vector<unsigned>{ M })) {
    const std::string stem = dir + (slots ? "/full" : "/compact");
    const U width = 4 + R * (4 + 2 * slots), npp = PAGE / width;
    if (!npp)
      throw std::runtime_error("topology record exceeds page");
    std::array<uint8_t, PAGE> meta{};
    unsigned nn = n, ne = entries.size();
    memcpy(meta.data(), &nn, 4);
    memcpy(meta.data() + 4, &d1, 4);
    memcpy(meta.data() + 8, &d2, 4);
    memcpy(meta.data() + 12, &R, 4);
    memcpy(meta.data() + 16, &slots, 4);
    memcpy(meta.data() + 20, &npp, 8);
    memcpy(meta.data() + 28, &ne, 4);
    if (!ne || 32 + ne * 4 > PAGE)
      throw std::runtime_error("entry metadata overflow");
    memcpy(meta.data() + 32, entries.data(), ne * 4);
    std::ofstream metadata(stem + ".meta", std::ios::binary);
    metadata.write(reinterpret_cast<char *>(meta.data()), PAGE);
    metadata.close();
    durable(stem + ".meta");
    if (std::getenv("HVS_SOURCE_ONLY"))
      continue;
    std::ofstream topology(stem + ".topo", std::ios::binary);
    for (U first = 0; first < n; first += npp) {
      std::array<uint8_t, PAGE> page{};
      for (U j = 0; j < npp && first + j < n; ++j) {
        auto record = page.data() + j * width;
        unsigned degree = graph[first + j].size();
        memcpy(record, &degree, 4);
        for (unsigned e = 0; e < degree; ++e) {
          const auto &edge = graph[first + j][e];
          unsigned id = ranges.id(edge);
          auto dst = record + 4 + e * (4 + 2 * slots);
          memcpy(dst, &id, 4);
          if (slots) {
            const auto compact = quantize(ranges.intervals(edge));
            for (U k = 0; k < compact.size(); ++k) {
              dst[4 + 2 * k] = compact[k].first;
              dst[5 + 2 * k] = compact[k].second;
            }
          }
        }
      }
      topology.write(reinterpret_cast<char *>(page.data()), PAGE);
    }
    topology.close();
    if (!metadata || !topology)
      throw std::runtime_error("canonical graph output failed");
  }
  std::ofstream report(dir + "/graph-audit.json");
  report << std::setprecision(12) << "{\"status\":\""
         << (std::getenv("HVS_SOURCE_ONLY") ? "SOURCE_COMPLETE" : "PASS")
         << "\",\"nodes\":" << n << ",\"edges\":" << edges
         << ",\"float_pairs\":" << pairs << ",\"full_slots\":" << M
         << ",\"quantized_lost_grid_points\":" << quantized_lost_points
         << ",\"old_truncation_shrunk_edges\":" << truncation_shrunk_edges
         << ",\"old_truncation_lost_width_pair_sum\":" << truncation_lost_width
         << ",\"quantization\":\"outward floor/ceil, overlapping intervals "
            "united\"}\n";
  report.close();
  durable(dir + "/graph-audit.json");
}

struct DualRanges {
  unsigned id(const pipeann::DEGNeighbor &edge) const { return edge.id_; }
  const Intervals &intervals(const pipeann::DEGNeighbor &edge) const {
    return edge.available_range;
  }
};
struct SingleRanges {
  unsigned id(unsigned edge) const { return edge; }
  Intervals intervals(unsigned) const {
    return {};
  }
};

int main(int argc, char **argv) {
  try {
    if (argc != 9)
      throw std::runtime_error(
          "dual|single base1 base2 n d1 d2 threads output-dir");
    const unsigned n = std::stoul(argv[4]), d1 = std::stoul(argv[5]),
                   d2 = std::stoul(argv[6]);
    unsigned threads = std::stoul(argv[7]);
    if (!threads || threads > 64)
      throw std::runtime_error("thread cap");
    omp_set_num_threads(threads);
    pipeann::Parameters p;
    p.Set<unsigned>("L", 200);
    p.Set<unsigned>("R", 40);
    p.Set<unsigned>("C", 750);
    p.Set<float>("alpha", 1.2f);
    p.Set<bool>("saturate_graph", true);
    p.Set<unsigned>("num_threads", threads);
    p.Set<std::string>("save_path", std::string(argv[8]) + "/build");
    if (std::string(argv[1]) == "dual" ||
        std::string(argv[1]) == "dual-prefix") {
      pipeann::HybridIndex<float> index(pipeann::Metric::L2, d1, d2, n, false,
                                        false);
      if (std::string(argv[1]) == "dual-prefix") {
        // Cost probes read a deterministic prefix without writing multi-GB
        // copies. The full run still uses the normal build() entry point.
        auto load_prefix = [n](const char *path, float *data,
                               unsigned dimension, unsigned aligned) {
          std::ifstream in(path, std::ios::binary);
          unsigned count = 0, dim = 0;
          in.read(reinterpret_cast<char *>(&count), 4);
          in.read(reinterpret_cast<char *>(&dim), 4);
          if (count < n || dim != dimension)
            throw std::runtime_error("prefix source identity");
          for (unsigned i = 0; i < n; ++i)
            in.read(reinterpret_cast<char *>(data + U(i) * aligned), dim * 4);
          if (!in)
            throw std::runtime_error("prefix source truncated");
        };
        load_prefix(argv[2], index._data1, d1, ROUND_UP(d1, 8));
        load_prefix(argv[3], index._data2, d2, ROUND_UP(d2, 8));
        index._nd = n;
        index.link(p);
      } else {
        index.build(argv[2], argv[3], n, p);
      }
      save_graph(index._final_graph, DualRanges{}, d1, d2, index._init_ids,
                 argv[8]);
    } else if (std::string(argv[1]) == "single") {
      pipeann::Index<float> index(pipeann::Metric::L2, d1 + d2, n, false,
                                  false);
      index.build(argv[2], n, p);
      save_graph(index._final_graph, SingleRanges{}, d1, d2, { index._ep },
                 argv[8]);
    } else
      throw std::runtime_error("graph target");
  }
  catch (const std::exception &error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
