// Streaming source-to-Full writer and independent coverage/layout verifier.
// No alternative format, density experiment or graph reordering is produced.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#include <vector>

constexpr uint64_t PAGE = 8192;
struct Map {
  int fd;
  uint64_t size;
  const uint8_t *p;
  Map(const std::string &path) {
    fd = open(path.c_str(), O_RDONLY);
    struct stat s {};
    if (fd < 0 || fstat(fd, &s))
      throw std::runtime_error("open map " + path);
    size = s.st_size;
    p = static_cast<const uint8_t *>(
        mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (p == MAP_FAILED)
      throw std::runtime_error("mmap");
  }
  ~Map() {
    munmap(const_cast<uint8_t *>(p), size);
    close(fd);
  }
};
template <class T> T get(const uint8_t *p) {
  T v;
  memcpy(&v, p, sizeof(v));
  return v;
}
template <class T> T read(std::ifstream &in) {
  T value;
  in.read(reinterpret_cast<char *>(&value), sizeof(value));
  if (!in)
    throw std::runtime_error("truncated float source");
  return value;
}
using Pair = std::pair<float, float>;
using Code = std::pair<uint8_t, uint8_t>;
std::vector<Code> encode(const std::vector<Pair> &source) {
  std::vector<Code> values;
  for (auto p : source) {
    if (!std::isfinite(p.first) || !std::isfinite(p.second) || p.first < 0 ||
        p.second > 1 || p.first > p.second)
      throw std::runtime_error("invalid source range");
    int a = std::floor(double(p.first) * 100),
        b = std::ceil(double(p.second) * 100);
    if (!a && !b)
      b = 1;
    values.emplace_back(a, b);
  }
  std::sort(values.begin(), values.end());
  std::vector<Code> merged;
  for (auto p : values) {
    if (!merged.empty() && p.first <= merged.back().second)
      merged.back().second = std::max(merged.back().second, p.second);
    else
      merged.push_back(p);
  }
  return merged;
}
void zero(const uint8_t *first, const uint8_t *last) {
  if (std::any_of(first, last, [](uint8_t v) { return v != 0; }))
    throw std::runtime_error("nonzero padding");
}
int main(int argc, char **argv) {
  try {
    if (argc != 6)
      throw std::runtime_error("write|check float-graph meta topo report");
    const bool writing = std::string(argv[1]) == "write";
    if (!writing && std::string(argv[1]) != "check")
      throw std::runtime_error("command");
    Map meta(argv[3]);
    if (meta.size != PAGE)
      throw std::runtime_error("meta size");
    uint32_t n = get<uint32_t>(meta.p), d1 = get<uint32_t>(meta.p + 4),
             d2 = get<uint32_t>(meta.p + 8);
    uint32_t R = get<uint32_t>(meta.p + 12), M = get<uint32_t>(meta.p + 16),
             ne = get<uint32_t>(meta.p + 28);
    uint64_t npp = get<uint64_t>(meta.p + 20);
    if (!n || R != 40 || !M || M > 64 || !npp || !ne ||
        32 + uint64_t(ne) * 4 > PAGE)
      throw std::runtime_error("meta fields");
    uint64_t width = 4 + uint64_t(R) * (4 + 2 * M),
             bytes = ((n + npp - 1) / npp) * PAGE;
    if (npp != PAGE / width)
      throw std::runtime_error("page density");
    std::vector<uint32_t> entries;
    for (uint32_t i = 0; i < ne; ++i) {
      auto id = get<uint32_t>(meta.p + 32 + i * 4);
      if (id >= n)
        throw std::runtime_error("entry ID");
      entries.push_back(id);
    }
    auto sorted = entries;
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
      throw std::runtime_error("duplicate entries");
    std::ifstream input(argv[2], std::ios::binary);
    if (read<uint64_t>(input) != n || read<uint32_t>(input) != d1 ||
        read<uint32_t>(input) != d2)
      throw std::runtime_error("source header");
    std::ofstream out;
    std::string temp = std::string(argv[4]) + ".partial";
    if (writing) {
      std::string path = argv[4];
      auto slash = path.find_last_of('/');
      std::string folder =
          slash == std::string::npos ? "." : path.substr(0, slash);
      struct statvfs space {};
      if (statvfs(folder.c_str(), &space) ||
          uint64_t(space.f_bavail) * space.f_frsize < bytes + (2ULL << 30))
        throw std::runtime_error(
            "topology needs output bytes plus 2 GiB reserve");
      out.open(temp, std::ios::binary | std::ios::trunc);
      if (!out)
        throw std::runtime_error("open destination");
    }
    std::unique_ptr<Map> topo;
    if (!writing) {
      topo.reset(new Map(argv[4]));
      if (topo->size != bytes)
        throw std::runtime_error("topology size");
    }
    uint64_t edges = 0, pairs = 0;
    uint32_t max_degree = 0, max_slots = 0;
    std::array<uint8_t, PAGE> page{};
    std::vector<Pair> ranges;
    for (uint64_t i = 0; i < n; ++i) {
      if (i % npp == 0)
        page.fill(0);
      auto target = page.data() + (i % npp) * width;
      const uint8_t *actual =
          writing ? nullptr : topo->p + (i / npp) * PAGE + (i % npp) * width;
      auto degree = read<uint32_t>(input);
      max_degree = std::max(max_degree, degree);
      if (degree > R || (!writing && get<uint32_t>(actual) != degree))
        throw std::runtime_error("degree");
      memcpy(target, &degree, 4);
      std::vector<uint32_t> ids;
      for (uint32_t e = 0; e < degree; ++e) {
        auto id = read<uint32_t>(input), count = read<uint32_t>(input);
        if (id >= n || id == i || !count || count > 64)
          throw std::runtime_error("source edge");
        ids.push_back(id);
        ranges.clear();
        for (uint32_t k = 0; k < count; ++k) {
          float a = read<float>(input), b = read<float>(input);
          ranges.emplace_back(a, b);
        }
        auto encoded = encode(ranges);
        max_slots = std::max(max_slots, uint32_t(encoded.size()));
        if (encoded.size() > M)
          throw std::runtime_error("slot truncation");
        auto dst = target + 4 + e * (4 + 2 * M);
        memcpy(dst, &id, 4);
        for (uint64_t k = 0; k < encoded.size(); ++k) {
          dst[4 + 2 * k] = encoded[k].first;
          dst[5 + 2 * k] = encoded[k].second;
        }
        for (auto p : ranges) {
          bool contained = false;
          for (auto q : encoded)
            contained |=
                q.first / 100. <= p.first && q.second / 100. >= p.second;
          if (!contained)
            throw std::runtime_error("narrowed range");
        }
        if (!writing && memcmp(dst, actual + 4 + e * (4 + 2 * M), 4 + 2 * M))
          throw std::runtime_error("edge/order/range identity");
        ++edges;
        pairs += count;
      }
      std::sort(ids.begin(), ids.end());
      if (std::adjacent_find(ids.begin(), ids.end()) != ids.end())
        throw std::runtime_error("duplicate neighbor");
      if (!writing)
        zero(actual + 4 + degree * (4 + 2 * M), actual + width);
      if (i % npp == npp - 1 || i + 1 == n) {
        if (writing)
          out.write(reinterpret_cast<const char *>(page.data()), PAGE);
        else
          zero(topo->p + (i / npp) * PAGE + ((i % npp) + 1) * width,
               topo->p + (i / npp + 1) * PAGE);
      }
    }
    if (input.peek() != std::char_traits<char>::eof() || max_slots != M)
      throw std::runtime_error("source tail or maximum slots");
    if (writing) {
      out.close();
      if (!out)
        throw std::runtime_error("destination write");
      int fd = open(temp.c_str(), O_RDWR);
      if (fd < 0 || fsync(fd))
        throw std::runtime_error("fsync");
      close(fd);
      if (rename(temp.c_str(), argv[4]))
        throw std::runtime_error("publish topology");
    }
    std::ofstream report(argv[5]);
    report << "{\"status\":\"PASS\",\"nodes\":" << n << ",\"edges\":" << edges
           << ",\"float_pairs\":" << pairs << ",\"full_slots\":" << M
           << ",\"nodes_per_page\":" << npp << ",\"topology_bytes\":" << bytes
           << ",\"max_degree\":" << max_degree
           << ",\"narrowed_ranges\":0,\"id_order_mismatches\":0,\"padding_"
              "mismatches\":0}\n";
  }
  catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
