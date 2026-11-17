#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

struct Mapping {
  int fd;
  size_t size;
  const unsigned char *data;
  explicit Mapping(const char *path) {
    fd = open(path, O_RDONLY);
    if (fd < 0)
      throw std::runtime_error("open topology");
    struct stat s;
    if (fstat(fd, &s) || s.st_size <= 0)
      throw std::runtime_error("stat topology");
    size = s.st_size;
    auto p = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED)
      throw std::runtime_error("mmap topology");
    data = static_cast<const unsigned char *>(p);
  }
  ~Mapping() {
    munmap(const_cast<unsigned char *>(data), size);
    close(fd);
  }
};
template <class T> T get(std::istream &in) {
  T v;
  if (!in.read(reinterpret_cast<char *>(&v), sizeof(v)))
    throw std::runtime_error("metadata truncated");
  return v;
}
int main(int argc, char **argv) {
  try {
    if (argc != 4)
      throw std::runtime_error(
          "connectivity_audit entry.meta compact.topo output-prefix");
    std::ifstream meta(argv[1], std::ios::binary);
    auto n = get<uint32_t>(meta), dx = get<uint32_t>(meta),
         dy = get<uint32_t>(meta);
    auto R = get<uint32_t>(meta), slots = get<uint32_t>(meta);
    get<uint64_t>(meta);
    auto ne = get<uint32_t>(meta);
    if (!n || R != 40 || !ne || ne > n)
      throw std::runtime_error("unsupported metadata");
    std::vector<uint32_t> entries(ne);
    for (auto &v : entries) {
      v = get<uint32_t>(meta);
      if (v >= n)
        throw std::runtime_error("entry bounds");
    }
    Mapping file(argv[2]);
    const unsigned record = 4 + R * 4, per = 8192 / record;
    if (file.size != uint64_t((uint64_t(n) + per - 1) / per) * 8192)
      throw std::runtime_error("compact size mismatch");
    auto row = [&](uint32_t id) {
      return reinterpret_cast<const uint32_t *>(
          file.data + uint64_t(id / per) * 8192 + id % per * record);
    };
    std::vector<uint32_t> incoming(n, 0);
    uint64_t edges = 0, low = 0;
    for (uint32_t id = 0; id < n; ++id) {
      auto p = row(id);
      if (p[0] > R)
        throw std::runtime_error("degree bounds");
      edges += p[0];
      low += p[0] <= 2;
      for (unsigned j = 1; j <= p[0]; ++j) {
        if (p[j] >= n)
          throw std::runtime_error("neighbor bounds");
        ++incoming[p[j]];
      }
    }
    std::vector<uint8_t> seen(n, 0);
    std::vector<uint32_t> queue;
    queue.reserve(n);
    for (auto id : entries)
      if (!seen[id]) {
        seen[id] = 1;
        queue.push_back(id);
      }
    for (uint64_t at = 0; at < queue.size(); ++at) {
      auto p = row(queue[at]);
      for (unsigned j = 1; j <= p[0]; ++j)
        if (!seen[p[j]]) {
          seen[p[j]] = 1;
          queue.push_back(p[j]);
        }
    }
    std::string prefix(argv[3]);
    std::ofstream low_file(prefix + "-low-degree.csv"),
        unreachable(prefix + "-unreachable.txt");
    low_file << "id,out_degree,in_degree,unreachable\n";
    uint64_t zero_in = 0, unreachable_low = 0;
    for (uint32_t id = 0; id < n; ++id) {
      zero_in += incoming[id] == 0;
      if (!seen[id])
        unreachable << id << '\n';
      if (row(id)[0] <= 2) {
        low_file << id << ',' << row(id)[0] << ',' << incoming[id] << ','
                 << unsigned(!seen[id]) << '\n';
        unreachable_low += !seen[id];
      }
    }
    std::cout << "{\"nodes\":" << n << ",\"dims\":[" << dx << ',' << dy
              << "],\"entries\":" << ne
              << ",\"source_meta_alpha_slots\":" << slots
              << ",\"topology_format\":\"compact IDs only\",\"edges\":" << edges
              << ",\"reachable\":" << queue.size()
              << ",\"unreachable\":" << n - queue.size()
              << ",\"out_degree_le_2\":" << low
              << ",\"unreachable_out_degree_le_2\":" << unreachable_low
              << ",\"zero_in_degree\":" << zero_in << "}\n";
    return 0;
  }
  catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
