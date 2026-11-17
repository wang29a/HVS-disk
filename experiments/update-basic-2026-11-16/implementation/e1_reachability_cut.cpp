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
static void check(bool b, const char *s) {
  if (!b)
    throw std::runtime_error(s);
}
static uint32_t u32(const unsigned char *p) {
  uint32_t v;
  std::memcpy(&v, p, 4);
  return v;
}
struct Graph {
  uint32_t n;
  std::vector<uint32_t> entries;
  int fd;
  size_t bytes;
  const unsigned char *data;
  Graph(const char *meta, const char *top) {
    std::ifstream f(meta, std::ios::binary);
    std::vector<unsigned char> b(8192);
    f.read(reinterpret_cast<char *>(b.data()), b.size());
    check(bool(f), "metadata");
    n = u32(b.data());
    check(u32(b.data() + 12) == 40 && u32(b.data() + 16) == 0,
          "compact metadata");
    unsigned count = u32(b.data() + 28);
    for (unsigned j = 0; j < count; ++j)
      entries.push_back(u32(b.data() + 32 + 4 * j));
    fd = open(top, O_RDONLY);
    struct stat s;
    check(fd >= 0 && !fstat(fd, &s), "open topology");
    bytes = s.st_size;
    check(bytes == uint64_t((n + 48) / 49) * 8192, "topology size");
    void *p = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    check(p != MAP_FAILED, "mmap");
    data = static_cast<unsigned char *>(p);
  }
  ~Graph() {
    munmap(const_cast<unsigned char *>(data), bytes);
    close(fd);
  }
  const unsigned char *row(uint32_t id) const {
    check(id < n, "row ID");
    return data + uint64_t(id / 49) * 8192 + (id % 49) * 164;
  }
  std::vector<uint32_t> neighbors(uint32_t id) const {
    auto p = row(id);
    unsigned d = u32(p);
    check(d <= 40, "degree");
    std::vector<uint32_t> v;
    for (unsigned j = 0; j < d; ++j) {
      auto nb = u32(p + 4 + 4 * j);
      check(nb < n && nb != id, "edge ID");
      v.push_back(nb);
    }
    return v;
  }
  std::vector<uint32_t> bfs(std::vector<uint32_t> &parent) const {
    parent.assign(n, UINT32_MAX);
    std::vector<uint32_t> q;
    q.reserve(n);
    for (auto id : entries) {
      check(id < n, "entry ID");
      if (parent[id] == UINT32_MAX) {
        parent[id] = id;
        q.push_back(id);
      }
    }
    for (size_t at = 0; at < q.size(); ++at)
      for (auto v : neighbors(q[at]))
        if (parent[v] == UINT32_MAX) {
          parent[v] = q[at];
          q.push_back(v);
        }
    return q;
  }
};
static void array(std::ostream &o, const std::vector<uint32_t> &v) {
  o << '[';
  for (size_t j = 0; j < v.size(); ++j) {
    if (j)
      o << ',';
    o << v[j];
  }
  o << ']';
}
int main(int argc, char **argv) {
  try {
    check(argc == 6,
          "before.meta before.topo after.meta after.topo output.json");
    Graph before(argv[1], argv[2]), after(argv[3], argv[4]);
    std::vector<uint32_t> old_parent, new_parent;
    before.bfs(old_parent);
    after.bfs(new_parent);
    std::vector<uint32_t> targets;
    for (unsigned id = 0; id < before.n; ++id)
      if (old_parent[id] != UINT32_MAX && new_parent[id] == UINT32_MAX)
        targets.push_back(id);
    std::vector<int> slot(after.n, -1);
    for (unsigned j = 0; j < targets.size(); ++j)
      slot[targets[j]] = j;
    std::vector<std::vector<uint32_t> > old_in(targets.size()),
        new_in(targets.size());
    for (unsigned id = 0; id < before.n; ++id)
      for (auto v : before.neighbors(id))
        if (slot[v] >= 0)
          old_in[slot[v]].push_back(id);
    for (unsigned id = 0; id < after.n; ++id)
      for (auto v : after.neighbors(id))
        if (slot[v] >= 0)
          new_in[slot[v]].push_back(id);
    std::ofstream out(argv[5]);
    out << "{\"old_entries\":";
    array(out, before.entries);
    out << ",\"new_entries\":";
    array(out, after.entries);
    out << ",\"new_unreachable\":" << targets.size() << ",\"nodes\":[";
    unsigned zero_after = 0;
    for (unsigned j = 0; j < targets.size(); ++j) {
      if (j)
        out << ',';
      auto id = targets[j];
      std::vector<uint32_t> removed;
      std::set_difference(old_in[j].begin(), old_in[j].end(), new_in[j].begin(),
                          new_in[j].end(), std::back_inserter(removed));
      std::vector<uint32_t> path;
      for (auto p = id;; p = old_parent[p]) {
        path.push_back(p);
        if (p == old_parent[p])
          break;
        check(path.size() <= before.n, "parent cycle");
      }
      std::reverse(path.begin(), path.end());
      out << "{\"id\":" << id << ",\"old_in\":";
      array(out, old_in[j]);
      out << ",\"new_in\":";
      array(out, new_in[j]);
      out << ",\"old_out\":";
      array(out, before.neighbors(id));
      out << ",\"new_out\":";
      array(out, after.neighbors(id));
      out << ",\"removed_in_sources\":";
      array(out, removed);
      out << ",\"old_reachability_path\":";
      array(out, path);
      out << ",\"removed_sources\":[";
      for (unsigned k = 0; k < removed.size(); ++k) {
        if (k)
          out << ',';
        auto source = removed[k];
        out << "{\"source\":" << source << ",\"source_still_reachable\":"
            << (new_parent[source] != UINT32_MAX ? "true" : "false")
            << ",\"before\":";
        array(out, before.neighbors(source));
        out << ",\"after\":";
        array(out, after.neighbors(source));
        out << '}';
      }
      out << "]}";
      zero_after += new_in[j].empty();
    }
    out << "],\"new_unreachable_zero_in_degree\":" << zero_after << "}\n";
    check(bool(out), "output");
    std::cout << "NEW_UNREACHABLE=" << targets.size()
              << " ZERO_IN=" << zero_after
              << " ENTRY_SETS_EQUAL=" << (before.entries == after.entries)
              << std::endl;
  }
  catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
