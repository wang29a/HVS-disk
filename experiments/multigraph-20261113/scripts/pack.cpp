// Read the immutable float checkpoint; write separate compact pages and ranges.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
using U = uint64_t;
template <class T> T get(std::istream &f) {
  T x{};
  if (!f.read((char *)&x, sizeof(x)))
    throw std::runtime_error("truncated");
  return x;
}
template <class T> void put(std::ostream &f, T x) {
  f.write((char *)&x, sizeof(x));
}
void set32(void *p, unsigned x) { memcpy(p, &x, 4); }
int main(int argc, char **argv) {
  try {
    if (argc != 7)
      throw std::runtime_error("source-meta float-graph output-meta "
                               "output-topo sidecar-or-- entry-meta");
    std::ifstream mf(argv[1], std::ios::binary), raw(argv[2], std::ios::binary),
        ep(argv[6], std::ios::binary);
    std::vector<uint8_t> meta(8192);
    if (!mf.read((char *)meta.data(), 8192))
      throw std::runtime_error("meta");
    unsigned n, d1, d2, R, M;
    memcpy(&n, meta.data(), 4);
    memcpy(&d1, meta.data() + 4, 4);
    memcpy(&d2, meta.data() + 8, 4);
    memcpy(&R, meta.data() + 12, 4);
    memcpy(&M, meta.data() + 16, 4);
    if (!n || !R || R > 1000 || M > 64 || get<U>(raw) != n ||
        get<unsigned>(raw) != d1 || get<unsigned>(raw) != d2)
      throw std::runtime_error("source identity");
    std::vector<uint8_t> entry(8192);
    if (!ep.read((char *)entry.data(), 8192))
      throw std::runtime_error("entry meta");
    unsigned en;
    memcpy(&en, entry.data() + 28, 4);
    if (!en || en > 1000)
      throw std::runtime_error("entry count");
    for (unsigned j = 0; j < 3; j++)
      if (memcmp(meta.data() + 4 * j, entry.data() + 4 * j, 4))
        throw std::runtime_error("entry dimensions");
    memcpy(meta.data() + 28, entry.data() + 28, 4 + 4 * en);
    unsigned width = 4 + 4 * R;
    U npp = 8192 / width;
    set32(meta.data() + 16, 0);
    memcpy(meta.data() + 20, &npp, 8);
    std::ofstream outm(argv[3], std::ios::binary),
        out(argv[4], std::ios::binary), side;
    bool ranges = std::string(argv[5]) != "-";
    if (ranges) {
      if (!M)
        throw std::runtime_error("sidecar slots");
      side.open(argv[5], std::ios::binary);
    }
    outm.write((char *)meta.data(), 8192);
    std::vector<uint8_t> page(8192), row(U(R) * 2 * M);
    U edges = 0;
    for (unsigned id = 0; id < n; id++) {
      unsigned nn = get<unsigned>(raw);
      if (nn > R)
        throw std::runtime_error("degree");
      auto *p = page.data() + (id % npp) * width;
      set32(p, nn);
      std::fill(row.begin(), row.end(), 0);
      std::vector<unsigned> seen;
      for (unsigned j = 0; j < nn; j++) {
        unsigned nb = get<unsigned>(raw), nr = get<unsigned>(raw);
        if (nb >= n || nb == id ||
            std::find(seen.begin(), seen.end(), nb) != seen.end() || nr > 4096)
          throw std::runtime_error("edge");
        seen.push_back(nb);
        set32(p + 4 + 4 * j, nb);
        ++edges;
        std::vector<std::pair<int, int> > q;
        for (unsigned k = 0; k < nr; k++) {
          float lo = get<float>(raw), hi = get<float>(raw);
          if (!std::isfinite(lo) || !std::isfinite(hi) || lo < 0 || hi > 1 ||
              lo > hi)
            throw std::runtime_error("range");
          int a = std::floor(double(lo) * 100), b = std::ceil(double(hi) * 100);
          if (!a && !b)
            b = 1;
          q.emplace_back(a, b);
        }
        std::sort(q.begin(), q.end());
        std::vector<std::pair<int, int> > merged;
        for (auto r : q) {
          if (!merged.empty() && r.first <= merged.back().second)
            merged.back().second = std::max(merged.back().second, r.second);
          else
            merged.push_back(r);
        }
        if (ranges) {
          if (merged.size() > M)
            throw std::runtime_error("range slots");
          for (unsigned k = 0; k < merged.size(); k++) {
            row[(j * M + k) * 2] = merged[k].first;
            row[(j * M + k) * 2 + 1] = merged[k].second;
          }
        }
      }
      if (ranges)
        side.write((char *)row.data(), row.size());
      if (id % npp == npp - 1 || id == n - 1) {
        out.write((char *)page.data(), 8192);
        std::fill(page.begin(), page.end(), 0);
      }
    }
    if (raw.peek() != EOF || !outm || !out || (ranges && !side))
      throw std::runtime_error("output/input size");
    std::cout << "{\"nodes\":" << n << ",\"edges\":" << edges
              << ",\"mean_degree\":" << double(edges) / n
              << ",\"record_bytes\":" << width << ",\"nodes_per_page\":" << npp
              << ",\"range_slots\":" << (ranges ? M : 0) << "}\n";
    return 0;
  }
  catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
