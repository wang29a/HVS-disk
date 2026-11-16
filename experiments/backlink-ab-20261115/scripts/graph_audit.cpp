#include "function_io.h"
#include <array>
#include <iostream>
#include <iomanip>
#include <set>
int main(int argc, char **argv) {
  try {
    if (argc != 2 && argc != 5)
      throw std::runtime_error("raw-graph [compact-meta compact-topo sidecar]");
    std::ifstream raw(argv[1], std::ios::binary);
    auto n = fnio::get<uint64_t>(raw);
    auto d1 = fnio::get<unsigned>(raw), d2 = fnio::get<unsigned>(raw);
    uint64_t edges = 0, pairs = 0;
    unsigned maxm = 0, maxquant = 0, maxdegree = 0;
    double width = 0;
    std::array<uint64_t, 41> hist{};
    std::array<uint64_t, 5> active{}, qactive{};
    std::array<float, 5> alpha{ { .1f, .3f, .5f, .7f, .9f } };
    bool verify = argc == 5;
    std::ifstream topo, side;
    unsigned M = 0;
    std::array<uint8_t, 8192> page{};
    if (verify) {
      std::ifstream meta(argv[2], std::ios::binary);
      std::array<uint8_t, 8192> b{};
      if (!meta.read(reinterpret_cast<char *>(b.data()), 8192))
        throw std::runtime_error("compact meta");
      unsigned nn, dd1, dd2, R, zero;
      uint64_t npp;
      std::memcpy(&nn, b.data(), 4);
      std::memcpy(&dd1, b.data() + 4, 4);
      std::memcpy(&dd2, b.data() + 8, 4);
      std::memcpy(&R, b.data() + 12, 4);
      std::memcpy(&zero, b.data() + 16, 4);
      std::memcpy(&npp, b.data() + 20, 8);
      if (nn != n || dd1 != d1 || dd2 != d2 || R != 40 || zero || npp != 49)
        throw std::runtime_error("compact identity");
      topo.open(argv[3], std::ios::binary);
      side.open(argv[4], std::ios::binary);
      side.seekg(0, std::ios::end);
      auto size = side.tellg();
      if (size <= 0 || uint64_t(size) % (n * 80))
        throw std::runtime_error("side shape");
      M = uint64_t(size) / (n * 80);
      side.seekg(0);
    }
    for (uint64_t u = 0; u < n; ++u) {
      auto deg = fnio::get<unsigned>(raw);
      if (deg > 40)
        throw std::runtime_error("degree exceeds40");
      ++hist[deg];
      maxdegree = std::max(maxdegree, deg);
      edges += deg;
      std::set<unsigned> seen;
      std::vector<uint8_t> expected(40 * 2 * M), got(expected.size());
      if (verify) {
        if (u % 49 == 0 &&
            !topo.read(reinterpret_cast<char *>(page.data()), 8192))
          throw std::runtime_error("short compact page");
        unsigned count;
        std::memcpy(&count, page.data() + (u % 49) * 164, 4);
        if (count != deg)
          throw std::runtime_error("export degree");
      }
      for (unsigned j = 0; j < deg; ++j) {
        unsigned id = fnio::get<unsigned>(raw), nr = fnio::get<unsigned>(raw);
        if (id >= n || id == u || !seen.insert(id).second || !nr || nr > 4096)
          throw std::runtime_error("edge structure");
        maxm = std::max(maxm, nr);
        pairs += nr;
        if (verify) {
          unsigned nb;
          std::memcpy(&nb, page.data() + (u % 49) * 164 + 4 + j * 4, 4);
          if (nb != id)
            throw std::runtime_error("export order/id");
        }
        std::vector<std::pair<int, int> > q;
        std::array<bool, 5> aa{}, qa{};
        float last = -1;
        for (unsigned k = 0; k < nr; ++k) {
          float lo = fnio::get<float>(raw), hi = fnio::get<float>(raw);
          if (!std::isfinite(lo) || !std::isfinite(hi) || lo < 0 || hi > 1 ||
              lo >= hi || lo <= last)
            throw std::runtime_error("float interval structure");
          last = hi;
          width += double(hi) - lo;
          for (unsigned z = 0; z < 5; ++z)
            aa[z] = aa[z] || (alpha[z] >= lo && alpha[z] <= hi);
          int a = std::floor(double(lo) * 100), b = std::ceil(double(hi) * 100);
          if (a == 0 && b == 0)
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
        maxquant = std::max(maxquant, unsigned(merged.size()));
        for (unsigned k = 0; k < merged.size(); ++k) {
          auto r = merged[k];
          for (unsigned z = 0; z < 5; ++z)
            qa[z] = qa[z] || (alpha[z] * 100.f >= r.first &&
                              alpha[z] * 100.f <= r.second);
          if (verify) {
            if (k >= M)
              throw std::runtime_error("export range capacity");
            expected[(j * M + k) * 2] = r.first;
            expected[(j * M + k) * 2 + 1] = r.second;
          }
        }
        for (unsigned z = 0; z < 5; ++z) {
          active[z] += aa[z];
          qactive[z] += qa[z];
        }
      }
      if (verify) {
        if (!side.read(reinterpret_cast<char *>(got.data()), got.size()) ||
            got != expected)
          throw std::runtime_error("side bytes differ");
        for (unsigned j = deg; j < 40; ++j) {
          unsigned nb;
          std::memcpy(&nb, page.data() + (u % 49) * 164 + 4 + j * 4, 4);
          if (nb)
            throw std::runtime_error("unused id slot");
        }
        if (u % 49 == 48 || u == n - 1) {
          unsigned first = (u % 49 + 1) * 164;
          for (unsigned off = first; off < 8192; ++off)
            if (page[off])
              throw std::runtime_error("page padding");
        }
      }
    }
    if (raw.peek() != EOF ||
        (verify && (topo.peek() != EOF || side.peek() != EOF)))
      throw std::runtime_error("trailing graph/export bytes");
    std::cout << std::setprecision(17) << "{\"status\":\"PASS\",\"nodes\":" << n
              << ",\"d1\":" << d1 << ",\"d2\":" << d2 << ",\"edges\":" << edges
              << ",\"mean_degree\":" << double(edges) / n
              << ",\"max_degree\":" << maxdegree << ",\"float_pairs\":" << pairs
              << ",\"mean_active_width\":" << width / edges
              << ",\"max_float_slots\":" << maxm
              << ",\"max_quantized_slots\":" << maxquant
              << ",\"export_checked\":" << (verify ? "true" : "false")
              << ",\"degree_histogram\":[";
    for (unsigned j = 0; j <= 40; ++j) {
      if (j)
        std::cout << ",";
      std::cout << hist[j];
    }
    std::cout << "],\"active_fractions\":[";
    for (unsigned z = 0; z < 5; ++z) {
      if (z)
        std::cout << ",";
      std::cout << double(active[z]) / edges;
    }
    std::cout << "],\"quantized_active_fractions\":[";
    for (unsigned z = 0; z < 5; ++z) {
      if (z)
        std::cout << ",";
      std::cout << double(qactive[z]) / edges;
    }
    std::cout << "]}\n";
  }
  catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
