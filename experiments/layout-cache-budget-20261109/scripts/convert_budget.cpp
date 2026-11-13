#include "layout_store.h"
#include <iomanip>
using namespace layout_exp;
using Range = std::pair<int, int>;
std::vector<Range> ranges(const uint8_t *edge, U slots) {
  std::vector<Range> out;
  bool empty = false;
  for (U j = 0; j < slots; j++) {
    int a = edge[4 + 2 * j], b = edge[5 + 2 * j];
    if (a == 0 && b == 0) {
      empty = true;
      continue;
    }
    if (empty || a > b || b > 100)
      throw std::runtime_error("invalid input intervals");
    out.emplace_back(a, b);
  }
  std::sort(out.begin(), out.end());
  std::vector<Range> normalized;
  for (auto r : out) {
    if (!normalized.empty() && r.first <= normalized.back().second)
      normalized.back().second = std::max(normalized.back().second, r.second);
    else
      normalized.push_back(r);
  }
  return normalized;
}
std::vector<Range> merge(std::vector<Range> v, U budget) {
  if (!budget)
    return {};
  while (v.size() > budget) {
    size_t best = 0;
    for (size_t j = 1; j + 1 < v.size(); j++)
      if (v[j + 1].first - v[j].second < v[best + 1].first - v[best].second)
        best = j;
    v[best].second = v[best + 1].second;
    v.erase(v.begin() + best + 1);
  }
  return v;
}
int main(int argc, char **argv) {
  try {
    if (argc != 8)
      throw std::runtime_error(
          "meta topo coord output_meta output_topo budget output_json");
    Source src(argv[1], argv[2], argv[3]);
    U budget = std::stoul(argv[6]);
    if (budget > src.M)
      throw std::runtime_error("budget exceeds source");
    U width = 4 + src.R * (4 + 2 * budget), npp = PAGE / width;
    auto start = Clock::now();
    std::array<uint8_t, PAGE> meta{};
    memcpy(meta.data(), src.meta.p, PAGE);
    uint32_t b = budget;
    memcpy(meta.data() + 16, &b, 4);
    memcpy(meta.data() + 20, &npp, 8);
    std::ofstream mo(argv[4], std::ios::binary);
    mo.write((char *)meta.data(), PAGE);
    mo.close();
    std::ofstream out(argv[5], std::ios::binary);
    std::array<uint8_t, PAGE> page{};
    U edges = 0, old_components = 0, new_components = 0, old_width = 0,
      new_width = 0;
    for (U first = 0; first < src.n; first += npp) {
      page.fill(0);
      for (U j = 0; j < npp && first + j < src.n; j++) {
        const uint8_t *p = src.field(first + j, 1);
        auto *d = page.data() + j * width;
        U degree = read32(p);
        if (degree > src.R)
          throw std::runtime_error("degree");
        memcpy(d, p, 4);
        for (U k = 0; k < degree; k++) {
          auto *e = p + 4 + k * (4 + 2 * src.M);
          auto *t = d + 4 + k * (4 + 2 * budget);
          if (read32(e) >= src.n)
            throw std::runtime_error("neighbor ID");
          memcpy(t, e, 4);
          auto old = ranges(e, src.M);
          auto now = merge(old, budget);
          edges++;
          old_components += old.size();
          new_components += now.size();
          for (auto r : old)
            old_width += r.second - r.first;
          if (!budget)
            new_width += 100;
          for (U z = 0; z < now.size(); z++) {
            t[4 + 2 * z] = now[z].first;
            t[5 + 2 * z] = now[z].second;
            new_width += now[z].second - now[z].first;
          }
        }
      }
      out.write((char *)page.data(), PAGE);
    }
    out.close();
    for (int index : { 4, 5 }) {
      int fd = open(argv[index], O_RDWR);
      if (fd < 0 || fsync(fd))
        throw std::runtime_error("flush");
      close(fd);
    }
    Source dst(argv[4], argv[5], argv[3]);
    for (U id = 0; id < src.n; id++) {
      auto *p = src.field(id, 1);
      auto *d = dst.field(id, 1);
      if (read32(p) != read32(d))
        throw std::runtime_error("degree changed");
      for (U k = 0; k < read32(p); k++) {
        auto *e = p + 4 + k * (4 + 2 * src.M);
        auto *t = d + 4 + k * (4 + 2 * budget);
        if (read32(e) != read32(t))
          throw std::runtime_error("neighbor/order changed");
        auto old = ranges(e, src.M), now = ranges(t, budget);
        if (budget)
          for (auto r : old) {
            bool contained = false;
            for (auto q : now)
              contained |= (q.first <= r.first && q.second >= r.second);
            if (!contained)
              throw std::runtime_error("lost original activity");
          }
      }
      U used = 4 + read32(d) * (4 + 2 * budget);
      for (U z = used; z < width; z++)
        if (d[z])
          throw std::runtime_error("unused neighbor not zero");
    }
    for (U p = 0; p < dst.topo.size / PAGE; p++) {
      U used = std::min(npp, src.n - p * npp) * width;
      for (U z = used; z < PAGE; z++)
        if (dst.topo.p[p * PAGE + z])
          throw std::runtime_error("nonzero page padding");
    }
    std::ofstream report(argv[7]);
    report << std::setprecision(12)
           << "{\"status\":\"PASS\",\"budget\":" << budget
           << ",\"nodes\":" << src.n << ",\"edges\":" << edges
           << ",\"source_slots\":" << src.M << ",\"record_bytes\":" << width
           << ",\"nodes_per_page\":" << npp
           << ",\"topology_bytes\":" << dst.topo.size
           << ",\"old_normalized_components\":" << old_components
           << ",\"new_components\":" << new_components
           << ",\"old_coverage_width\":" << old_width
           << ",\"new_coverage_width\":" << new_width
           << ",\"neighbor_ids_and_order_preserved\":true,\"original_activity_"
              "contained\":true,\"seconds\":" << us(start) / 1e6 << "}\n";
    if (!report)
      throw std::runtime_error("report write");
  }
  catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
