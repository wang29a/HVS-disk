#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
using U = uint64_t;
using Range = std::pair<float, float>;
template <class T> T get(std::istream &f) {
  T v;
  if (!f.read(reinterpret_cast<char *>(&v), sizeof(v)))
    throw std::runtime_error("truncated graph");
  return v;
}
std::vector<Range> merge(std::vector<Range> x) {
  std::sort(x.begin(), x.end());
  std::vector<Range> out;
  for (auto p : x) {
    if (!out.empty() && p.first <= out.back().second)
      out.back().second = std::max(out.back().second, p.second);
    else
      out.push_back(p);
  }
  return out;
}
double width(const std::vector<Range> &x) {
  double v = 0;
  for (auto p : x)
    v += double(p.second) - p.first;
  return v;
}
double overlap(const std::vector<Range> &a, const std::vector<Range> &b) {
  double v = 0;
  size_t i = 0, j = 0;
  while (i < a.size() && j < b.size()) {
    v += std::max(0., double(std::min(a[i].second, b[j].second)) -
                          std::max(a[i].first, b[j].first));
    if (a[i].second < b[j].second)
      ++i;
    else
      ++j;
  }
  return v;
}
bool active(const std::vector<Range> &x, float a, bool quant) {
  for (auto r : x) {
    if (!quant) {
      if (a >= r.first && a <= r.second)
        return true;
    } else {
      int lo = std::floor(double(r.first) * 100),
          hi = std::ceil(double(r.second) * 100);
      if (!lo && !hi)
        hi = 1;
      float point = a * 100.f;
      if (point >= lo && point <= hi)
        return true;
    }
  }
  return false;
}
struct Edge {
  unsigned id;
  std::vector<Range> ranges;
};
struct Reader {
  std::ifstream f;
  U n, physical_edges = 0;
  unsigned d1, d2;
  std::array<U, 5> float_active{}, quant_active{};
  explicit Reader(const char *p) : f(p, std::ios::binary) {
    n = get<U>(f);
    d1 = get<unsigned>(f);
    d2 = get<unsigned>(f);
  }
  std::vector<Edge> node(U id, bool filter, float alpha) {
    unsigned count = get<unsigned>(f);
    if (count > 40)
      throw std::runtime_error("degree >40");
    physical_edges += count;
    std::vector<Edge> out;
    for (unsigned k = 0; k < count; k++) {
      Edge e;
      e.id = get<unsigned>(f);
      unsigned nr = get<unsigned>(f);
      if (e.id >= n || e.id == id || nr > 64)
        throw std::runtime_error("edge bounds");
      for (unsigned j = 0; j < nr; j++) {
        float lo = get<float>(f), hi = get<float>(f);
        if (!std::isfinite(lo) || !std::isfinite(hi) || lo < 0 || hi > 1 ||
            lo > hi)
          throw std::runtime_error("interval bounds");
        e.ranges.emplace_back(lo, hi);
      }
      e.ranges = merge(e.ranges);
      const float weights[5] = { 0, .1f, .5f, .9f, 1 };
      for (unsigned j = 0; j < 5; j++) {
        float_active[j] += active(e.ranges, weights[j], false);
        quant_active[j] += active(e.ranges, weights[j], true);
      }
      if (!filter || active(e.ranges, alpha, true))
        out.push_back(e);
    }
    std::sort(out.begin(), out.end(),
              [](const Edge &a, const Edge &b) { return a.id < b.id; });
    for (size_t k = 1; k < out.size(); k++)
      if (out[k - 1].id == out[k].id)
        throw std::runtime_error("duplicate edge ID");
    return out;
  }
};
int main(int argc, char **argv) {
  try {
    bool filtered = argc == 6 && std::string(argv[1]) == "--active";
    if (argc != 4 && !filtered)
      throw std::runtime_error("[--active] old new output [alpha]");
    int offset = filtered ? 1 : 0;
    Reader a(argv[1 + offset]), b(argv[2 + offset]);
    if (a.n != b.n || a.d1 != b.d1 || a.d2 != b.d2)
      throw std::runtime_error("graph shape mismatch");
    float alpha = filtered ? std::stof(argv[5]) : 0;
    U old_edges = 0, new_edges = 0, shared = 0, added = 0, removed = 0,
      changed_nodes = 0, changed_ranges = 0, changed_range_nodes = 0;
    double old_width = 0, new_width = 0, common_old_width = 0,
           common_new_width = 0, common_removed_width = 0,
           common_added_width = 0;
    for (U node = 0; node < a.n; node++) {
      auto x = a.node(node, filtered, alpha), y = b.node(node, filtered, alpha);
      old_edges += x.size();
      new_edges += y.size();
      for (auto &e : x)
        old_width += width(e.ranges);
      for (auto &e : y)
        new_width += width(e.ranges);
      size_t i = 0, j = 0;
      bool set_changed = false, range_changed = false;
      while (i < x.size() || j < y.size()) {
        if (j == y.size() || (i < x.size() && x[i].id < y[j].id)) {
          ++removed;
          ++i;
          set_changed = true;
        } else if (i == x.size() || y[j].id < x[i].id) {
          ++added;
          ++j;
          set_changed = true;
        } else {
          ++shared;
          double wa = width(x[i].ranges), wb = width(y[j].ranges),
                 both = overlap(x[i].ranges, y[j].ranges);
          common_old_width += wa;
          common_new_width += wb;
          common_removed_width += std::max(0., wa - both);
          common_added_width += std::max(0., wb - both);
          if (x[i].ranges != y[j].ranges) {
            ++changed_ranges;
            range_changed = true;
          }
          ++i;
          ++j;
        }
      }
      changed_nodes += set_changed;
      changed_range_nodes += range_changed;
    }
    if (a.f.peek() != std::char_traits<char>::eof() ||
        b.f.peek() != std::char_traits<char>::eof())
      throw std::runtime_error("trailing graph bytes");
    std::ofstream out(argv[3 + offset]);
    out << std::setprecision(16)
        << "{\"status\":\"PASS\",\"scope\":\"canonical interval unions; ID "
           "sets ignore adjacency ordering\",\"filtered_quantized_alpha\":";
    if (filtered)
      out << alpha;
    else
      out << "null";
    out << ",\"nodes\":" << a.n << ",\"old_edges\":" << old_edges
        << ",\"new_edges\":" << new_edges << ",\"shared_edges\":" << shared
        << ",\"added_edges\":" << added << ",\"removed_edges\":" << removed
        << ",\"edge_symmetric_difference_over_union\":"
        << double(added + removed) / (shared + added + removed)
        << ",\"old_mean_degree\":" << double(old_edges) / a.n
        << ",\"new_mean_degree\":" << double(new_edges) / a.n
        << ",\"nodes_with_edge_set_changes\":" << changed_nodes
        << ",\"common_edges_with_range_changes\":" << changed_ranges
        << ",\"nodes_with_common_range_changes\":" << changed_range_nodes
        << ",\"old_total_width\":" << old_width
        << ",\"new_total_width\":" << new_width
        << ",\"common_old_width\":" << common_old_width
        << ",\"common_new_width\":" << common_new_width
        << ",\"common_removed_width\":" << common_removed_width
        << ",\"common_added_width\":" << common_added_width
        << ",\"old_physical_edges\":" << a.physical_edges
        << ",\"new_physical_edges\":" << b.physical_edges << ",\"activity\":[";
    const float weights[5] = { 0, .1f, .5f, .9f, 1 };
    for (unsigned k = 0; k < 5; k++) {
      if (k)
        out << ",";
      out << "{\"alpha\":" << weights[k]
          << ",\"old_float_edges\":" << a.float_active[k]
          << ",\"new_float_edges\":" << b.float_active[k]
          << ",\"old_quantized_edges\":" << a.quant_active[k]
          << ",\"new_quantized_edges\":" << b.quant_active[k] << "}";
    }
    out << "]}\n";
    if (!out)
      throw std::runtime_error("output failed");
  }
  catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
  return 0;
}
