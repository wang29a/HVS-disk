#include "entry_frontier.h"
#include <iostream>
#include <random>
#include <stdexcept>
struct Entry {
  unsigned id;
  float x, y;
};
// Literal tuple equivalent of NeighborH ordering and the upstream reverse
// scan, independent of the helper's comparator.
static std::vector<Entry> reference(std::vector<Entry> pool) {
  std::vector<std::pair<std::pair<float, float>, unsigned> > ordered;
  for (auto e : pool)
    ordered.push_back({ { e.y, e.x }, e.id });
  std::sort(ordered.begin(), ordered.end());
  std::vector<Entry> out;
  float maximum = 0;
  for (auto it = ordered.rbegin(); it != ordered.rend(); ++it)
    if (it->first.second > maximum) {
      out.push_back({ it->second, it->first.second, it->first.first });
      maximum = it->first.second;
    }
  return out;
}
static void equal(const std::vector<Entry> &a, const std::vector<Entry> &b) {
  if (a.size() != b.size())
    throw std::runtime_error("entry count mismatch");
  for (unsigned j = 0; j < a.size(); ++j)
    if (a[j].id != b[j].id || a[j].x != b[j].x || a[j].y != b[j].y)
      throw std::runtime_error("entry record/order mismatch");
}
int main() {
  try {
    auto far = hvs_update::builder_entry_frontier(
        std::vector<Entry>{ { 0, 9, 9 }, { 1, 1, 1 } });
    if (far.size() != 1 || far[0].id != 0)
      throw std::runtime_error("farther skyline counterexample");
    equal(hvs_update::builder_entry_frontier(
              std::vector<Entry>{ { 0, 1, 1 }, { 1, 1, 1 } }),
          std::vector<Entry>{ { 1, 1, 1 } });
    std::mt19937 rng(20261116);
    unsigned cases = 0;
    for (unsigned trial = 0; trial < 1000; ++trial) {
      std::vector<Entry> all, stream;
      for (unsigned id = 0; id < 100; ++id) {
        Entry e{ id, float(rng() % 64), float(rng() % 64) };
        all.push_back(e);
        stream.push_back(e);
        stream = hvs_update::builder_entry_frontier(stream);
        equal(stream, reference(all));
        ++cases;
      }
    }
    std::cout << "ENTRY_FRONTIER_PARITY_PASS prefixes=" << cases
              << " farther=1 equal_distance_highest_id=1\n";
  }
  catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
