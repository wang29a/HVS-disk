#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

namespace hvs_update {
struct DegreeCounters {
  uint64_t reverse_reprunes = 0, full_reprunes = 0, protection_events = 0;
  uint64_t protected_edges = 0, protected_old_edges = 0,
           protected_new_edges = 0;
  uint64_t reverse_capacity_conflicts = 0, full_before_edges = 0,
           full_after_edges = 0;
  std::array<uint64_t, 41> full_after_degree{}, full_raw_after_degree{};
};
struct ProtectedNeighbors {
  std::vector<uint32_t> ids;
  uint64_t old_restored = 0, new_restored = 0;
  bool capacity_conflict = false;
};
// Caller serializes this decision with physical page write completion and
// committed-source indegree accounting. Pending reverse insertion participates
// logically before pruning; its first incoming edge must survive if it fits.
template <class Count, class Alive>
ProtectedNeighbors
protect_last_incoming(const std::vector<uint32_t> &old,
                      std::vector<uint32_t> ranked, unsigned limit, Count count,
                      Alive alive,
                      uint32_t pending = std::numeric_limits<uint32_t>::max()) {
  if (old.size() > limit || ranked.size() > limit)
    throw std::runtime_error("guard degree overflow");
  auto contains = [](const std::vector<uint32_t> &v, uint32_t id) {
    return std::find(v.begin(), v.end(), id) != v.end();
  };
  auto unique = [](std::vector<uint32_t> v) {
    std::sort(v.begin(), v.end());
    return std::adjacent_find(v.begin(), v.end()) == v.end();
  };
  if (!unique(old) || !unique(ranked))
    throw std::runtime_error("guard duplicate input");
  std::vector<uint32_t> required;
  for (auto v : old)
    if (alive(v) && count(v) == 1)
      required.push_back(v);
  bool need_new = pending != std::numeric_limits<uint32_t>::max() &&
                  !contains(old, pending) && alive(pending) &&
                  count(pending) == 0;
  ProtectedNeighbors result;
  if (need_new && required.size() == limit) {
    result.capacity_conflict = true;
    ranked.erase(std::remove(ranked.begin(), ranked.end(), pending),
                 ranked.end());
    need_new = false;
  }
  if (need_new)
    required.push_back(pending);
  for (auto v : required)
    if (!contains(ranked, v)) {
      if (ranked.size() < limit)
        ranked.push_back(v);
      else {
        auto last =
            std::find_if(ranked.rbegin(), ranked.rend(),
                         [&](uint32_t id) { return !contains(required, id); });
        if (last == ranked.rend())
          throw std::runtime_error("no unprotected replacement slot");
        *last = v;
      }
      if (contains(old, v))
        ++result.old_restored;
      else
        ++result.new_restored;
    }
  result.ids = std::move(ranked);
  return result;
}
} // namespace hvs_update
