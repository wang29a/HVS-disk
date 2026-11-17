#pragma once
#include <algorithm>
#include <vector>

namespace hvs_update {
// Match the verified B builder's UpdateEnterpointSet: sort ascending by
// (space2, space1, ID), scan in reverse, retain strictly increasing space1.
// This is the FARTHER skyline. Equal-distance entries keep the highest ID.
template <class Entry>
std::vector<Entry> builder_entry_frontier(std::vector<Entry> candidates) {
  std::sort(candidates.begin(), candidates.end(),
            [](const Entry &a, const Entry &b) {
    return a.y < b.y ||
           (a.y == b.y && (a.x < b.x || (a.x == b.x && a.id < b.id)));
  });
  float maximum = 0;
  std::vector<Entry> selected;
  for (auto it = candidates.rbegin(); it != candidates.rend(); ++it)
    if (it->x > maximum) {
      selected.push_back(*it);
      maximum = it->x;
    }
  return selected;
}
} // namespace hvs_update
