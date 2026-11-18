#pragma once
#include "neighbor.h"
#include "tsl/robin_set.h"
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace hvs_update {
// Extracted from multigraph.cpp Engine::traverse (source SHA256
// 0b2bdfac57a34aabd3bbc3d1b9426aa0fc907ff87186649b0194169abd0cc3b9).
// Retain the beam4 / InsertIntoPool / rewind control flow. Storage and
// publication checks are callbacks so every page still uses PageStore locks.
template <class Neighbor>
std::vector<unsigned> take_beam(std::vector<Neighbor> &pool, unsigned begin,
                                unsigned count) {
  std::vector<unsigned> frontier;
  for (unsigned i = begin; i < count && frontier.size() < 4; ++i)
    if (pool[i].flag) {
      frontier.push_back(pool[i].id);
      pool[i].flag = false;
    }
  return frontier;
}
template <class Score, class Fetch, class Row, class Ready>
std::vector<pipeann::Neighbor>
query_beam(const std::vector<unsigned> &entries, unsigned L, Score scores,
           Fetch fetch, Row row, Ready ready, uint64_t &expanded) {
  if (entries.size() > L)
    throw std::runtime_error("L below entry count");
  std::vector<pipeann::Neighbor> ret(
      std::max(size_t(L) * 10, entries.size() + 1));
  tsl::robin_set<uint64_t> visited;
  std::vector<unsigned> live;
  for (auto id : entries)
    if (ready(id))
      live.push_back(id);
  auto ds = scores(live);
  unsigned cur = 0;
  for (size_t j = 0; j < live.size(); ++j) {
    ret[cur++] = pipeann::Neighbor(live[j], ds[j], true);
    visited.insert(live[j]);
  }
  std::sort(ret.begin(), ret.begin() + cur);
  unsigned k = 0;
  while (k < cur) {
    unsigned nk = cur;
    auto frontier = take_beam(ret, k, cur);
    fetch(frontier);
    expanded += frontier.size();
    for (auto id : frontier) {
      std::vector<unsigned> nbrs;
      for (auto nb : row(id)) {
        if (!ready(nb) || visited.find(nb) != visited.end())
          continue;
        visited.insert(nb);
        nbrs.push_back(nb);
      }
      auto d = scores(nbrs);
      for (size_t j = 0; j < nbrs.size(); ++j)
        if (cur < L || d[j] < ret[cur - 1].distance) {
          auto pos = pipeann::InsertIntoPool(
              ret.data(), cur, pipeann::Neighbor(nbrs[j], d[j], true));
          if (cur < L)
            ++cur;
          if (pos < nk)
            nk = pos;
        }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
  ret.resize(cur);
  return ret;
}
} // namespace hvs_update
