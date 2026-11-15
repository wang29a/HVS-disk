#pragma once
#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <vector>
namespace fixed_pruning {
struct Candidate {
  unsigned id;
  float score;
};
// Greedy relative-neighborhood pruning for one fixed fused metric. Candidate
// order is (fused distance, ID); no interval algebra or Pareto-layer threshold.
template <class PairDistance>
std::vector<Candidate> select(std::vector<Candidate> candidates,
                              unsigned degree, PairDistance pair_distance) {
  if (!degree)
    throw std::runtime_error("positive degree required");
  std::sort(candidates.begin(), candidates.end(), [](Candidate a, Candidate b) {
    return a.score < b.score || (a.score == b.score && a.id < b.id);
  });
  std::vector<Candidate> picked;
  std::unordered_set<unsigned> seen;
  for (auto q : candidates) {
    if (!seen.insert(q.id).second)
      continue;
    bool covered = false;
    for (auto p : picked) {
      if (p.score <= q.score && pair_distance(p.id, q.id) <= q.score) {
        covered = true;
        break;
      }
    }
    if (!covered) {
      picked.push_back(q);
      if (picked.size() == degree)
        break;
    }
  }
  return picked;
}
} // namespace fixed_pruning
