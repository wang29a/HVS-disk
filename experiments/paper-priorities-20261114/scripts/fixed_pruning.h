#pragma once
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <unordered_set>
#include <vector>
namespace fixed_pruning {
struct Candidate {
  unsigned id;
  float score;
};
inline float rho() {
  static const float value = [] {
    const char *s = std::getenv("HVS_PRUNE_RHO");
    char *end = nullptr;
    if (!s)
      throw std::runtime_error("HVS_PRUNE_RHO required");
    float v = std::strtof(s, &end);
    if (!end || *end || !std::isfinite(v) || v < 1 || v > 2)
      throw std::runtime_error("rho must be in [1,2]");
    return v;
  }();
  return value;
}
template <class PairDistance>
std::vector<Candidate> select(std::vector<Candidate> candidates,
                              unsigned degree, PairDistance distance) {
  if (!degree)
    throw std::runtime_error("positive degree required");
  const float strength = rho();
  std::sort(candidates.begin(), candidates.end(), [](Candidate a, Candidate b) {
    return a.score < b.score || (a.score == b.score && a.id < b.id);
  });
  std::vector<Candidate> picked;
  std::unordered_set<unsigned> seen;
  for (auto q : candidates) {
    if (!seen.insert(q.id).second)
      continue;
    bool covered = false;
    for (auto p : picked)
      if (p.score <= q.score && strength * distance(p.id, q.id) <= q.score) {
        covered = true;
        break;
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
