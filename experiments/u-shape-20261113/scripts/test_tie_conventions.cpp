#include "fixed_pruning.h"
#include "pruning_semantics.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
template <class PairDistance>
std::vector<unsigned> strict(std::vector<fixed_pruning::Candidate> v,
                             PairDistance pair) {
  std::sort(v.begin(), v.end(), [](auto a, auto b) {
    return a.score < b.score || (a.score == b.score && a.id < b.id);
  });
  std::vector<fixed_pruning::Candidate> picked;
  for (auto q : v) {
    bool covered = false;
    for (auto p : picked)
      if (p.score < q.score && pair(p.id, q.id) < q.score)
        covered = true;
    if (!covered)
      picked.push_back(q);
  }
  std::vector<unsigned> result;
  for (auto p : picked)
    result.push_back(p.id);
  return result;
}
int main() {
  std::vector<fixed_pruning::Candidate> allzero{ { 3, 0 }, { 1, 0 }, { 2, 0 } };
  auto le = fixed_pruning::select(allzero, 40,
                                  [](unsigned, unsigned) { return 0.f; });
  auto lt = strict(allzero, [](unsigned, unsigned) { return 0.f; });
  assert(le.size() == 1 && le[0].id == 1);
  assert(lt == std::vector<unsigned>({ 1, 2, 3 }));
  // A zero slope with positive RHS is a missed case under either convention.
  assert(pruning_semantics::solve(0, 2).width() == 1 &&
         pruning_semantics::legacy(0, 2).empty());
  // A zero slope and zero RHS is instead a genuine policy distinction.
  assert(pruning_semantics::solve(0, 0).width() == 1);
  assert(!(0.f < 0.f));
  // Endpoint differences are explicit, not hidden by a numeric epsilon.
  assert(pruning_semantics::active(pruning_semantics::solve(-1, 0), 0));
  assert(!(-1.f * 0.f < 0.f));
  assert(-1.f * 1.f < 0.f);
  std::cout << "PASS: LE and strict ties differ; positive-margin missing zero "
               "branch remains an upstream issue\n";
}
