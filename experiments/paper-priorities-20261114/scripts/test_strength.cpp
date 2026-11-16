#include "fixed_pruning.h"
#include <array>
#include <cassert>
#include <iostream>
#include <random>
int main() {
  std::mt19937 rng(20261114);
  std::uniform_int_distribution<int> draw(-20, 20);
  unsigned checked = 0;
  for (unsigned trial = 0; trial < 500; trial++) {
    std::vector<std::array<float, 2> > x(50);
    for (auto &v : x)
      v = { float(draw(rng)), float(draw(rng)) };
    auto d = [&](unsigned a, unsigned b) {
      float p = x[a][0] - x[b][0], q = x[a][1] - x[b][1];
      return std::sqrt(p * p + q * q);
    };
    std::vector<fixed_pruning::Candidate> pool;
    for (unsigned j = 1; j < 50; j++)
      pool.push_back({ j, d(0, j) });
    auto actual = fixed_pruning::select(pool, 40, d);
    // Independent sequential reference: classify every candidate against the
    // witnesses already retained, then verify the retained-pair invariant.
    std::sort(pool.begin(), pool.end(), [](auto a, auto b) {
      return a.score < b.score || (a.score == b.score && a.id < b.id);
    });
    std::vector<unsigned> reference;
    for (auto q : pool) {
      unsigned dominators = 0;
      for (auto p : reference)
        dominators +=
            d(0, p) <= q.score && fixed_pruning::rho() * d(p, q.id) <= q.score;
      if (!dominators)
        reference.push_back(q.id);
      if (reference.size() == 40)
        break;
    }
    assert(actual.size() == reference.size());
    for (unsigned j = 0; j < actual.size(); j++) {
      assert(actual[j].id == reference[j]);
      for (unsigned i = 0; i < j; i++)
        assert(!(actual[i].score <= actual[j].score &&
                 fixed_pruning::rho() * d(actual[i].id, actual[j].id) <=
                     actual[j].score));
    }
    ++checked;
  }
  std::cout << "PASS strength " << fixed_pruning::rho() << " cases " << checked
            << "\n";
}
