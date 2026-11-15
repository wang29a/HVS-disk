#include "fixed_pruning.h"
#include "pruning_semantics.h"
#include <cassert>
#include <iostream>
#include <numeric>
#include <random>
namespace ps = pruning_semantics;
namespace fp = fixed_pruning;
float fuse(float x, float y, float a) { return a * x + (1 - a) * y; }
int main() {
  unsigned equations = 0, graphs = 0;
  for (int slope = -20; slope <= 20; slope++)
    for (int rhs = -20; rhs <= 20; rhs++)
      for (int k = 0; k <= 16; k++) {
        double a = k / 16.;
        auto feasible = ps::solve(float(slope), float(rhs));
        bool brute = static_cast<long double>(slope) * a <= rhs;
        assert(ps::active(feasible, a) == brute);
        ++equations;
      }
  assert(ps::solve(0, 2).width() == 1 && ps::legacy(0, 2).empty());
  assert(ps::solve(0, 0).width() == 1 && ps::solve(0, -1).empty());
  assert(ps::active(ps::solve(2, 0), 0) && !ps::active(ps::solve(2, 0), .5));
  assert(ps::solve(-2, 0).width() == 1);
  assert(ps::active(ps::solve(-2, -2), 1) && !ps::active(ps::solve(-2, -2), 0));
  std::mt19937 rng(20261113);
  // Integer coordinates on two one-dimensional spaces make every component
  // Euclidean distance exact. This isolates pruning semantics from PQ/roundoff.
  for (unsigned trial = 0; trial < 1000; trial++) {
    const unsigned n = 12;
    int x[n], y[n];
    for (unsigned i = 0; i < n; i++) {
      x[i] = int(rng() % 41) - 20;
      y[i] = int(rng() % 41) - 20;
    }
    if (trial % 4 == 0) {
      x[1] = x[2];
      y[1] = y[2];
    }
    if (trial % 4 == 1) {
      x[1] = x[0];
      x[2] = x[0];
    }
    if (trial % 4 == 2)
      for (unsigned i = 0; i < n; i++)
        y[i] = x[i];
    for (float a : { 0.f, .125f, .25f, .5f, .75f, .875f, 1.f }) {
      auto dx = [&](unsigned p,
                    unsigned q) { return float(std::abs(x[p] - x[q])); };
      auto dy = [&](unsigned p,
                    unsigned q) { return float(std::abs(y[p] - y[q])); };
      auto dist = [&](unsigned p,
                      unsigned q) { return fuse(dx(p, q), dy(p, q), a); };
      std::vector<fp::Candidate> candidates;
      for (unsigned i = 1; i < n; i++)
        candidates.push_back({ i, dist(0, i) });
      std::shuffle(candidates.begin(), candidates.end(), rng);
      auto got = fp::select(candidates, 8, dist);
      // Independent brute reference: explicit scan over the sorted candidates
      // and all previously accepted witnesses, no production selection helper.
      std::vector<unsigned> order(n - 1), expected, symbolic;
      std::iota(order.begin(), order.end(), 1);
      std::sort(order.begin(), order.end(), [&](unsigned p, unsigned q) {
        return dist(0, p) < dist(0, q) || (dist(0, p) == dist(0, q) && p < q);
      });
      for (auto q : order) {
        bool occluded = false;
        for (auto p : expected)
          if (dist(0, p) <= dist(0, q) && dist(p, q) <= dist(0, q))
            occluded = true;
        if (!occluded)
          expected.push_back(q);
        if (expected.size() == 8)
          break;
      }
      // Same candidate pool/order/budget: pointwise interval predicate must
      // agree with direct fixed-distance pruning, including alpha endpoints.
      for (auto q : order) {
        bool occluded = false;
        for (auto p : symbolic) {
          float s1 = dx(0, p) - dx(0, q) + dy(0, q) - dy(0, p),
                r1 = dy(0, q) - dy(0, p);
          float s2 = dx(p, q) - dx(0, q) + dy(0, q) - dy(p, q),
                r2 = dy(0, q) - dy(p, q);
          if (ps::active(ps::intersect(ps::solve(s1, r1), ps::solve(s2, r2)),
                         a))
            occluded = true;
        }
        if (!occluded)
          symbolic.push_back(q);
        if (symbolic.size() == 8)
          break;
      }
      assert(got.size() == expected.size() && symbolic == expected);
      for (unsigned i = 0; i < got.size(); i++) {
        assert(got[i].id == expected[i]);
        for (unsigned j = 0; j < i; j++)
          assert(!(got[j].score <= got[i].score &&
                   dist(got[j].id, got[i].id) <= got[i].score));
      }
      ++graphs;
    }
  }
  // Duplicate IDs must not become duplicate edges, and all-distance ties have
  // a deterministic ID witness rather than an undefined Pareto ordering.
  auto duplicate = fp::select(
      std::vector<fp::Candidate>{ { 3, 0 }, { 1, 0 }, { 1, 0 }, { 2, 0 } }, 40,
      [](unsigned, unsigned) { return 0.f; });
  assert(duplicate.size() == 1 && duplicate[0].id == 1);
  std::cout << "PASS " << equations << " affine checks, " << graphs
            << " geometric brute/interval/fixed equivalence checks\n";
}
