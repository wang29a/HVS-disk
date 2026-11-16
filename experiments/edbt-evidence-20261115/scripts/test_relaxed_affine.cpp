#include "relaxation.h"
#include "pruning_semantics.h"
#include <random>
#include <iostream>
int main() {
  std::mt19937 rng(20261115);
  unsigned checked = 0, near_boundary = 0;
  float rho = hvs_unified_rho();
  for (unsigned i = 0; i < 1000; ++i) {
    auto norm = [&]() {
      double x = int(rng() % 201) - 100, y = int(rng() % 201) - 100;
      return float(std::sqrt(x * x + y * y));
    };
    float pe = norm(), ps = norm(), qe = norm(), qs = norm(), xe = rho * norm(),
          xs = rho * norm();
    if (i % 10 == 0) {
      pe = qe;
      ps = qs;
      xe = qe;
      xs = qs;
    }
    auto a = pruning_semantics::solve(pe - qe + qs - ps, qs - ps);
    auto b = pruning_semantics::solve(xe - qe + qs - xs, qs - xs);
    auto both = pruning_semantics::intersect(a, b);
    for (unsigned k = 0; k <= 100; ++k) {
      double w = k / 100.;
      double p = w * pe + (1 - w) * ps, q = w * qe + (1 - w) * qs,
             x = w * xe + (1 - w) * xs;
      // Floating interval division rounds endpoints; ambiguous random cuts
      // are counted separately, not silently asserted with an epsilon solver.
      if ((std::abs(p - q) < 1e-4 && (pe != qe || ps != qs)) ||
          (std::abs(x - q) < 1e-4 && (xe != qe || xs != qs))) {
        ++near_boundary;
        continue;
      }
      if (pruning_semantics::active(both, w) != (p <= q && x <= q))
        throw std::runtime_error("affine/grid geometric mismatch");
      ++checked;
    }
  }
  for (float rhs : { -1.f, 0.f, 1.f })
    for (double w : { 0., .1, .5, .9, 1. })
      if (pruning_semantics::active(pruning_semantics::solve(0, rhs), w) !=
          (rhs >= 0))
        throw std::runtime_error("zero slope closure");
  std::cout << "PASS rho=" << rho << " affine-grid comparisons=" << checked
            << " near-float-boundary=" << near_boundary << "\n";
}
