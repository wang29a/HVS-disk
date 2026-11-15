#pragma once
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>
namespace pruning_semantics {
struct Interval {
  double lo = 0, hi = -1;
  bool empty() const { return lo > hi; }
  double width() const { return empty() ? 0 : hi - lo; }
};
inline Interval intersect(Interval a, Interval b) {
  return { std::max(a.lo, b.lo), std::min(a.hi, b.hi) };
}
// Closed feasible set of slope * alpha <= rhs on [0,1]. No epsilon:
// a singleton endpoint is distinct from an empty set.
inline Interval solve(float slope, float rhs) {
  if (slope == 0)
    return rhs >= 0 ? Interval{ 0, 1 }:Interval{};
  float cut = rhs / slope;
  return slope > 0 ? intersect({ 0, 1 }, { 0, cut })
                   : intersect({ 0, 1 }, { cut, 1 });
}
inline Interval legacy(float slope, float rhs) {
  Interval out;
  if (slope > 0 && rhs > 0)
    out = { 0, std::min(rhs / slope, 1.f) };
  else if (slope < 0 && rhs < 0)
    out = { std::min(rhs / slope, 1.f), 1 };
  else if (slope < 0 && rhs > 0)
    out = { 0, 1 };
  // The old code uses (0,0) and every other zero-width interval as empty.
  if (out.width() <= 0)
    return {};
  return out;
}
inline bool active(Interval p, double a) {
  return !p.empty() && a >= p.lo && a <= p.hi;
}
inline double union_width(std::vector<Interval> x) {
  x.erase(
      std::remove_if(x.begin(), x.end(), [](Interval r) { return r.empty(); }),
      x.end());
  std::sort(x.begin(), x.end(),
            [](Interval a, Interval b) { return a.lo < b.lo; });
  double result = 0;
  Interval current;
  for (auto p : x) {
    if (current.empty())
      current = p;
    else if (p.lo <= current.hi)
      current.hi = std::max(current.hi, p.hi);
    else {
      result += current.width();
      current = p;
    }
  }
  return result + current.width();
}
inline std::vector<Interval> difference(Interval a, Interval b) {
  if (a.empty())
    return {};
  b = intersect(a, b);
  if (b.empty())
    return { a };
  std::vector<Interval> out;
  if (a.lo < b.lo)
    out.push_back({ a.lo, b.lo });
  if (b.hi < a.hi)
    out.push_back({ b.hi, a.hi });
  return out;
}
} // namespace pruning_semantics
