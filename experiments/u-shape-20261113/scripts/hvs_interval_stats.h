#pragma once
#include "pruning_semantics.h"
#include <cstdint>
#include <stdexcept>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <vector>
namespace hvs_pruning {
struct Counts {
  uint64_t inequalities = 0, zero_slope = 0, zero_rhs = 0,
           missing_positive_width = 0;
  uint64_t changed_endpoint0 = 0, changed_endpoint1 = 0, fixed_calls = 0;
  uint64_t fixed_candidates = 0, fixed_pair_checks = 0, fixed_kept = 0;
  uint64_t candidate_cases = 0, candidate_width_shrunk = 0,
           legacy_keep_new_drop = 0;
  double candidate_width_reduction_sum = 0;
  double missing_width_sum = 0;
};
inline std::vector<Counts *> &registry() {
  static std::vector<Counts *> v;
  return v;
}
inline std::mutex &registry_mutex() {
  static std::mutex m;
  return m;
}
inline Counts &counts() {
  static thread_local Counts *p = [] {
    auto *v = new Counts;
    std::lock_guard<std::mutex> lock(registry_mutex());
    registry().push_back(v);
    return v;
  }();
  return *p;
}
inline std::pair<float, float> solve_record(float slope, float rhs) {
  auto &c = counts();
  ++c.inequalities;
  c.zero_slope += slope == 0;
  c.zero_rhs += rhs == 0;
  auto now = pruning_semantics::solve(slope, rhs),
       old = pruning_semantics::legacy(slope, rhs);
  double missing = now.width() - pruning_semantics::intersect(now, old).width();
  if (missing > 0) {
    ++c.missing_positive_width;
    c.missing_width_sum += missing;
  }
  c.changed_endpoint0 +=
      pruning_semantics::active(now, 0) != pruning_semantics::active(old, 0);
  c.changed_endpoint1 +=
      pruning_semantics::active(now, 1) != pruning_semantics::active(old, 1);
  // Preserve the existing float interval representation. Zero-width pruning
  // sets are conservative no-ops downstream; no format change is hidden here.
  return now.empty() ? std::make_pair(0.f, 0.f)
                     : std::make_pair(float(now.lo), float(now.hi));
}
}
extern "C" void hvs_write_pruning_stats(const char *path) {
  hvs_pruning::Counts sum;
  std::lock_guard<std::mutex> lock(hvs_pruning::registry_mutex());
  for (auto p : hvs_pruning::registry()) {
#define ADD(k) sum.k += p->k
    ADD(inequalities);
    ADD(zero_slope);
    ADD(zero_rhs);
    ADD(missing_positive_width);
    ADD(changed_endpoint0);
    ADD(changed_endpoint1);
    ADD(missing_width_sum);
    ADD(fixed_calls);
    ADD(fixed_candidates);
    ADD(fixed_pair_checks);
    ADD(fixed_kept);
    ADD(candidate_cases);
    ADD(candidate_width_shrunk);
    ADD(legacy_keep_new_drop);
    ADD(candidate_width_reduction_sum);
#undef ADD
  }
  std::ofstream out(path);
  out << std::setprecision(16) << "{\"scope\":\"actual calls in this corrected "
                                  "build; legacy predicate evaluated on the "
                                  "same call arguments, not a replay of old "
                                  "construction\"";
#define FIELD(k) out << ",\"" #k "\":" << sum.k
  FIELD(inequalities);
  FIELD(zero_slope);
  FIELD(zero_rhs);
  FIELD(missing_positive_width);
  FIELD(changed_endpoint0);
  FIELD(changed_endpoint1);
  FIELD(missing_width_sum);
  FIELD(fixed_calls);
  FIELD(fixed_candidates);
  FIELD(fixed_pair_checks);
  FIELD(fixed_kept);
  FIELD(candidate_cases);
  FIELD(candidate_width_shrunk);
  FIELD(legacy_keep_new_drop);
  FIELD(candidate_width_reduction_sum);
#undef FIELD
  out << "}\n";
  if (!out)
    throw std::runtime_error("pruning stats write");
}
