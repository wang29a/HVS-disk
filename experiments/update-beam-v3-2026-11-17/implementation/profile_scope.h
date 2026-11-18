#pragma once
#ifdef HVS_PROFILE_PHASES
#include "update_store.h"
#include <chrono>
namespace hvs_update {
inline uint64_t profile_now() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct ProfileFinish {
  IOStats &stats;
  uint64_t start;
  uint64_t &last;
  ~ProfileFinish() {
    auto end = profile_now();
    stats.phase_ns[6] += end - last;
    stats.insert_total_ns += end - start;
  }
};
}
#endif
