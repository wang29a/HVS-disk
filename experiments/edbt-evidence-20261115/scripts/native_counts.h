#pragma once
#include <cstdint>
namespace native_measure {
struct Counts {
  uint64_t pq_candidates = 0, pq_spaces = 0, exact_spaces = 0, expanded = 0;
  uint64_t requested = 0, requested_bytes = 0;
};
inline thread_local Counts counts;
}
