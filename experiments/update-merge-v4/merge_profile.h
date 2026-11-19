#pragma once

#include <chrono>
#include <cstdint>
#include <ostream>
#include <vector>

namespace hvs_update {
using MergeClock = std::chrono::steady_clock;

inline uint64_t merge_elapsed(MergeClock::time_point start) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(MergeClock::now() -
                                                               start)
      .count();
}

struct MergeTimer {
  explicit MergeTimer(uint64_t *count)
      : count(count), start(count ? MergeClock::now() : MergeClock::time_point{}) {}
  ~MergeTimer() {
    if (count)
      *count += merge_elapsed(start);
  }

  uint64_t *count;
  MergeClock::time_point start;
};

struct MergeProfile {
  uint64_t total_ns = 0, reprune_ns = 0, repair_ns = 0, write_ns = 0;
  uint64_t repruned_nodes = 0, coordinate_io_ns = 0, reprune_coordinate_io_ns = 0;
  uint64_t coordinate_reads = 0, distinct_pages = 0;
  uint64_t vector_loads = 0, distinct_vectors = 0;
  unsigned merge_threads = 0;
  std::vector<uint8_t> seen_pages, seen_vectors;

  void page(uint64_t number) {
    ++coordinate_reads;
    auto &seen = seen_pages.at(number);
    distinct_pages += !seen;
    seen = true;
  }

  void vector(uint32_t id) {
    ++vector_loads;
    auto &seen = seen_vectors.at(id);
    distinct_vectors += !seen;
    seen = true;
  }

  void write(std::ostream &out) const {
    out << "{\"total_ns\":" << total_ns
        << ",\"planning_ns\":" << total_ns - reprune_ns - repair_ns - write_ns
        << ",\"reprune_ns\":" << reprune_ns
        << ",\"repair_ns\":" << repair_ns
        << ",\"write_ns\":" << write_ns
        << ",\"repruned_nodes\":" << repruned_nodes
        << ",\"coordinate_io_ns\":" << coordinate_io_ns
        << ",\"reprune_coordinate_io_ns\":" << reprune_coordinate_io_ns
        << ",\"coordinate_read_pages\":" << coordinate_reads
        << ",\"unique_coordinate_pages\":" << distinct_pages
        << ",\"coordinate_vector_loads\":" << vector_loads
        << ",\"unique_coordinate_vectors\":" << distinct_vectors
        << ",\"merge_threads\":" << merge_threads << '}';
  }
};
} // namespace hvs_update
