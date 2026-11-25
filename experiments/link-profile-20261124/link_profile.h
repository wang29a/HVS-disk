#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace hvs_link_profile {
using Clock = std::chrono::steady_clock;
using Tick = Clock::time_point;
using Count = std::uint64_t;

inline Tick now() { return Clock::now(); }
inline Count elapsed(Tick start, Tick end) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

enum Phase { search, prune, reverse, bookkeeping, cleanup, phases };
enum Metric { visited, expanded, candidates, skyline, retained, overflow, metrics };

struct alignas(64) Worker {
  std::array<Count, phases> time{};
  std::array<std::vector<Count>, metrics> hist;
  Count inserted = 0, initialized = 0, cleaned_rows = 0, reverse_attempts = 0;
  Count current_expanded = 0, current_layers = 0, current_overflows = 0;
  bool read_new_layers = false;

  void add(Metric metric, Count value) {
    auto &bins = hist[metric];
    if (value >= bins.size())
      bins.resize(value + 1);
    ++bins[value];
  }
};

inline thread_local Worker *worker = nullptr;

struct Session {
  static constexpr std::array<const char *, phases> phase_names{
      "candidate_search", "new_point_prune", "reverse_maintenance", "bookkeeping",
      "cleanup"};
  static constexpr std::array<const char *, metrics> metric_names{
      "search_visited", "search_expanded", "prune_candidates", "skyline_layers",
      "retained_edges", "overflow_reprunes"};
  std::vector<Worker> workers;
  Tick start = now(), insertion_begin, insertion_end, cleanup_begin, cleanup_end;

  explicit Session(unsigned n) : workers(n) {}

  void write(Count point_count, unsigned L, unsigned R, Count link_us) const {
    const char *path = std::getenv("HVS_LINK_PROFILE_JSON");
    if (!path)
      throw std::runtime_error("HVS_LINK_PROFILE_JSON is required with instrumentation enabled");
    Worker sum;
    for (const auto &w : workers) {
      sum.inserted += w.inserted;
      sum.initialized += w.initialized;
      sum.cleaned_rows += w.cleaned_rows;
      sum.reverse_attempts += w.reverse_attempts;
      for (unsigned i = 0; i < phases; ++i)
        sum.time[i] += w.time[i];
      for (unsigned i = 0; i < metrics; ++i) {
        if (w.hist[i].size() > sum.hist[i].size())
          sum.hist[i].resize(w.hist[i].size());
        for (unsigned j = 0; j < w.hist[i].size(); ++j)
          sum.hist[i][j] += w.hist[i][j];
      }
    }
    std::ofstream out(path);
    out << "{\"points\":" << point_count << ",\"L\":" << L << ",\"R\":" << R
        << ",\"insertions\":" << sum.inserted << ",\"init_threads\":" << sum.initialized
        << ",\"cleanup_rewritten_nodes\":" << sum.cleaned_rows
        << ",\"reverse_attempts\":" << sum.reverse_attempts
        << ",\"link_wall_us\":" << link_us
        << ",\"setup_wall_ns\":" << elapsed(start, insertion_begin)
        << ",\"insert_wall_ns\":" << elapsed(insertion_begin, insertion_end)
        << ",\"cleanup_wall_ns\":" << elapsed(cleanup_begin, cleanup_end)
        << ",\"worker_ns\":{";
    for (unsigned i = 0; i < phases; ++i)
      out << (i ? ",\"" : "\"") << phase_names[i] << "\":" << sum.time[i];
    out << "},\"histograms\":{";
    for (unsigned i = 0; i < metrics; ++i) {
      out << (i ? ",\"" : "\"") << metric_names[i] << "\":[";
      for (unsigned j = 0; j < sum.hist[i].size(); ++j)
        out << (j ? "," : "") << sum.hist[i][j];
      out << "]";
    }
    out << "}}\n";
    out.close();
    if (!out)
      throw std::runtime_error("failed to write Link profile");
  }
};
}
