#pragma once

#include <cstdint>
#include <vector>

namespace shard_degree {
inline const std::vector<uint8_t> *caps = nullptr;
inline thread_local unsigned prune_limit = 0;

inline unsigned cap(unsigned node, unsigned fallback) {
    return caps ? (*caps)[node] : fallback;
}

class ScopedPruneLimit {
  public:
    explicit ScopedPruneLimit(unsigned value) : previous(prune_limit) {
        prune_limit = value;
    }

    ~ScopedPruneLimit() { prune_limit = previous; }

    ScopedPruneLimit(const ScopedPruneLimit &) = delete;
    ScopedPruneLimit &operator=(const ScopedPruneLimit &) = delete;

  private:
    unsigned previous;
};
} // namespace shard_degree
