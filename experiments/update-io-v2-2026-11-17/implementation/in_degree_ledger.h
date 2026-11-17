#pragma once
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>
namespace hvs_update {
// Counts are conservative while a page write is in flight: net removals are
// reserved before writing, net additions become usable only after completion.
// Thus another page cannot spend an incoming edge that is not durable yet.
struct DegreeConflict : std::runtime_error {
  DegreeConflict() : std::runtime_error("stale optimistic indegree") {}
};
class AtomicDegreeLedger {
  uint32_t capacity_;
  std::unique_ptr<std::atomic<uint32_t> []> count_;
  std::unique_ptr<std::atomic<uint8_t> []> positive_;

public:
  explicit AtomicDegreeLedger(uint32_t n)
      : capacity_(n), count_(new std::atomic<uint32_t>[n]),
        positive_(new std::atomic<uint8_t>[n]) {
    for (uint32_t i = 0; i < n; ++i) {
      count_[i].store(0);
      positive_[i].store(0);
    }
  }
  uint32_t count(uint32_t id) const {
    return count_[id].load(std::memory_order_acquire);
  }
  uint8_t positive(uint32_t id) const {
    return positive_[id].load(std::memory_order_acquire);
  }
  // Used only for initialized sources and the noexcept publication tail.
  void add_committed(uint32_t id, uint32_t amount = 1) noexcept {
    count_[id].fetch_add(amount, std::memory_order_acq_rel);
    positive_[id].store(1, std::memory_order_release);
  }
  std::vector<uint32_t> snapshot(uint32_t n) const {
    std::vector<uint32_t> v(n);
    for (uint32_t i = 0; i < n; ++i)
      v[i] = count(i);
    return v;
  }
  std::vector<uint8_t> history(uint32_t n) const {
    std::vector<uint8_t> v(n);
    for (uint32_t i = 0; i < n; ++i)
      v[i] = positive(i);
    return v;
  }
  class Transaction {
    struct Value {
      uint32_t base;
      int64_t value;
    };
    AtomicDegreeLedger &ledger_;
    std::map<uint32_t, Value> values_;
    std::vector<std::pair<uint32_t, uint32_t> > reserved_;
    bool finished_ = false;

  public:
    explicit Transaction(AtomicDegreeLedger &ledger) : ledger_(ledger) {}
    Transaction(const Transaction &) = delete;
    ~Transaction() { rollback(); }
    uint32_t count(uint32_t id) {
      auto i = values_.find(id);
      if (i == values_.end()) {
        auto n = ledger_.count(id);
        i = values_.emplace(id, Value{ n, n }).first;
      }
      return uint32_t(i->second.value);
    }
    void change(uint32_t id, int delta) {
      count(id);
      auto &v = values_.at(id);
      v.value += delta;
      if (v.value < 0)
        throw DegreeConflict();
      if (v.value > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("indegree overflow");
    }
    void rollback() noexcept {
      if (finished_)
        return;
      for (auto &v : reserved_)
        ledger_.count_[v.first].fetch_add(v.second, std::memory_order_acq_rel);
      reserved_.clear();
    }
    bool reserve(const std::function<bool(uint32_t)> &alive) {
      // Allocate before touching atomic counts, so rollback never allocates.
      reserved_.reserve(values_.size());
      for (auto &item : values_) {
        auto id = item.first;
        auto &v = item.second;
        if (v.value < int64_t(v.base)) {
          if (alive(id) && v.value == 0) {
            rollback();
            return false;
          }
          uint32_t expected = v.base;
          if (!ledger_.count_[id].compare_exchange_strong(
                   expected, uint32_t(v.value), std::memory_order_acq_rel)) {
            rollback();
            return false;
          }
          reserved_.emplace_back(id, v.base - uint32_t(v.value));
        }
      }
      return true;
    }
    void commit() noexcept {
      for (auto &item : values_) {
        auto &v = item.second;
        if (v.value > int64_t(v.base))
          ledger_.add_committed(item.first, uint32_t(v.value - v.base));
      }
      reserved_.clear();
      finished_ = true;
    }
  };
};
} // namespace hvs_update
