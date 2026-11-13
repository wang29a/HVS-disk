#pragma once
#include "linux_aligned_file_reader.h"
#include "query_buf.h"
#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

// Immutable topology only. Shares completed 8 KiB pages across queries/threads.
// Requests in flight use the native query scratch, never an evictable cache
// slot.
class SharedTopoReader : public LinuxAlignedFileReader {
public:
  static constexpr uint64_t page_bytes = 8192;
  using Clock = std::chrono::steady_clock;
  struct Observation {
    uint64_t requested = 0, hits = 0, backend_pages = 0, coordinate_pages = 0;
    double topology_reader_us = 0, coordinate_reader_us = 0;
  };

private:
  struct Pending {
    bool active = false;
    uint64_t polls = 0, real_polls = 0;
    std::vector<IORequest> misses;
    Observation observation;
  };
  static thread_local Pending pending_;
  struct Entry {
    uint64_t slot;
    std::list<uint64_t>::iterator position;
  };
  std::mutex mutex_;
  std::unique_ptr<uint8_t[]> memory_;
  std::unordered_map<uint64_t, Entry> pages_;
  std::list<uint64_t> lru_;
  uint64_t capacity_ = 0, used_ = 0;
  static double micros(Clock::time_point t) {
    return std::chrono::duration<double, std::micro>(Clock::now() - t).count();
  }
  bool lookup(const IORequest &r) {
    if (!capacity_)
      return false;
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = pages_.find(r.offset / page_bytes);
    if (it == pages_.end())
      return false;
    std::memcpy(r.buf, memory_.get() + it->second.slot * page_bytes,
                page_bytes);
    lru_.splice(lru_.begin(), lru_, it->second.position);
    return true;
  }
  void publish(const IORequest &r) {
    if (!capacity_)
      return;
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t key = r.offset / page_bytes, slot;
    auto it = pages_.find(key);
    if (it != pages_.end()) {
      lru_.splice(lru_.begin(), lru_, it->second.position);
      return; // Concurrent miss: existing immutable bytes are identical.
    }
    if (used_ < capacity_)
      slot = used_++;
    else {
      uint64_t old = lru_.back();
      slot = pages_.at(old).slot;
      pages_.erase(old);
      lru_.pop_back();
    }
    std::memcpy(memory_.get() + slot * page_bytes, r.buf, page_bytes);
    lru_.push_front(key);
    pages_.emplace(key, Entry{ slot, lru_.begin() });
  }

public:
  void configure(uint64_t bytes) {
    if (bytes % page_bytes)
      throw std::runtime_error("unaligned cache budget");
    capacity_ = bytes / page_bytes;
    memory_.reset(capacity_ ? new uint8_t[bytes] : nullptr);
    clear_cache();
  }
  void clear_cache() { // Called outside all parallel query loops.
    std::lock_guard<std::mutex> lock(mutex_);
    pages_.clear();
    lru_.clear();
    used_ = 0;
  }
  uint64_t budget_bytes() const { return capacity_ * page_bytes; }
  void begin() {
    if (pending_.polls)
      throw std::runtime_error("unfinished previous query");
    pending_.observation = Observation{};
    pending_.active = true;
  }
  Observation end() {
    if (pending_.polls || pending_.real_polls)
      throw std::runtime_error("unfinished topology batch");
    auto result = pending_.observation;
    if (result.requested != result.hits + result.backend_pages)
      throw std::runtime_error("cache accounting mismatch");
    pending_.active = false;
    return result;
  }
  void send_io(std::vector<IORequest> &reqs, void *ctx, bool write,
               FileHandle fd) override {
    if (!pending_.active || fd != topo_file_desc) {
      LinuxAlignedFileReader::send_io(reqs, ctx, write, fd);
      return;
    }
    if (write || pending_.polls || reqs.size() > 128)
      throw std::runtime_error("unsupported topology batch");
    // The native query-local map retains pointers into a ring of scratch pages.
    if (pending_.observation.requested + reqs.size() > MAX_N_SECTOR_READS)
      throw std::runtime_error("HVS_TOPO_SCRATCH_LIMIT");
    auto t = Clock::now();
    pending_.misses.clear();
    for (auto &r : reqs) {
      if (r.len != page_bytes || r.offset % page_bytes)
        throw std::runtime_error("topology request is not one 8 KiB page");
      ++pending_.observation.requested;
      if (lookup(r)) {
        ++pending_.observation.hits;
        r.finished = true;
      } else
        pending_.misses.push_back(r);
    }
    pending_.polls = reqs.size();
    pending_.real_polls = pending_.misses.size();
    pending_.observation.backend_pages += pending_.misses.size();
    if (!pending_.misses.empty()) {
      std::array<iocb, 128> controls;
      std::array<iocb *, 128> batch;
      for (size_t i = 0; i < pending_.misses.size(); ++i) {
        auto &r = pending_.misses[i];
        io_prep_pread(&controls[i], fd, r.buf, r.len, r.offset);
        batch[i] = &controls[i];
      }
      const int count = pending_.misses.size();
      if (io_submit(static_cast<io_context_t>(ctx), count, batch.data()) !=
          count)
        throw std::runtime_error("partial topology submission");
    }
    pending_.observation.topology_reader_us += micros(t);
  }
  void poll_wait(void *ctx) override {
    if (!pending_.active || !pending_.polls) {
      LinuxAlignedFileReader::poll_wait(ctx);
      return;
    }
    auto t = Clock::now();
    if (pending_.real_polls) {
      io_event event;
      int count;
      do {
        count =
            io_getevents(static_cast<io_context_t>(ctx), 1, 1, &event, nullptr);
      } while (count == -EINTR);
      if (count != 1 || event.res != page_bytes || event.res2 != 0)
        throw std::runtime_error("incomplete topology read");
      if (!--pending_.real_polls)
        for (auto &r : pending_.misses)
          publish(r);
    }
    // The unchanged native caller polls once per requested page. Cache hits
    // complete in memory; only misses consume kernel completions.
    if (!--pending_.polls)
      pending_.misses.clear();
    pending_.observation.topology_reader_us += micros(t);
  }
  void read_fd(int fd, std::vector<IORequest> &reqs, void *ctx) override {
    auto t = Clock::now();
    LinuxAlignedFileReader::read_fd(fd, reqs, ctx);
    if (pending_.active && fd == coord_file_desc) {
      for (auto &r : reqs) {
        if (r.len != page_bytes)
          throw std::runtime_error("coordinate page size");
        ++pending_.observation.coordinate_pages;
      }
      pending_.observation.coordinate_reader_us += micros(t);
    }
  }
};

inline thread_local SharedTopoReader::Pending SharedTopoReader::pending_{};
