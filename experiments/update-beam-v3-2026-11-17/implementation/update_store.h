#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <shared_mutex>
#include <string>
#include <vector>

namespace hvs_update {
constexpr uint64_t page_bytes = 8192;
constexpr size_t read_batch_pages = 48;
enum class File : unsigned {
  topology = 0,
  coordinates = 1
};

struct IOStats {
  uint64_t read_pages[2] = {}, write_pages[2] = {};
  uint64_t page_lock_wait_ns = 0;
#ifdef HVS_PROFILE_PHASES
  uint64_t phase_ns[7] = {}, insert_total_ns = 0, reprune_ns = 0;
  uint64_t read_io_ns[2] = {}, write_io_ns[2] = {};
  uint64_t coordinate_calls = 0, coordinate_cache_hits = 0,
           coordinate_materialized_bytes = 0;
#endif
  uint64_t read_batches = 0, write_batches = 0, max_read_batch = 0;
  uint64_t optimistic_retries = 0, degree_reservation_retries = 0;
};

// Each read returns an owned page, never a pointer into a shared cache.
class Page {
public:
  explicit Page(bool zero = true);
  ~Page();
  Page(const Page &) = delete;
  Page &operator=(const Page &) = delete;
  unsigned char *data() { return data_; }
  const unsigned char *data() const { return data_; }

private:
  unsigned char *data_;
};

// One physical lock domain shared by every read and read-modify-write.
// Bounds are fixed at load time and include future append pages.
class PageStore {
public:
  PageStore(const std::string &topology, const std::string &coordinates,
            uint64_t max_topology_pages, uint64_t max_coordinate_pages);
  ~PageStore();
  PageStore(const PageStore &) = delete;
  PageStore &operator=(const PageStore &) = delete;
  uint64_t read(File file, uint64_t page, Page &private_page, IOStats &stats);
  std::vector<uint64_t> read_many(File file, const std::vector<uint64_t> &pages,
                                  const std::vector<Page *> &buffers,
                                  IOStats &stats);
  // False means the page version or an atomic indegree reservation changed.
  // Preparation is outside the write lock; commit runs after completed write.
  bool compare_write(File file, uint64_t page, uint64_t expected, Page &buffer,
                     const std::function<bool()> &reserve,
                     const std::function<void()> &commit, IOStats &stats);
  uint64_t read_for_update(File file, uint64_t page, Page &buffer,
                           IOStats &stats);
  uint64_t version(File file, uint64_t page) const;
  void modify(File file, uint64_t page,
              const std::function<void(unsigned char *)> &change,
              IOStats &stats);
  void sync();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class AtomicBitmap {
public:
  explicit AtomicBitmap(uint32_t capacity);
  bool test(uint32_t id) const;
  void set(uint32_t id);
  std::unique_ptr<uint64_t[]> snapshot() const;
  uint32_t capacity() const { return capacity_; }
  uint64_t words() const { return (uint64_t(capacity_) + 63) / 64; }

private:
  uint32_t capacity_;
  std::unique_ptr<std::atomic<uint64_t> []> bits_;
};
} // namespace hvs_update
