#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <shared_mutex>
#include <string>

namespace hvs_update {
constexpr uint64_t page_bytes = 8192;
enum class File : unsigned {
  topology = 0,
  coordinates = 1
};

struct IOStats {
  uint64_t read_pages[2] = {}, write_pages[2] = {};
  uint64_t page_lock_wait_ns = 0;
};

// Each read returns an owned page, never a pointer into a shared cache.
class Page {
public:
  Page();
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
