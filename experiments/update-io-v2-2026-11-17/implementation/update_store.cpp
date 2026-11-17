#include "update_store.h"
#include "linux_aligned_file_reader.h"
#include <chrono>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace hvs_update {
namespace {
uint64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
} // namespace

Page::Page() {
  void *p = nullptr;
  if (posix_memalign(&p, page_bytes, page_bytes))
    throw std::bad_alloc();
  data_ = static_cast<unsigned char *>(p);
  std::memset(data_, 0, page_bytes);
}
Page::~Page() { std::free(data_); }

struct PageStore::Impl {
  LinuxAlignedFileReader reader;
  int fd[2] = { -1, -1 };
  uint64_t capacity[2];
  std::atomic<uint64_t> initialized_pages[2];
  std::unique_ptr<std::shared_mutex[]> locks[2];
  std::unique_ptr<std::atomic<uint64_t> []> versions[2];

  Impl(const std::string &topology, const std::string &coordinates, uint64_t nt,
       uint64_t nc)
      : capacity{ nt, nc } {
    require(nt && nc, "zero page capacity");
    const std::string paths[2] = { topology, coordinates };
    for (unsigned f = 0; f < 2; ++f) {
      struct stat s;
      require(::stat(paths[f].c_str(), &s) == 0 && S_ISREG(s.st_mode),
              "page store requires existing regular files");
      require(s.st_size >= 0 && uint64_t(s.st_size) % page_bytes == 0 &&
                  uint64_t(s.st_size) <= capacity[f] * page_bytes,
              "unaligned or oversized page file");
      initialized_pages[f].store(uint64_t(s.st_size) / page_bytes);
      locks[f] = std::make_unique<std::shared_mutex[]>(capacity[f]);
      versions[f] = std::make_unique<std::atomic<uint64_t> []>(capacity[f]);
      for (uint64_t page = 0; page < capacity[f]; ++page)
        versions[f][page].store(0, std::memory_order_relaxed);
    }
    // Legacy open(false) only binds this file descriptor; it does not load an
    // SSDIndex, alpha sidecar, cache or index-format-specific metadata.
    reader.strategy = 0;
    reader.open(topology, true, false, false);
    fd[0] = reader.file_desc;
    fd[1] = ::open(coordinates.c_str(), O_RDWR | O_DIRECT | O_LARGEFILE);
    if (fd[1] < 0) {
      reader.close();
      throw std::runtime_error("open coordinate file");
    }
  }
  ~Impl() {
    ::close(fd[1]);
    reader.close();
  }
  unsigned checked(File file, uint64_t page) {
    unsigned f = static_cast<unsigned>(file);
    require(f < 2 && page < capacity[f], "physical page outside max-points");
    return f;
  }
  void read_locked(unsigned f, uint64_t page, Page &p) {
    require(page < initialized_pages[f].load(std::memory_order_acquire),
            "page read beyond initialized file");
    std::vector<IORequest> requests;
    requests.emplace_back(page * page_bytes, page_bytes,
                          reinterpret_cast<char *>(p.data()), 0, 0);
    reader.read_fd(fd[f], requests, reader.get_ctx());
  }
};

PageStore::PageStore(const std::string &topology,
                     const std::string &coordinates, uint64_t nt, uint64_t nc)
    : impl_(std::make_unique<Impl>(topology, coordinates, nt, nc)) {}
PageStore::~PageStore() = default;
uint64_t PageStore::read(File file, uint64_t page, Page &p, IOStats &stats) {
  unsigned f = impl_->checked(file, page);
  uint64_t start = now_ns();
  std::shared_lock<std::shared_mutex> lock(impl_->locks[f][page]);
  stats.page_lock_wait_ns += now_ns() - start;
  impl_->read_locked(f, page, p);
  ++stats.read_pages[f];
  ++stats.read_batches;
  stats.max_read_batch = std::max<uint64_t>(stats.max_read_batch, 1);
  return impl_->versions[f][page].load(std::memory_order_relaxed);
}
uint64_t PageStore::read_for_update(File file, uint64_t page, Page &buffer,
                                    IOStats &stats) {
  unsigned f = impl_->checked(file, page);
  auto tick = now_ns();
  std::shared_lock<std::shared_mutex> held(impl_->locks[f][page]);
  stats.page_lock_wait_ns += now_ns() - tick;
  if (page < impl_->initialized_pages[f].load(std::memory_order_acquire)) {
    impl_->read_locked(f, page, buffer);
    ++stats.read_pages[f];
    ++stats.read_batches;
    stats.max_read_batch = std::max<uint64_t>(stats.max_read_batch, 1);
  } else
    std::memset(buffer.data(), 0, page_bytes);
  return impl_->versions[f][page].load();
}
uint64_t PageStore::version(File file, uint64_t page) const {
  unsigned f = impl_->checked(file, page);
  return impl_->versions[f][page].load(std::memory_order_acquire);
}
void PageStore::modify(File file, uint64_t page,
                       const std::function<void(unsigned char *)> &change,
                       IOStats &stats) {
  unsigned f = impl_->checked(file, page);
  uint64_t start = now_ns();
  std::unique_lock<std::shared_mutex> lock(impl_->locks[f][page]);
  stats.page_lock_wait_ns += now_ns() - start;
  Page private_page;
  if (page < impl_->initialized_pages[f].load(std::memory_order_acquire)) {
    impl_->read_locked(f, page, private_page);
    ++stats.read_pages[f];
    ++stats.read_batches;
    stats.max_read_batch = std::max<uint64_t>(stats.max_read_batch, 1);
  }
  // New append pages start completely zeroed, including their unused slots.
  // No read is issued beyond EOF. pwrite extends in exact page units.
  change(private_page.data());
  std::vector<IORequest> requests;
  requests.emplace_back(page * page_bytes, page_bytes,
                        reinterpret_cast<char *>(private_page.data()), 0, 0);
  impl_->reader.write_fd(impl_->fd[f], requests, impl_->reader.get_ctx());
  ++stats.write_pages[f];
  ++stats.write_batches;
  auto size = impl_->initialized_pages[f].load();
  while (size < page + 1 &&
         !impl_->initialized_pages[f].compare_exchange_weak(size, page + 1)) {
  }
  impl_->versions[f][page].fetch_add(1, std::memory_order_release);
}
std::vector<uint64_t> PageStore::read_many(File file,
                                           const std::vector<uint64_t> &pages,
                                           const std::vector<Page *> &buffers,
                                           IOStats &stats) {
  require(pages.size() == buffers.size(), "batch size mismatch");
  unsigned f = static_cast<unsigned>(file);
  require(f < 2, "invalid batch file");
  for (size_t i = 0; i < pages.size(); ++i) {
    impl_->checked(file, pages[i]);
    require(!i || pages[i - 1] < pages[i],
            "batch pages must be sorted and unique");
  }
  std::vector<uint64_t> versions(pages.size());
  // Bound locks as well as I/O requests. Each page remains locked through its
  // completed read into the private buffer; no cross-page snapshot is promised.
  // 32 also leaves headroom beneath GCC11 TSan's64-held-lock detector limit.
  for (size_t first = 0; first < pages.size(); first += 32) {
    size_t n = std::min<size_t>(32, pages.size() - first);
    std::vector<std::shared_lock<std::shared_mutex> > held;
    held.reserve(n);
    std::vector<IORequest> requests;
    requests.reserve(n);
    for (size_t j = 0; j < n; ++j) {
      auto page = pages[first + j];
      auto tick = now_ns();
      held.emplace_back(impl_->locks[f][page]);
      stats.page_lock_wait_ns += now_ns() - tick;
      require(page <
                  impl_->initialized_pages[f].load(std::memory_order_acquire),
              "batch beyond EOF");
      versions[first + j] = impl_->versions[f][page].load();
      requests.emplace_back(
          page * page_bytes, page_bytes,
          reinterpret_cast<char *>(buffers[first + j]->data()), 0, 0);
    }
    impl_->reader.read_fd(impl_->fd[f], requests, impl_->reader.get_ctx());
    stats.read_pages[f] += n;
    ++stats.read_batches;
    stats.max_read_batch = std::max<uint64_t>(stats.max_read_batch, n);
  }
  return versions;
}
bool PageStore::compare_write(File file, uint64_t page, uint64_t expected,
                              Page &buffer,
                              const std::function<bool()> &reserve,
                              const std::function<void()> &commit,
                              IOStats &stats) {
  unsigned f = impl_->checked(file, page);
  auto tick = now_ns();
  std::unique_lock<std::shared_mutex> held(impl_->locks[f][page]);
  stats.page_lock_wait_ns += now_ns() - tick;
  if (impl_->versions[f][page].load() != expected)
    return false;
  if (!reserve()) {
    ++stats.degree_reservation_retries;
    return false;
  }
  std::vector<IORequest> requests;
  requests.emplace_back(page * page_bytes, page_bytes,
                        reinterpret_cast<char *>(buffer.data()), 0, 0);
  impl_->reader.write_fd(impl_->fd[f], requests, impl_->reader.get_ctx());
  ++stats.write_pages[f];
  ++stats.write_batches;
  auto size = impl_->initialized_pages[f].load();
  while (size < page + 1 &&
         !impl_->initialized_pages[f].compare_exchange_weak(size, page + 1)) {
  }
  commit();
  impl_->versions[f][page].fetch_add(1, std::memory_order_release);
  return true;
}
void PageStore::sync() {
  for (unsigned f = 0; f < 2; ++f)
    require(::fsync(impl_->fd[f]) == 0, "fsync snapshot");
}

AtomicBitmap::AtomicBitmap(uint32_t capacity) : capacity_(capacity) {
  require(capacity > 0, "zero bitmap capacity");
  bits_ = std::make_unique<std::atomic<uint64_t> []>(words());
  for (uint64_t i = 0; i < words(); ++i)
    bits_[i].store(0, std::memory_order_relaxed);
}
bool AtomicBitmap::test(uint32_t id) const {
  require(id < capacity_, "bitmap ID outside max-points");
  return (bits_[id / 64].load(std::memory_order_acquire) >> (id % 64)) & 1;
}
void AtomicBitmap::set(uint32_t id) {
  require(id < capacity_, "bitmap ID outside max-points");
  bits_[id / 64].fetch_or(uint64_t(1) << (id % 64), std::memory_order_release);
}
std::unique_ptr<uint64_t[]> AtomicBitmap::snapshot() const {
  auto out = std::make_unique<uint64_t[]>(words());
  for (uint64_t i = 0; i < words(); ++i)
    out[i] = bits_[i].load(std::memory_order_acquire);
  return out;
}
} // namespace hvs_update
