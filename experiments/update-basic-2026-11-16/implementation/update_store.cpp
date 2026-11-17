#include "update_store.h"
#include "linux_aligned_file_reader.h"
#include <chrono>
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
    struct stat s;
    require(::fstat(fd[f], &s) == 0 &&
                uint64_t(s.st_size) >= (page + 1) * page_bytes,
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
  return impl_->versions[f][page].load(std::memory_order_relaxed);
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
  struct stat s;
  require(::fstat(impl_->fd[f], &s) == 0, "fstat page store");
  if (uint64_t(s.st_size) >= (page + 1) * page_bytes) {
    impl_->read_locked(f, page, private_page);
    ++stats.read_pages[f];
  }
  // New append pages start completely zeroed, including their unused slots.
  // No read is issued beyond EOF. pwrite extends in exact page units.
  change(private_page.data());
  std::vector<IORequest> requests;
  requests.emplace_back(page * page_bytes, page_bytes,
                        reinterpret_cast<char *>(private_page.data()), 0, 0);
  impl_->reader.write_fd(impl_->fd[f], requests, impl_->reader.get_ctx());
  ++stats.write_pages[f];
  impl_->versions[f][page].fetch_add(1, std::memory_order_release);
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
