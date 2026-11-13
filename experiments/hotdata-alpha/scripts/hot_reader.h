#pragma once

#include "linux_aligned_file_reader.h"
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// Experiment-only backend. The d2a39e2 graph/PQ/rerank library is unchanged.
// Memory requests copy the same aligned byte ranges to the original scratch.
class HotReader : public LinuxAlignedFileReader {
  struct Mapping {
    const char *bytes = nullptr;
    size_t size = 0;
    void load(const std::string &path, int original_fd) {
      int fd = ::open(path.c_str(), O_RDONLY);
      struct stat a {
      }, b{};
      if (fd < 0 || fstat(fd, &a) || fstat(original_fd, &b) ||
          a.st_size != b.st_size)
        throw std::runtime_error("Memory image missing or size mismatch: " +
                                 path);
      size = a.st_size;
      void *p = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
      ::close(fd);
      if (p == MAP_FAILED)
        throw std::runtime_error("mmap failed: " + path);
      bytes = static_cast<const char *>(p);
      // Explicit untimed prefault and residency check, one thread.
      volatile unsigned checksum = 0;
      for (size_t offset = 0; offset < size; offset += 4096)
        checksum += static_cast<unsigned char>(bytes[offset]);
      std::vector<unsigned char> resident((size + 4095) / 4096);
      if (mincore(p, size, resident.data()))
        throw std::runtime_error("mincore failed");
      for (auto v : resident)
        if (!(v & 1))
          throw std::runtime_error("Memory image is not resident");
      std::cout << "HVS_RESIDENT " << path << " bytes=" << size << std::endl;
    }
    ~Mapping() {
      if (bytes)
        munmap(const_cast<char *>(bytes), size);
    }
  } topo_, coord_;

public:
  struct alignas(64) Counters {
    uint64_t memory_topology = 0, memory_coordinate = 0;
    uint64_t ssd_topology = 0, ssd_coordinate = 0;
    uint64_t memory_bytes = 0, pending = 0;
  };
  std::array<Counters, 4> counters{};
  std::string backend = "ssd";
  bool ready = false;

  Counters &slot() {
    unsigned t = omp_in_parallel() ? omp_get_thread_num() : 0;
    if (t >= counters.size())
      throw std::runtime_error("Thread cap exceeded");
    return counters[t];
  }
  void attach(const std::string &mode, const std::string &dir,
              const std::string &scheme) {
    if (mode != "ssd" && mode != "coord-hot" && mode != "all-hot")
      throw std::runtime_error("Unknown backend");
    backend = mode;
    if (backend != "ssd")
      coord_.load(dir + "/coord.bin", coord_file_desc);
    if (backend == "all-hot")
      topo_.load(dir + "/" + scheme + "-topo.bin", topo_file_desc);
    if (coord_.size + topo_.size > (uint64_t(64) << 30))
      throw std::runtime_error("Equal 64 GiB image budget exceeded");
    ready = true;
    reset_counters();
  }
  void reset_counters() {
    for (auto &s : counters) {
      if (s.pending)
        throw std::runtime_error("Unconsumed completion");
      s = Counters{};
    }
  }
  Counters totals() const {
    Counters total{};
    for (const auto &s : counters) {
      if (s.pending)
        throw std::runtime_error("Outstanding memory completion");
      total.memory_topology += s.memory_topology;
      total.memory_coordinate += s.memory_coordinate;
      total.ssd_topology += s.ssd_topology;
      total.ssd_coordinate += s.ssd_coordinate;
      total.memory_bytes += s.memory_bytes;
    }
    return total;
  }
  void copy(const Mapping &image, IORequest &r, bool topology) {
    if (r.offset > image.size || r.len > image.size - r.offset)
      throw std::runtime_error("Memory request outside image");
    memcpy(r.buf, image.bytes + r.offset, r.len);
    r.finished = true;
    auto &s = slot();
    if (topology)
      ++s.memory_topology;
    else
      ++s.memory_coordinate;
    s.memory_bytes += r.len;
  }
  void read_fd(int fd, std::vector<IORequest> &reqs, void *ctx) override {
    if (!ready)
      return LinuxAlignedFileReader::read_fd(fd, reqs, ctx);
    if (fd != coord_file_desc)
      throw std::runtime_error("Unexpected blocking FD");
    if (backend != "ssd") {
      for (auto &r : reqs)
        copy(coord_, r, false);
    } else {
      slot().ssd_coordinate += reqs.size();
      LinuxAlignedFileReader::read_fd(fd, reqs, ctx);
    }
  }
  void send_io(std::vector<IORequest> &reqs, void *ctx, bool write,
               FileHandle fd) override {
    if (!ready)
      return LinuxAlignedFileReader::send_io(reqs, ctx, write, fd);
    if (write || fd != topo_file_desc)
      throw std::runtime_error("Unexpected async request");
    if (backend == "all-hot") {
      for (auto &r : reqs)
        copy(topo_, r, true);
      slot().pending += reqs.size();
    } else {
      slot().ssd_topology += reqs.size();
      LinuxAlignedFileReader::send_io(reqs, ctx, false, fd);
    }
  }
  void poll_wait(void *ctx) override {
    if (!ready || backend != "all-hot")
      return LinuxAlignedFileReader::poll_wait(ctx);
    auto &s = slot();
    if (!s.pending)
      throw std::runtime_error("Missing memory completion");
    --s.pending;
  }
};
