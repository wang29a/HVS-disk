#pragma once
#include "linux_aligned_file_reader.h"
#include <chrono>
#include <stdexcept>

namespace io_measure {
using Clock = std::chrono::steady_clock;
using U = uint64_t;
inline double elapsed(Clock::time_point t) {
  return std::chrono::duration<double, std::micro>(Clock::now() - t).count();
}
enum class Kind {
  Coupled,
  Topology,
  Coordinate
};
struct Observation {
  bool active = false, coupled = false;
  U depth = 0, pending = 0;
  Kind pending_kind = Kind::Topology;
  double coupled_reader_us = 0, coupled_backend_us = 0;
  double topology_submit_us = 0, topology_wait_us = 0, topology_blocking_us = 0;
  double coordinate_submit_us = 0, coordinate_wait_us = 0,
         coordinate_blocking_us = 0;
  U coupled_requested = 0, coupled_backend_requests = 0,
    topology_backend_requests = 0, coordinate_backend_requests = 0;
  U coupled_backend_bytes = 0, topology_backend_bytes = 0,
    coordinate_backend_bytes = 0;
  double topology_us() const {
    return topology_submit_us + topology_wait_us + topology_blocking_us;
  }
  double coordinate_us() const {
    return coordinate_submit_us + coordinate_wait_us + coordinate_blocking_us;
  }
  double total_us() const {
    return coupled_reader_us + topology_us() + coordinate_us();
  }
};
inline thread_local Observation obs;

// Only observes public reader calls; the unchanged d2a39e2 library still
// performs
// all cache lookup, submission, waiting, parsing, traversal and distance work.
class Reader : public LinuxAlignedFileReader {
public:
  explicit Reader(bool enabled = true) : enabled_(enabled) {}
  void begin(unsigned mode) {
    obs = Observation{};
    obs.active = enabled_;
    obs.coupled = (mode == 0);
  }
  Observation end() {
    if (obs.depth || obs.pending)
      throw std::runtime_error("reader observation has unfinished operations");
    Observation value = obs;
    obs.active = false;
    return value;
  }
  void read_alloc(std::vector<IORequest> &reqs, void *ctx,
                  std::vector<uint64_t> *refs = nullptr) override {
    if (!obs.active) {
      LinuxAlignedFileReader::read_alloc(reqs, ctx, refs);
      return;
    }
    if (!obs.coupled || obs.depth)
      throw std::runtime_error("unexpected nested/topology read_alloc");
    obs.coupled_requested += reqs.size();
    obs.depth++;
    auto t = Clock::now();
    LinuxAlignedFileReader::read_alloc(reqs, ctx, refs);
    obs.coupled_reader_us += elapsed(t);
    obs.depth--;
  }
  void read(std::vector<IORequest> &reqs, void *ctx,
            bool async = false) override {
    if (!obs.active) {
      LinuxAlignedFileReader::read(reqs, ctx, async);
      return;
    }
    if (async)
      throw std::runtime_error("unexpected asynchronous blocking read");
    Kind k = classify(file_desc);
    backend(k, reqs);
    bool root = obs.depth == 0;
    obs.depth++;
    auto t = Clock::now();
    LinuxAlignedFileReader::read(reqs, ctx, async);
    double duration = elapsed(t);
    obs.depth--;
    if (k == Kind::Coupled) {
      obs.coupled_backend_us += duration;
      if (root)
        obs.coupled_reader_us += duration;
    } else
      blocking_time(k, duration);
  }
  void read_fd(int fd, std::vector<IORequest> &reqs, void *ctx) override {
    if (!obs.active) {
      LinuxAlignedFileReader::read_fd(fd, reqs, ctx);
      return;
    }
    if (obs.depth || obs.pending)
      throw std::runtime_error("overlapping/nested read_fd not covered");
    Kind k = classify(fd);
    backend(k, reqs);
    obs.depth++;
    auto t = Clock::now();
    LinuxAlignedFileReader::read_fd(fd, reqs, ctx);
    double duration = elapsed(t);
    obs.depth--;
    if (k == Kind::Coupled) {
      obs.coupled_reader_us += duration;
      obs.coupled_backend_us += duration;
    } else
      blocking_time(k, duration);
  }
  void send_io(std::vector<IORequest> &reqs, void *ctx, bool write,
               FileHandle fd) override {
    if (!obs.active) {
      LinuxAlignedFileReader::send_io(reqs, ctx, write, fd);
      return;
    }
    send_before(fd, write, reqs);
    auto t = Clock::now();
    LinuxAlignedFileReader::send_io(reqs, ctx, write, fd);
    send_time(elapsed(t));
  }
  void send_io(std::vector<IORequest> &reqs, void *ctx, bool write) override {
    if (!obs.active) {
      LinuxAlignedFileReader::send_io(reqs, ctx, write);
      return;
    }
    send_before(file_desc, write, reqs);
    auto t = Clock::now();
    LinuxAlignedFileReader::send_io(reqs, ctx, write);
    send_time(elapsed(t));
  }
  void send_io(IORequest &req, void *ctx, bool write,
               FileHandle fd = -1) override {
    if (!obs.active) {
      LinuxAlignedFileReader::send_io(req, ctx, write, fd);
      return;
    }
    std::vector<IORequest> items{ req };
    send_before(fd < 0 ? file_desc : fd, write, items);
    auto t = Clock::now();
    LinuxAlignedFileReader::send_io(req, ctx, write, fd);
    send_time(elapsed(t));
  }
  void poll_wait(void *ctx) override {
    if (!obs.active) {
      LinuxAlignedFileReader::poll_wait(ctx);
      return;
    }
    if (!obs.pending)
      throw std::runtime_error("poll without observed submission");
    auto t = Clock::now();
    LinuxAlignedFileReader::poll_wait(ctx);
    double duration = elapsed(t);
    if (obs.pending_kind == Kind::Topology)
      obs.topology_wait_us += duration;
    else
      obs.coordinate_wait_us += duration;
    obs.pending--;
  }

private:
  bool enabled_;
  Kind classify(int fd) {
    if (obs.coupled && fd == file_desc)
      return Kind::Coupled;
    if (fd >= 0 && fd == coord_file_desc)
      return Kind::Coordinate;
    if (fd >= 0 && fd == topo_file_desc)
      return Kind::Topology;
    throw std::runtime_error("unclassified query descriptor");
  }
  static void backend(Kind kind, const std::vector<IORequest> &reqs) {
    U bytes = 0;
    for (auto &r : reqs)
      bytes += r.len;
    if (kind == Kind::Coupled) {
      obs.coupled_backend_requests += reqs.size();
      obs.coupled_backend_bytes += bytes;
    } else if (kind == Kind::Topology) {
      obs.topology_backend_requests += reqs.size();
      obs.topology_backend_bytes += bytes;
    } else {
      obs.coordinate_backend_requests += reqs.size();
      obs.coordinate_backend_bytes += bytes;
    }
  }
  static void blocking_time(Kind kind, double duration) {
    if (kind == Kind::Topology)
      obs.topology_blocking_us += duration;
    else
      obs.coordinate_blocking_us += duration;
  }
  void send_before(int fd, bool write, const std::vector<IORequest> &reqs) {
    if (write || obs.depth || reqs.size() > 1024)
      throw std::runtime_error("unsupported query submission");
    Kind kind = classify(fd);
    if (kind == Kind::Coupled)
      throw std::runtime_error("unexpected async coupled path");
    if (obs.pending && kind != obs.pending_kind)
      throw std::runtime_error("mixed phase asynchronous requests");
    obs.pending_kind = kind;
    obs.pending += reqs.size();
    backend(kind, reqs);
  }
  static void send_time(double duration) {
    if (obs.pending_kind == Kind::Topology)
      obs.topology_submit_us += duration;
    else
      obs.coordinate_submit_us += duration;
  }
};
} // namespace io_measure
