#include "measured_reader.h"
#include <cstring>
#include <fstream>
#include <iostream>
#include <omp.h>
using namespace io_measure;
static void check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
static std::vector<IORequest> requests(uint8_t *buf, unsigned count = 3) {
  std::vector<IORequest> result;
  for (unsigned i = 0; i < count; i++)
    result.emplace_back(i * SECTOR_LEN, SECTOR_LEN, buf + i * SECTOR_LEN,
                        i * SECTOR_LEN, SECTOR_LEN);
  return result;
}
static void verify(uint8_t *buf, unsigned count = 3) {
  for (unsigned i = 0; i < count; i++)
    for (unsigned j = 0; j < SECTOR_LEN; j++)
      check(buf[i * SECTOR_LEN + j] == uint8_t(i + 3), "page content changed");
}
int main(int argc, char **argv) {
  try {
    check(argc == 2, "unit needs owned fixture path");
    uint8_t *memory = nullptr;
    check(!posix_memalign((void **)&memory, SECTOR_LEN, SECTOR_LEN * 3),
          "aligned memory");
    for (unsigned i = 0; i < 3; i++)
      memset(memory + i * SECTOR_LEN, i + 3, SECTOR_LEN);
    int fd = ::open(argv[1], O_CREAT | O_TRUNC | O_RDWR, 0600);
    check(fd >= 0, "fixture open");
    check(::write(fd, memory, SECTOR_LEN * 3) == SECTOR_LEN * 3,
          "fixture write");
    check(!fsync(fd), "fixture flush");
    ::close(fd);
    Reader reader;
    reader.strategy = 0;
    reader.open(argv[1], false, false, false);
    void *ctx = reader.get_ctx();
    auto req = requests(memory);
    std::vector<uint64_t> refs1, refs2;
    reader.begin(0);
    reader.read_alloc(req, ctx, &refs1);
    auto first = reader.end();
    verify(memory);
    check(first.coupled_requested == 3 && first.coupled_backend_requests == 3,
          "cold baseline counters");
    check(first.coupled_reader_us >= first.coupled_backend_us &&
              first.total_us() == first.coupled_reader_us,
          "baseline nested timing counted twice");
    reader.begin(0);
    reader.read_alloc(req, ctx, &refs2);
    auto cached = reader.end();
    verify(memory);
    check(cached.coupled_requested == 3 && cached.coupled_backend_requests == 0,
          "reader cache-hit counters");
    reader.deref(&refs1, ctx);
    reader.deref(&refs2, ctx);
    check(v2::cache.cache.size() == 0, "cache refs not released");
    reader.topo_file_desc = reader.file_desc;
    reader.coord_file_desc = ::dup(reader.file_desc);
    auto call = Clock::now();
    reader.begin(6);
    reader.send_io(req, ctx, false, reader.topo_file_desc);
    for (unsigned i = 0; i < 3; i++)
      reader.poll_wait(ctx);
    auto coord = requests(memory, 2);
    reader.read_fd(reader.coord_file_desc, coord, ctx);
    auto split = reader.end();
    verify(memory, 2);
    check(split.topology_backend_requests == 3 &&
              split.coordinate_backend_requests == 2,
          "split request accounting");
    check(split.topology_backend_bytes == 3 * SECTOR_LEN &&
              split.coordinate_backend_bytes == 2 * SECTOR_LEN,
          "split byte accounting");
    check(split.total_us() <= elapsed(call) && split.coupled_reader_us == 0,
          "split timer overlap");
    reader.begin(6);
    bool rejected = false;
    try {
      reader.poll_wait(ctx);
    }
    catch (const std::exception &) {
      rejected = true;
    }
    reader.end();
    check(rejected, "orphan wait not rejected");
    unsigned failures = 0;
#pragma omp parallel num_threads(4) reduction(+ : failures)
    {
      uint8_t *b = nullptr;
      posix_memalign((void **)&b, SECTOR_LEN, SECTOR_LEN * 3);
      auto rs = requests(b);
      auto c = reader.get_ctx();
      std::vector<uint64_t> refs;
      reader.begin(0);
      reader.read_alloc(rs, c, &refs);
      auto o = reader.end();
      try {
        verify(b);
        check(o.coupled_requested == 3 && o.coupled_backend_requests <= 3,
              "thread-local accounting");
      }
      catch (...) {
        failures++;
      }
      reader.deref(&refs, c);
      free(b);
    }
    check(!failures, "concurrent reader/TLS test");
    check(v2::cache.cache.size() == 0, "concurrent cache refs");
    Reader disabled(false);
    disabled.strategy = 0;
    disabled.open(argv[1], false, false, false);
    auto offctx = disabled.get_ctx();
    auto offreq = requests(memory);
    std::vector<uint64_t> offrefs;
    disabled.begin(0);
    disabled.read_alloc(offreq, offctx, &offrefs);
    auto off = disabled.end();
    verify(memory);
    check(off.total_us() == 0 && off.coupled_requested == 0 &&
              off.coupled_backend_requests == 0,
          "disabled observer");
    disabled.deref(&offrefs, offctx);
    std::cout << "READER_UNIT_PASS cache-miss/hit, no-double-counting, "
                 "submit/wait, coord read, TLS4, disabled observer"
              << std::endl;
    std::exit(0);
  }
  catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
