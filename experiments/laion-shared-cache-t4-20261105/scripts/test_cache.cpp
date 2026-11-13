#include "shared_topo_reader.h"
#include <iostream>
#include <omp.h>
static void check(bool b, const char *s) {
  if (!b)
    throw std::runtime_error(s);
}
static void verify(uint8_t *p, unsigned id) {
  for (unsigned j = 0; j < 8192; ++j)
    check(p[j] == uint8_t(id * 17 + j / 4096), "8 KiB content mismatch");
}
static SharedTopoReader::Observation get(SharedTopoReader &r,
                                         const std::vector<unsigned> &ids) {
  uint8_t *p = nullptr;
  check(!posix_memalign((void **)&p, 8192,
                        std::max(size_t(1), ids.size()) * 8192),
        "aligned allocation");
  std::vector<IORequest> req;
  for (size_t i = 0; i < ids.size(); ++i)
    req.emplace_back(ids[i] * 8192ULL, 8192, p + i * 8192, ids[i] * 8192ULL,
                     8192);
  r.begin();
  auto ctx = r.get_ctx();
  r.send_io(req, ctx, false, r.topo_file_desc);
  for (size_t i = 0; i < req.size(); ++i)
    r.poll_wait(ctx);
  auto o = r.end();
  for (size_t i = 0; i < ids.size(); ++i)
    verify(p + i * 8192, ids[i]);
  free(p);
  return o;
}
int main(int argc, char **argv) {
  try {
    check(argc == 2, "fixture path required");
    int fd = ::open(argv[1], O_CREAT | O_EXCL | O_WRONLY, 0600);
    check(fd >= 0, "fresh owned fixture required");
    std::vector<uint8_t> p(8192);
    for (unsigned i = 0; i < 8; ++i) {
      for (unsigned j = 0; j < 8192; ++j)
        p[j] = uint8_t(i * 17 + j / 4096);
      check(::write(fd, p.data(), p.size()) == 8192, "fixture write");
    }
    check(!fsync(fd), "fixture fsync");
    ::close(fd);
    SharedTopoReader r;
    r.strategy = 0;
    r.open(argv[1], false, false, false);
    r.topo_file_desc = r.file_desc;
    r.configure(2 * 8192);
    check(get(r, { 0 }).backend_pages == 1, "cold miss");
    check(get(r, { 0 }).hits == 1, "cross-query hit");
    auto mixed = get(r, { 0, 1 });
    check(mixed.hits == 1 && mixed.backend_pages == 1, "mixed hit/miss batch");
    check(get(r, {}).requested == 0, "empty batch");
    get(r, { 2 });
    get(r, { 3 });
    check(get(r, { 0 }).backend_pages == 1, "bounded LRU eviction");
    r.configure(0);
    get(r, { 0 });
    check(get(r, { 0 }).backend_pages == 1, "disabled cache persists no pages");
    r.configure(2 * 8192);
    omp_set_num_threads(4);
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < 400; ++i) {
      auto o = get(r, { unsigned(i % 8), unsigned((i + 1) % 8) });
      check(o.requested == 2 && o.hits + o.backend_pages == 2,
            "four-thread observation isolation");
    }
    r.clear_cache();
    check(get(r, { 7 }).backend_pages == 1, "explicit cold reset");
    check(get(r, { 7 }).hits == 1, "high half of cached page preserved");
    std::cout << "SHARED_TOPO_CACHE_TEST_PASS: cold, warm, mixed, eviction, "
                 "disabled, empty, 4 threads, full 8 KiB content" << std::endl;
    std::exit(0);
  }
  catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 2;
  }
}
