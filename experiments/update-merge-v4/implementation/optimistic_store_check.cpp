#include "update_store.h"
#include <fstream>
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace hvs_update;
static void check(bool b) {
  if (!b)
    throw std::runtime_error("optimistic store check");
}
int main(int argc, char **argv) {
  try {
    check(argc == 2);
    std::string root = argv[1];
    for (auto name : { "topo", "coord" }) {
      std::ofstream f(root + "/" + name, std::ios::binary);
      std::string p(page_bytes, 0);
      f.write(p.data(), p.size());
    }
    PageStore store(root + "/topo", root + "/coord", 2, 2);
    IOStats stats;
    Page stale, fresh;
    auto v = store.read_for_update(File::topology, 0, stale, stats);
    store.modify(File::topology, 0, [](unsigned char *p) { p[0] = 7; }, stats);
    bool called = false;
    check(!store.compare_write(File::topology, 0, v, stale, [&] {
                                                              called = true;
                                                              return true;
                                                            },
                               [] {}, stats));
    check(!called);
    v = store.read_for_update(File::topology, 0, fresh, stats);
    check(fresh.data()[0] == 7);
    fresh.data()[0] = 9;
    check(store.compare_write(File::topology, 0, v, fresh, [] { return true; },
                              [] {}, stats));
    Page append;
    std::memset(append.data(), 0x77, page_bytes);
    v = store.read_for_update(File::topology, 1, append, stats);
    check(append.data()[0] == 0);
    append.data()[0] = 11;
    check(store.compare_write(File::topology, 1, v, append, [] { return true; },
                              [] {}, stats));
    Page a, b;
    auto versions =
        store.read_many(File::topology, { 0, 1 }, { &a, &b }, stats);
    check(a.data()[0] == 9 && b.data()[0] == 11 && stats.max_read_batch == 2 &&
          versions.size() == 2);
    check(versions[0] == 2 && versions[1] == 1);
    for (auto name : { "large-topo", "large-coord" }) {
      std::ofstream f(root + "/" + name, std::ios::binary);
      std::string zeros(page_bytes, 0);
      for (unsigned i = 0; i < 96; ++i)
        f.write(zeros.data(), zeros.size());
    }
    PageStore large(root + "/large-topo", root + "/large-coord", 96, 96);
    std::vector<uint64_t> ids;
    std::vector<std::unique_ptr<Page> > owners;
    std::vector<Page *> buffers;
    for (unsigned i = 0; i < 96; ++i) {
      ids.push_back(i);
      owners.emplace_back(new Page());
      buffers.push_back(owners.back().get());
    }
    std::shared_mutex outer;
    std::shared_lock<std::shared_mutex> operation(outer);
    IOStats large_stats;
    auto large_versions =
        large.read_many(File::coordinates, ids, buffers, large_stats);
    check(large_versions.size() == 96 && large_stats.max_read_batch == 48 &&
          large_stats.read_batches == 2);
    std::cout << "OPTIMISTIC_STORE_PASS stale_version_rejected=1 append=1 "
                 "batch=2 large_pages=96 lock_batch=48" << std::endl;
  }
  catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
