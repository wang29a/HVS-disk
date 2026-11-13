#include "layout_store.h"
#include <omp.h>
using namespace layout_exp;
int main(int argc, char **argv) {
  try {
    if (argc != 6)
      throw std::runtime_error(
          "test_layout_io meta topo coord directory layout");
    Source source(argv[1], argv[2], argv[3]);
    Store store(source, argv[5], argv[4]);
    SharedPages shared(8);
    {
      Cache a(store, &shared), b(store, &shared);
      a.fetch({ 0 }, 1);
      b.fetch({ 0 }, 6);
      const bool coupled = std::string(argv[5]) == "TXY";
      if (a.count.submitted != 1 || a.count.shared_hits != 0 ||
          b.count.shared_hits != U(coupled) ||
          memcmp(b.field(0, 4), source.field(0, 4), source.bytes(4)))
        throw std::runtime_error("coordinate lookup policy");
      b.reset();
      b.fetch({ 0 }, 1);
      if (b.count.submitted != 0 || b.count.shared_hits != 1 ||
          memcmp(b.field(0, 1), source.field(0, 1), source.bytes(1)))
        throw std::runtime_error("cross-query topology reuse");
      shared.clear();
    }
    omp_set_num_threads(4);
    unsigned errors = 0;
#pragma omp parallel reduction(+ : errors)
    {
      Cache cache(store, &shared);
      for (unsigned r = 0; r < 14; r++) {
        std::vector<unsigned> ids;
        for (unsigned j = 0; j < 257; j++)
          ids.push_back((r * 257 + j + omp_get_thread_num() * 31) % source.n);
        cache.fetch(ids, 1);
        cache.fetch(ids, 6);
        for (auto id : ids)
          for (int bit : { 1, 2, 4 })
            if (memcmp(cache.field(id, bit), source.field(id, bit),
                       source.bytes(bit)))
              errors++;
      }
    }
    if (errors)
      throw std::runtime_error("read mismatch");
    std::cout << "LAYOUT_IO_TEST_PASS concurrent fields, partial pages, "
                 "request batching, LRU eviction" << std::endl;
  }
  catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}
