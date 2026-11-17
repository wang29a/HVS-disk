#include "update_store.h"
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
using namespace hvs_update;
static void check(bool x, const char *s) {
  if (!x)
    throw std::runtime_error(s);
}
int main(int argc, char **argv) {
  try {
    check(argc == 2, "topology_interleave workdir");
    std::string root = argv[1];
    for (auto name : { "topology", "coordinates" }) {
      std::ofstream f(root + "/" + name, std::ios::binary);
      std::string p(page_bytes, 0);
      f.write(p.data(), p.size());
      check(bool(f), "create page");
    }
    PageStore store(root + "/topology", root + "/coordinates", 1, 1);
    IOStats initial;
    std::array<unsigned char, 164> old{}, replacement{};
    auto put = [](unsigned char *p, uint32_t v) { std::memcpy(p, &v, 4); };
    put(old.data(), 40);
    for (unsigned j = 0; j < 40; ++j)
      put(old.data() + 4 + 4 * j, j + 1);
    put(replacement.data(), 2);
    put(replacement.data() + 4, 41);
    put(replacement.data() + 8, 42);
    store.modify(File::topology, 0, [&](unsigned char *p) {
                                      std::memcpy(p, old.data(), old.size());
                                    },
                 initial);
    Page before, after;
    IOStats read, write;
    store.read(File::topology, 0, before, read);
    std::mutex mutex;
    std::condition_variable cv;
    bool half = false, finish = false;
    std::atomic<bool> entered{ false }, done{ false };
    std::exception_ptr writer_error, reader_error;
    std::thread writer([&] {
      try {
        store.modify(File::topology, 0,
                     [&](unsigned char *p) {
                       std::memcpy(p, replacement.data(), 82);
                       std::unique_lock<std::mutex> g(mutex);
                       half = true;
                       cv.notify_all();
                       cv.wait(g, [&] { return finish; });
                       std::memcpy(p + 82, replacement.data() + 82, 82);
                     },
                     write);
      }
      catch (...) {
        writer_error = std::current_exception();
        std::lock_guard<std::mutex> g(mutex);
        half = true;
        cv.notify_all();
      }
    });
    {
      std::unique_lock<std::mutex> g(mutex);
      cv.wait(g, [&] { return half; });
    }
    std::thread reader([&] {
      try {
        Page warm;
        IOStats local;
        store.read(File::coordinates, 0, warm, local);
        entered = true;
        store.read(File::topology, 0, after, read);
        done = true;
      }
      catch (...) {
        reader_error = std::current_exception();
        entered = true;
      }
    });
    while (!entered)
      std::this_thread::yield();
    // An independent file at the same page number must complete while the
    // topology writer holds its lock. No shared query cache is involved.
    Page coordinate;
    IOStats other;
    store.read(File::coordinates, 0, coordinate, other);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    bool blocked = !done;
    {
      std::lock_guard<std::mutex> g(mutex);
      finish = true;
    }
    cv.notify_all();
    writer.join();
    reader.join();
    if (writer_error)
      std::rethrow_exception(writer_error);
    if (reader_error)
      std::rethrow_exception(reader_error);
    check(blocked, "query page copy bypassed writer lock");
    check(std::memcmp(before.data(), old.data(), 164) == 0,
          "old query-private topology mutated");
    check(std::memcmp(after.data(), replacement.data(), 164) == 0,
          "mixed topology record");
    uint32_t degree;
    std::memcpy(&degree, after.data(), 4);
    check(degree == 2, "new topology degree");
    for (unsigned j = 0; j < degree; ++j) {
      uint32_t id;
      std::memcpy(&id, after.data() + 4 + j * 4, 4);
      check(id == 41 + j, "new topology neighbors");
    }
    for (unsigned j = 4 + degree * 4; j < 164; ++j)
      check(after.data()[j] == 0, "unused record slot not zero");
    std::cout << "TOPOLOGY_INTERLEAVE_PASS old_degree=40 new_degree=2 "
                 "query_lock_wait_ns=" << read.page_lock_wait_ns << std::endl;
  }
  catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
