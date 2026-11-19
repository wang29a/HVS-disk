#include "update_store.h"
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>

using namespace hvs_update;
static void check(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
int main(int argc, char **argv) {
  try {
    check(argc == 2, "store_smoke work-directory");
    std::string top = std::string(argv[1]) + "/smoke.topology";
    std::string coord = std::string(argv[1]) + "/smoke.coordinates";
    for (const auto &p : { top, coord }) {
      std::ofstream out(p, std::ios::binary | std::ios::trunc);
      std::string zeros(page_bytes, '\0');
      out.write(zeros.data(), zeros.size());
      check(bool(out), "create smoke page");
    }
    PageStore store(top, coord, 2, 2);
    IOStats write_stats, read_stats;
    Page old, observed;
    store.modify(File::topology, 0,
                 [](unsigned char *p) { std::memset(p, 0x11, page_bytes); },
                 write_stats);
    store.read(File::topology, 0, old, read_stats);
    std::mutex mu;
    std::condition_variable cv;
    bool half = false, finish = false;
    std::atomic<bool> reader_started{ false }, reader_finished{ false };
    std::thread writer([&] {
      store.modify(File::topology, 0,
                   [&](unsigned char *p) {
                     std::memset(p, 0x22, page_bytes / 2);
                     std::unique_lock<std::mutex> lk(mu);
                     half = true;
                     cv.notify_all();
                     cv.wait(lk, [&] { return finish; });
                     std::memset(p + page_bytes / 2, 0x22, page_bytes / 2);
                   },
                   write_stats);
    });
    {
      std::unique_lock<std::mutex> lk(mu);
      cv.wait(lk, [&] { return half; });
    }
    std::thread reader([&] {
      reader_started.store(true);
      store.read(File::topology, 0, observed, read_stats);
      reader_finished.store(true);
    });
    while (!reader_started.load())
      std::this_thread::yield();
    // The coordinate file uses a separate lock at the same physical page ID.
    IOStats other;
    Page coordinate_page;
    store.read(File::coordinates, 0, coordinate_page, other);
    bool blocked = !reader_finished.load();
    {
      std::lock_guard<std::mutex> lk(mu);
      finish = true;
    }
    cv.notify_all();
    writer.join();
    reader.join();
    check(blocked, "topology read bypassed writer lock");
    for (uint64_t i = 0; i < page_bytes; ++i) {
      check(old.data()[i] == 0x11, "old private page mutated");
      check(observed.data()[i] == 0x22, "reader observed partial page");
    }
    store.modify(File::topology, 1, [](unsigned char *p) { p[0] = 0x33; },
                 write_stats);
    store.read(File::topology, 1, observed, read_stats);
    check(observed.data()[0] == 0x33, "append page lost");
    for (uint64_t i = 1; i < page_bytes; ++i)
      check(observed.data()[i] == 0, "append padding not zero");
    bool rejected = false;
    try {
      store.read(File::topology, 2, observed, read_stats);
    }
    catch (const std::exception &) {
      rejected = true;
    }
    check(rejected, "out-of-capacity page accepted");
    AtomicBitmap bits(130);
    bits.set(0);
    bits.set(64);
    bits.set(129);
    auto frozen = bits.snapshot();
    bits.set(65);
    check(bits.test(65) && !(frozen[1] & 2), "snapshot changed after deletion");
    check(bits.test(0) && bits.test(64) && bits.test(129), "bitmap boundary");
    store.sync();
    std::cout << "G0_STORE_SMOKE_PASS lock_wait_ns="
              << read_stats.page_lock_wait_ns << "\n";
    return 0;
  }
  catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
