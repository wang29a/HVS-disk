#include "in_degree_ledger.h"
#include "in_degree_guard.h"
#include <thread>
#include <iostream>
using namespace hvs_update;
static void check(bool b) {
  if (!b)
    throw std::runtime_error("atomic degree check");
}
int main() {
  try {
    AtomicDegreeLedger ledger(4);
    ledger.add_committed(0, 2);
    ledger.add_committed(1);
    {
      AtomicDegreeLedger::Transaction a(ledger);
      a.change(0, -1);
      a.change(1, -1);
      check(!a.reserve([](uint32_t) { return true; }));
      check(ledger.count(0) == 2 && ledger.count(1) == 1);
    }
    {
      AtomicDegreeLedger::Transaction a(ledger);
      a.change(2, 1);
      check(a.reserve([](uint32_t) { return true; }));
      check(ledger.count(2) == 0);
      a.commit();
      check(ledger.count(2) == 1);
    }
    {
      AtomicDegreeLedger::Transaction a(ledger);
      a.change(0, -1);
      check(a.reserve([](uint32_t) { return true; }));
      check(ledger.count(0) == 1);
    }
    check(ledger.count(0) == 2);
    for (unsigned rep = 0; rep < 1000; ++rep) {
      if (rep)
        ledger.add_committed(0);
      std::atomic<unsigned> ready{ 0 }, won{ 0 };
      auto worker = [&] {
        AtomicDegreeLedger::Transaction a(ledger);
        check(a.count(0) == 2);
        a.change(0, -1);
        ++ready;
        while (ready.load() != 2)
          std::this_thread::yield();
        if (a.reserve([](uint32_t) { return true; })) {
          a.commit();
          ++won;
        }
      };
      std::thread a(worker), b(worker);
      a.join();
      b.join();
      check(won == 1 && ledger.count(0) == 1);
      auto row =
          protect_last_incoming(std::vector<uint32_t>{ 0 }, {}, 40,
                                [&](uint32_t id) { return ledger.count(id); },
                                [](uint32_t) { return true; });
      check(row.ids == std::vector<uint32_t>{ 0 });
    }
    {
      AtomicDegreeLedger::Transaction a(ledger);
      a.change(1, -1);
      check(a.reserve([](uint32_t) { return false; }));
      a.commit();
      check(ledger.count(1) == 0 && ledger.positive(1));
    }
    std::cout << "ATOMIC_DEGREE_PASS races=1000 rollback=1 deferred_addition=1"
              << std::endl;
  }
  catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
