#include "in_degree_guard.h"
#include <iostream>
#include <mutex>
#include <thread>
using namespace hvs_update;
static void check(bool b, const char *s) {
  if (!b)
    throw std::runtime_error(s);
}
int main() {
  try {
    auto alive = [](uint32_t) { return true; };
    auto short_row = protect_last_incoming(
        std::vector<uint32_t>{ 1, 2 }, std::vector<uint32_t>{ 2 }, 3,
        [](uint32_t v) { return v == 1 ? 1u : 2u; }, alive);
    check(short_row.ids == std::vector<uint32_t>({ 2, 1 }) &&
              short_row.old_restored == 1,
          "spare-slot protection");
    auto full = protect_last_incoming(
        std::vector<uint32_t>{ 1, 2, 3 }, std::vector<uint32_t>{ 2, 3, 4 }, 3,
        [](uint32_t v) { return v == 1 || v == 3 ? 1u : 2u; }, alive);
    check(full.ids == std::vector<uint32_t>({ 2, 3, 1 }),
          "replace last-ranked removable edge");
    auto skip = protect_last_incoming(
        std::vector<uint32_t>{ 1, 2, 3 }, std::vector<uint32_t>{ 2, 4, 3 }, 3,
        [](uint32_t v) { return v == 1 || v == 3 ? 1u : 2u; }, alive);
    check(skip.ids == std::vector<uint32_t>({ 2, 1, 3 }),
          "do not evict another sole incoming edge");
    auto pending = protect_last_incoming(
        std::vector<uint32_t>{ 1, 2 }, std::vector<uint32_t>{ 1 }, 3,
        [](uint32_t v) { return v == 9 ? 0u : 2u; }, alive, 9);
    check(pending.ids == std::vector<uint32_t>({ 1, 9 }) &&
              pending.new_restored == 1,
          "publishable first reverse edge");
    auto conflict = protect_last_incoming(
        std::vector<uint32_t>{ 1, 2 }, std::vector<uint32_t>{ 9, 2 }, 2,
        [](uint32_t v) { return v == 9 ? 0u : 1u; }, alive, 9);
    check(conflict.capacity_conflict &&
              conflict.ids == std::vector<uint32_t>({ 2, 1 }),
          "strict R wins when every existing edge is mandatory");
    // A deterministic two-writer removal race: the later decision sees the
    // earlier completed write, so two concurrent removals cannot both succeed.
    std::mutex ledger;
    unsigned count = 2, removed = 0, protected_rows = 0;
    auto writer = [&] {
      std::lock_guard<std::mutex> g(ledger);
      auto p = protect_last_incoming(std::vector<uint32_t>{ 7 },
                                     std::vector<uint32_t>{}, 1,
                                     [&](uint32_t) { return count; }, alive);
      if (p.ids.empty()) {
        --count;
        ++removed;
      } else
        ++protected_rows;
    };
    std::thread a(writer), b(writer);
    a.join();
    b.join();
    check(count == 1 && removed == 1 && protected_rows == 1,
          "concurrent last-incoming race");
    std::cout << "IN_DEGREE_GUARD_PASS spare=1 full=1 last_mandatory=1 "
                 "pending=1 conflict=1 concurrent=1\n";
  }
  catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
