#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <libaio.h>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace layout_exp {
using U = uint64_t;
constexpr U PAGE = 8192, CACHE_PAGES = 2048;
using Clock = std::chrono::steady_clock;
inline double us(Clock::time_point p) {
  return std::chrono::duration<double, std::micro>(Clock::now() - p).count();
}
inline Clock::time_point phase_now() {
#ifdef LAYOUT_NO_TIMING
  return Clock::time_point{};
#else
  return Clock::now();
#endif
}
inline double phase_us(Clock::time_point p) {
#ifdef LAYOUT_NO_TIMING
  return 0;
#else
  return us(p);
#endif
}
struct Order {
  std::vector<unsigned> ids, positions;
  void init(U n) {
    ids.resize(n);
    positions.resize(n);
    std::iota(ids.begin(), ids.end(), 0);
    if (const char *path = std::getenv("LAYOUT_ORDER")) {
      std::ifstream input(path, std::ios::binary);
      U count = 0;
      input.read((char *)&count, 8);
      input.read((char *)ids.data(), n * 4);
      if (!input || count != n || input.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("order size/header");
    }
    std::vector<uint8_t> seen(n, 0);
    for (U pos = 0; pos < n; pos++) {
      unsigned id = ids[pos];
      if (id >= n || seen[id]++)
        throw std::runtime_error("order is not a permutation");
      positions[id] = pos;
    }
  }
};
inline uint32_t read32(const void *p) {
  uint32_t v;
  memcpy(&v, p, 4);
  return v;
}
inline U read64(const void *p) {
  U v;
  memcpy(&v, p, 8);
  return v;
}
struct Map {
  int fd = -1;
  U size = 0;
  const uint8_t *p = nullptr;
  explicit Map(const std::string &name) {
    fd = ::open(name.c_str(), O_RDONLY);
    struct stat s {};
    if (fd < 0 || fstat(fd, &s))
      throw std::runtime_error("open source " + name);
    size = s.st_size;
    auto b = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (b == MAP_FAILED)
      throw std::runtime_error("map source " + name);
    p = (const uint8_t *)b;
  }
  ~Map() {
    if (p)
      munmap((void *)p, size);
    if (fd >= 0)
      ::close(fd);
  }
};
struct Source {
  Map meta, topo, coord;
  U n, R, M, npp, width, d1, d2, cnpp;
  std::vector<unsigned> entries;
  Order order;
  Source(const std::string &m, const std::string &t, const std::string &c)
      : meta(m), topo(t), coord(c) {
    n = read32(meta.p);
    d1 = read32(meta.p + 4);
    d2 = read32(meta.p + 8);
    R = read32(meta.p + 12);
    M = read32(meta.p + 16);
    npp = read64(meta.p + 20);
    width = 4 + R * (4 + 2 * M);
    cnpp = PAGE / (4 * (d1 + d2));
    U ne = read32(meta.p + 28);
    if (meta.size != PAGE || !n || !cnpp || width > PAGE ||
        npp != PAGE / width || 32 + ne * 4 > PAGE)
      throw std::runtime_error("bad metadata");
    for (U i = 0; i < ne; i++) {
      auto id = read32(meta.p + 32 + 4 * i);
      if (id >= n)
        throw std::runtime_error("bad entry");
      entries.push_back(id);
    }
    if (entries.empty() || topo.size != ((n + npp - 1) / npp) * PAGE ||
        coord.size != ((n + cnpp - 1) / cnpp) * PAGE)
      throw std::runtime_error("bad source sizes");
    order.init(n);
  }
  U bytes(int bit) const { return bit == 1 ? width : (bit == 2 ? d1 : d2) * 4; }
  const uint8_t *field(U id, int bit) const {
    if (id >= n)
      throw std::runtime_error("ID range");
    if (bit == 1)
      return topo.p + (id / npp) * PAGE + (id % npp) * width;
    return coord.p + (id / cnpp) * PAGE + (id % cnpp) * 4 * (d1 + d2) +
           (bit == 4 ? d1 * 4 : 0);
  }
};
inline std::vector<int> masks(const std::string &layout) {
  if (layout == "S_TX")
    return { 3 };
  if (layout == "S_T_X")
    return { 1, 2 };
  if (layout == "S_TD")
    return { 7 };
  if (layout == "S_T_D")
    return { 1, 6 };
  if (layout == "TXY")
    return { 7 };
  if (layout == "TX_Y")
    return { 3, 4 };
  if (layout == "TY_X")
    return { 5, 2 };
  if (layout == "T_XY")
    return { 1, 6 };
  if (layout == "T_X_Y")
    return { 1, 2, 4 };
  throw std::runtime_error("unknown layout");
}
struct Group {
  int mask, fd;
  U width, npp, bytes;
  std::array<U, 3> offsets{};
};
inline std::vector<Group> groups(const Source &s, const std::string &layout,
                                 const std::string &dir, bool open_files) {
  std::vector<Group> out;
  for (int mask : masks(layout)) {
    Group g{};
    g.mask = mask;
    g.fd = -1;
    for (int k = 0; k < 3; k++)
      if (mask & (1 << k)) {
        g.offsets[k] = g.width;
        g.width += s.bytes(1 << k);
      }
    if (!g.width || g.width > PAGE)
      throw std::runtime_error(
          "overflow record not supported by this prototype");
    g.npp = PAGE / g.width;
    g.bytes = ((s.n + g.npp - 1) / g.npp) * PAGE;
    if (open_files) {
      g.fd = ::open((dir + "/group-" + std::to_string(mask) + ".bin").c_str(),
                    O_RDWR | O_DIRECT);
      struct stat st {};
      if (g.fd < 0 || fstat(g.fd, &st) || U(st.st_size) != g.bytes)
        throw std::runtime_error("layout file mismatch");
    }
    out.push_back(g);
  }
  return out;
}
inline void build_and_validate(const Source &s, const std::string &layout,
                               const std::string &dir,
                               const std::string &output) {
  auto all = groups(s, layout, dir, false);
  U total = 0, padding = 0;
  auto start = Clock::now();
  for (auto &g : all) {
    const std::string name = dir + "/group-" + std::to_string(g.mask) + ".bin";
    // A read-only coordinate group can be reused across budgets with the same
    // order. Run the complete content and padding checks in either case.
    if (!(g.mask == 6 && std::getenv("LAYOUT_REUSE_COORD"))) {
      std::vector<char> buffering(8 << 20);
      std::ofstream out;
      out.rdbuf()->pubsetbuf(buffering.data(), buffering.size());
      out.open(name, std::ios::binary | std::ios::trunc);
      if (!out)
        throw std::runtime_error("create layout");
      std::array<uint8_t, PAGE> page{};
      for (U first = 0; first < s.n; first += g.npp) {
        page.fill(0);
        for (U j = 0; j < g.npp && first + j < s.n; j++)
          for (int k = 0; k < 3; k++)
            if (g.mask & (1 << k))
              memcpy(page.data() + j * g.width + g.offsets[k],
                     s.field(s.order.ids[first + j], 1 << k), s.bytes(1 << k));
        out.write((char *)page.data(), PAGE);
      }
      out.close();
      int fd = ::open(name.c_str(), O_RDWR);
      if (fd < 0 || fsync(fd))
        throw std::runtime_error("flush layout");
      ::close(fd);
    }
    Map actual(name);
    if (actual.size != g.bytes)
      throw std::runtime_error("layout length");
    for (U id = 0; id < s.n; id++)
      for (int k = 0; k < 3; k++)
        if (g.mask & (1 << k))
          if (memcmp(actual.p + (s.order.positions[id] / g.npp) * PAGE +
                         (s.order.positions[id] % g.npp) * g.width +
                         g.offsets[k],
                     s.field(id, 1 << k), s.bytes(1 << k)))
            throw std::runtime_error("field identity mismatch");
    for (U p = 0; p < actual.size / PAGE; p++) {
      U used = std::min(g.npp, s.n - p * g.npp) * g.width;
      for (U j = used; j < PAGE; j++)
        if (actual.p[p * PAGE + j])
          throw std::runtime_error("nonzero output padding");
    }
    total += g.bytes;
    padding += g.bytes - s.n * g.width;
  }
  std::ofstream out(output);
  out << "{\"status\":\"PASS\",\"nodes\":" << s.n
      << ",\"physical_bytes\":" << total << ",\"padding_bytes\":" << padding
      << ",\"mapping_resident_bytes\":" << s.n * 8
      << ",\"record_mismatches\":0,\"seconds\":" << us(start) / 1e6 << "}\n";
}
struct Counters {
  U requested = 0, hits = 0, submitted = 0, bytes = 0, logicalT = 0,
    logicalX = 0, logicalY = 0, useful = 0;
  U shared_hits = 0, evictions = 0;
  U topology_memory_nodes = 0;
  double wait_us = 0, shared_us = 0;
};
struct Store {
  const Source &s;
  std::vector<Group> all;
  std::vector<std::vector<uint8_t> > resident;
  // Canonical topology only: never preload a co-located coordinate page.
  // Identical logical records serve traversal in TXY and T_XY.
  std::vector<uint8_t> topology;
  std::array<int, 3> which{};
  Store(const Source &source, const std::string &layout, const std::string &dir)
      : s(source), all(groups(source, layout, dir, true)) {
    if (std::getenv("LAYOUT_TOPO_DRAM")) {
      topology.resize(s.topo.size);
      std::memcpy(topology.data(), s.topo.p, s.topo.size);
    }
    for (size_t g = 0; g < all.size(); g++)
      for (int k = 0; k < 3; k++)
        if (all[g].mask & (1 << k))
          which[k] = g;
    if (std::getenv("LAYOUT_DRAM")) {
      resident.resize(all.size());
      for (size_t g = 0; g < all.size(); ++g) {
        std::ifstream input(dir + "/group-" + std::to_string(all[g].mask) +
                                ".bin",
                            std::ios::binary);
        resident[g].resize(all[g].bytes);
        if (!input.read(reinterpret_cast<char *>(resident[g].data()),
                        all[g].bytes))
          throw std::runtime_error("load resident page image");
      }
    }
  }
  ~Store() {
    for (auto &g : all)
      ::close(g.fd);
  }
  U key(U id, int bit) const {
    int k = bit == 1 ? 0 : bit == 2 ? 1 : 2;
    U g = which[k];
    return (s.order.positions[id] / all[g].npp) * 8 + g;
  }
  U offset(U id, int bit) const {
    int k = bit == 1 ? 0 : bit == 2 ? 1 : 2;
    const auto &g = all[which[k]];
    return (s.order.positions[id] % g.npp) * g.width + g.offsets[k];
  }
};
// Process-wide LRU of immutable physical pages. The query-local copy remains
// valid even if another worker evicts the shared entry.
class SharedPages {
  struct Entry {
    U slot;
    std::list<U>::iterator lru;
  };
  std::vector<uint8_t> bytes;
  std::unordered_map<U, Entry> map;
  std::list<U> lru;
  std::mutex lock;

public:
  explicit SharedPages(U pages) : bytes(pages * PAGE) { map.reserve(pages); }

  void clear() {
    // Called by the coordinator between points, with no active queries.
    map.clear();
    lru.clear();
  }

  bool lookup(U key, uint8_t *destination) {
    std::lock_guard<std::mutex> guard(lock);
    auto hit = map.find(key);
    if (hit == map.end())
      return false;
    memcpy(destination, bytes.data() + hit->second.slot * PAGE, PAGE);
    lru.splice(lru.begin(), lru, hit->second.lru);
    return true;
  }

  void insert(U key, const uint8_t *source) {
    std::lock_guard<std::mutex> guard(lock);
    auto hit = map.find(key);
    if (hit != map.end()) {
      lru.splice(lru.begin(), lru, hit->second.lru);
      return;
    }
    U slot = map.size();
    if (slot == bytes.size() / PAGE) {
      U old = lru.back();
      slot = map.at(old).slot;
      map.erase(old);
      lru.pop_back();
    }
    memcpy(bytes.data() + slot * PAGE, source, PAGE);
    lru.push_front(key);
    map.emplace(key, Entry{ slot, lru.begin() });
  }
};
struct Cache {
  SharedPages *shared;
  Store &store;
  io_context_t ctx = 0;
  uint8_t *memory = nullptr;
  Counters count;
  struct Entry {
    U slot;
    std::list<U>::iterator lru;
  };
  std::unordered_map<U, Entry> map;
  std::unordered_set<U> ever;
  std::list<U> lru;
  std::vector<U> free;
  explicit Cache(Store &s, SharedPages *pages = nullptr)
      : shared(pages), store(s) {
    if (posix_memalign((void **)&memory, PAGE, CACHE_PAGES * PAGE) ||
        io_setup(128, &ctx))
      throw std::runtime_error("cache or AIO allocation");
    reset();
  }
  ~Cache() {
    if (ctx)
      io_destroy(ctx);
    ::free(memory);
  }
  void reset() {
    count = Counters{};
    map.clear();
    ever.clear();
    lru.clear();
    free.clear();
    for (U i = 0; i < CACHE_PAGES; i++)
      free.push_back(CACHE_PAGES - 1 - i);
  }
  const uint8_t *field(U id, int bit) {
    if (bit == 1 && !store.topology.empty()) {
      if (id >= store.s.n)
        throw std::runtime_error("resident topology ID range");
      return store.topology.data() + (id / store.s.npp) * PAGE +
             (id % store.s.npp) * store.s.width;
    }
    return memory + map.at(store.key(id, bit)).slot * PAGE +
           store.offset(id, bit);
  }
  void fetch(const std::vector<unsigned> &ids, int mask) {
    if ((mask & 1) && !store.topology.empty()) {
      for (auto id : ids)
        if (id >= store.s.n)
          throw std::runtime_error("resident topology fetch ID range");
      count.topology_memory_nodes += ids.size();
      count.logicalT += ids.size();
      count.useful += ids.size() * store.s.width;
      mask &= ~1;
      if (!mask)
        return;
    }
    std::vector<U> keys;
    keys.reserve(ids.size() * 3);
    for (auto id : ids) {
      if (id >= store.s.n)
        throw std::runtime_error("query ID range");
      for (int k = 0; k < 3; k++)
        if (mask & (1 << k)) {
          keys.push_back(store.key(id, 1 << k));
          count.useful += store.s.bytes(1 << k);
          if (k == 0)
            count.logicalT++;
          else if (k == 1)
            count.logicalX++;
          else
            count.logicalY++;
        }
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    if (keys.size() > CACHE_PAGES)
      throw std::runtime_error("batch exceeds equal cache budget");
    count.requested += keys.size();
    ever.insert(keys.begin(), keys.end());
    std::unordered_set<U> needed(keys.begin(), keys.end());
    std::vector<U> misses;
    for (auto key : keys) {
      auto it = map.find(key);
      if (it != map.end()) {
        count.hits++;
        lru.splice(lru.begin(), lru, it->second.lru);
        continue;
      }
      if (free.empty()) {
        ++count.evictions;
        auto it = lru.end();
        do {
          --it;
        } while (needed.count(*it));
        U old = *it;
        free.push_back(map.at(old).slot);
        map.erase(old);
        lru.erase(it);
      }
      U slot = free.back();
      free.pop_back();
      lru.push_front(key);
      map.emplace(key, Entry{ slot, lru.begin() });
      if (shared && (store.all[key % 8].mask & 1)) {
        auto start = phase_now();
        bool hit = shared->lookup(key, memory + slot * PAGE);
        count.shared_us += phase_us(start);
        if (hit) {
          ++count.shared_hits;
          continue;
        }
      }
      misses.push_back(key);
    }
    if (!store.resident.empty()) {
      auto start = phase_now();
      for (auto key : misses)
        memcpy(memory + map.at(key).slot * PAGE,
               store.resident[key % 8].data() + (key / 8) * PAGE, PAGE);
      count.wait_us += phase_us(start);
    } else
      for (U first = 0; first < misses.size(); first += 128) {
        U n = std::min(U(128), U(misses.size()) - first);
        std::vector<iocb> cb(n);
        std::vector<iocb *> ptrs(n);
        std::vector<io_event> events(n);
        for (U j = 0; j < n; j++) {
          U key = misses[first + j];
          io_prep_pread(&cb[j], store.all[key % 8].fd,
                        memory + map.at(key).slot * PAGE, PAGE,
                        (key / 8) * PAGE);
          ptrs[j] = &cb[j];
        }
        auto start = phase_now();
        int ret = io_submit(ctx, n, ptrs.data());
        if (ret != int(n))
          throw std::runtime_error("io_submit short/error");
        U done = 0;
        while (done < n) {
          int got = io_getevents(ctx, n - done, n - done, events.data() + done,
                                 nullptr);
          if (got <= 0)
            throw std::runtime_error("io_getevents error");
          done += got;
        }
        count.wait_us += phase_us(start);
        for (auto &e : events)
          if (e.res != PAGE)
            throw std::runtime_error("short direct read");
      }
    // Only traversal populates the shared cache. Rerank may look up a
    // co-located page but does not promote coordinate reads into the cache.
    if (shared && mask == 1 && !misses.empty()) {
      auto start = phase_now();
      for (auto key : misses)
        shared->insert(key, memory + map.at(key).slot * PAGE);
      count.shared_us += phase_us(start);
    }
    count.submitted += misses.size();
    count.bytes += misses.size() * PAGE;
  }
};
struct Trace {
  std::vector<std::vector<unsigned> > batches;
  std::vector<unsigned> coords;
};
inline void put32(std::ostream &o, unsigned v) { o.write((char *)&v, 4); }
inline unsigned get32(std::istream &i) {
  unsigned v;
  i.read((char *)&v, 4);
  if (!i)
    throw std::runtime_error("trace truncated");
  return v;
}
inline void save_trace(const std::string &path,
                       const std::vector<Trace> &traces) {
  std::ofstream o(path, std::ios::binary);
  put32(o, 0x48565331);
  put32(o, traces.size());
  for (auto &t : traces) {
    put32(o, t.batches.size());
    for (auto &b : t.batches) {
      put32(o, b.size());
      for (auto id : b)
        put32(o, id);
    }
    put32(o, t.coords.size());
    for (auto id : t.coords)
      put32(o, id);
  }
  if (!o)
    throw std::runtime_error("trace write");
}
inline std::vector<Trace> load_trace(const std::string &path) {
  std::ifstream i(path, std::ios::binary);
  if (get32(i) != 0x48565331)
    throw std::runtime_error("trace magic");
  std::vector<Trace> out(get32(i));
  for (auto &t : out) {
    unsigned n = get32(i);
    while (n--) {
      std::vector<unsigned> b(get32(i));
      for (auto &id : b)
        id = get32(i);
      t.batches.push_back(std::move(b));
    }
    t.coords.resize(get32(i));
    for (auto &id : t.coords)
      id = get32(i);
  }
  return out;
}
} // namespace layout_exp
