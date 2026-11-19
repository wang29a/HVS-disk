#include "dual_update.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <thread>

using namespace hvs_update;
static void require(bool v, const char *s) {
  if (!v)
    throw std::runtime_error(s);
}
template <class T> static void put(std::ostream &o, T v) {
  o.write(reinterpret_cast<char *>(&v), sizeof(v));
}
static std::vector<float> matrix(const std::string &p, uint32_t &n,
                                 uint32_t &d) {
  std::ifstream f(p, std::ios::binary);
  f.read(reinterpret_cast<char *>(&n), 4);
  f.read(reinterpret_cast<char *>(&d), 4);
  require(bool(f) && n && d, "matrix header");
  std::vector<float> v(uint64_t(n) * d);
  f.read(reinterpret_cast<char *>(v.data()), v.size() * 4);
  require(bool(f) && f.peek() == EOF, "matrix length");
  return v;
}
int main(int argc, char **argv) {
  try {
    require(argc == 3, "fixture pq|exact");
    const std::string root = argv[1];
    Options options;
    options.exact_reprune = std::string(argv[2]) == "exact";
    require(options.exact_reprune || std::string(argv[2]) == "pq",
            "reprune mode");
    DualUpdate index(root, options);
    uint32_t nx, dx, ny, dy;
    auto x = matrix(root + "/insert-x.bin", nx, dx),
         y = matrix(root + "/insert-y.bin", ny, dy);
    require(nx == 10000 && nx == ny && dx == index.dim1() && dy == index.dim2(),
            "insertion shape");
    const auto initial = index.initial_points();
    require(initial == 90000, "initial count");
    std::vector<uint32_t> tags(index.capacity());
    std::iota(tags.begin(), tags.end(), 0);
    std::atomic<unsigned> next{ 0 }, done{ 0 };
    std::atomic<bool> failed{ false };
    std::mutex mutex;
    std::exception_ptr error;
    std::ofstream progress(root + "/insert-progress.jsonl");
    std::vector<std::thread> workers;
    for (unsigned t = 0; t < 4; ++t)
      workers.emplace_back([&] {
        try {
          while (!failed) {
            auto i = next.fetch_add(1);
            if (i >= nx)
              break;
            IOStats stats;
            auto id =
                index.insert(x.data() + uint64_t(i) * dx,
                             y.data() + uint64_t(i) * dy, initial + i, stats);
            tags[id] = initial + i;
            auto n = done.fetch_add(1) + 1;
            if (n % 100 == 0) {
              std::lock_guard<std::mutex> g(mutex);
              progress << "{\"completed\":" << n << "}\n";
              progress.flush();
            }
          }
        }
        catch (...) {
          failed = true;
          std::lock_guard<std::mutex> g(mutex);
          if (!error)
            error = std::current_exception();
        }
      });
    for (auto &w : workers)
      w.join();
    if (error)
      std::rethrow_exception(error);
    require(done == nx, "incomplete insertion");
    IOStats stats;
    auto graph = index.audit_topology(stats);
    const auto n = index.allocated();
    require(n == 100000, "final count");
    std::vector<unsigned char> meta(8192, 0),
        topo(uint64_t((n + 48) / 49) * 8192, 0);
    auto assign = [&](unsigned off,
                      uint32_t v) { std::memcpy(meta.data() + off, &v, 4); };
    assign(0, n);
    assign(4, dx);
    assign(8, dy);
    assign(12, 40);
    assign(16, 0);
    uint64_t npp = 49;
    std::memcpy(meta.data() + 20, &npp, 8);
    auto entries = index.entries();
    assign(28, entries.size());
    require(32 + 4 * entries.size() <= 8192, "entry count");
    for (unsigned j = 0; j < entries.size(); ++j)
      assign(32 + 4 * j, tags[entries[j]]);
    uint64_t edges = 0;
    for (unsigned id = 0; id < n; ++id) {
      require(index.committed(id) && !index.deleted(id) &&
                  graph[id].size() <= 40,
              "invalid final state");
      unsigned tag = tags[id], degree = graph[id].size();
      auto p = topo.data() + uint64_t(tag / 49) * 8192 + (tag % 49) * 164;
      std::memcpy(p, &degree, 4);
      std::vector<uint32_t> row;
      for (auto v : graph[id]) {
        require(v < n && v != id, "invalid neighbor");
        row.push_back(tags[v]);
      }
      auto copy = row;
      std::sort(copy.begin(), copy.end());
      require(std::adjacent_find(copy.begin(), copy.end()) == copy.end(),
              "duplicate neighbor");
      std::memcpy(p + 4, row.data(), row.size() * 4);
      edges += degree;
    }
    for (auto name : { "eval.meta", "eval.topo" }) {
      const auto &v = std::string(name) == "eval.meta" ? meta : topo;
      std::ofstream f(root + "/" + name, std::ios::binary);
      f.write(reinterpret_cast<const char *>(v.data()), v.size());
      require(bool(f), "export write");
    }
    std::ofstream mapping(root + "/id-to-tag.u32", std::ios::binary);
    mapping.write(reinterpret_cast<char *>(tags.data()), n * 4);
    std::ofstream summary(root + "/insert-result.json");
    summary << "{\"status\":\"INSERT_EXPORT_COMPLETE\",\"insertions\":" << nx
            << ",\"threads\":4,\"allocated\":" << n << ",\"edges\":" << edges
            << ",\"average_degree\":" << double(edges) / n << "}\n";
    std::cout << "G2_INSERT_COMPLETE " << argv[2]
              << " avg_degree=" << double(edges) / n << std::endl;
  }
  catch (const std::exception &e) {
    std::cerr << "G2_INSERT_FAIL " << e.what() << std::endl;
    return 1;
  }
}
