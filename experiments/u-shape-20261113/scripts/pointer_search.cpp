// Research-only exact graph traversal: direct RAM adjacency/vector pointers.
// No page addressing, cache, PQ, rerank, topology I/O, or coordinate I/O.
#include "distance.h"
#include "neighbor.h"
#include "tsl/robin_set.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <vector>
using U = uint64_t;
using Clock = std::chrono::steady_clock;
template <class T> T get(std::istream &f) {
  T x{};
  if (!f.read(reinterpret_cast<char *>(&x), sizeof(x)))
    throw std::runtime_error("truncated input");
  return x;
}
template <class T> void put(std::ostream &f, T x) {
  f.write(reinterpret_cast<const char *>(&x), sizeof(x));
}
template <class T>
std::vector<T> matrix(const std::string &path, unsigned &n, unsigned &d) {
  std::ifstream f(path, std::ios::binary);
  n = get<unsigned>(f);
  d = get<unsigned>(f);
  if (!n || !d || U(n) * d > (U(1) << 36))
    throw std::runtime_error("matrix shape");
  std::vector<T> x(U(n) * d);
  if (!f.read(reinterpret_cast<char *>(x.data()), x.size() * sizeof(T)) ||
      f.peek() != std::char_traits<char>::eof())
    throw std::runtime_error("matrix size");
  return x;
}
unsigned read32(const uint8_t *p) {
  unsigned x;
  memcpy(&x, p, 4);
  return x;
}
struct Meta {
  unsigned n, d1, d2, R, M;
  std::vector<unsigned> entries;
  explicit Meta(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    n = get<unsigned>(f);
    d1 = get<unsigned>(f);
    d2 = get<unsigned>(f);
    R = get<unsigned>(f);
    M = get<unsigned>(f);
    get<U>(f);
    unsigned ne = get<unsigned>(f);
    if (!n || !d1 || !d2 || R > 1000 || M > 64 || !ne || ne > 1000)
      throw std::runtime_error("metadata bounds");
    for (unsigned i = 0; i < ne; i++) {
      unsigned id = get<unsigned>(f);
      if (id >= n)
        throw std::runtime_error("entry ID");
      entries.push_back(id);
    }
  }
};
struct Graph {
  Meta m;
  U width, edges = 0, reachable = 0;
  std::vector<uint8_t> rows;
  double mean_degree = 0, median_degree = 0;
  unsigned max_degree = 0;
  Graph(const std::string &meta, const std::string &raw) : m(meta) {
    if (const char *entry = std::getenv("PTR_ENTRY_META")) {
      Meta e(entry);
      if (e.n != m.n || e.d1 != m.d1 || e.d2 != m.d2)
        throw std::runtime_error("entry override identity");
      m.entries = e.entries;
    }
    width = 4 + U(m.R) * (4 + 2 * m.M);
    rows.resize(U(m.n) * width, 0);
    std::ifstream f(raw, std::ios::binary);
    if (get<U>(f) != m.n || get<unsigned>(f) != m.d1 ||
        get<unsigned>(f) != m.d2)
      throw std::runtime_error("source graph identity");
    std::vector<unsigned> degrees(m.n);
    for (unsigned id = 0; id < m.n; id++) {
      unsigned nn = get<unsigned>(f);
      auto *p = rows.data() + U(id) * width;
      if (nn > m.R)
        throw std::runtime_error("degree budget");
      memcpy(p, &nn, 4);
      degrees[id] = nn;
      edges += nn;
      std::vector<unsigned> seen;
      for (unsigned k = 0; k < nn; k++) {
        unsigned nb = get<unsigned>(f), nr = get<unsigned>(f);
        if (nb >= m.n || nb == id || nr > 4096 ||
            std::find(seen.begin(), seen.end(), nb) != seen.end())
          throw std::runtime_error("edge bounds/duplicate/self");
        seen.push_back(nb);
        std::vector<std::pair<int, int> > ranges;
        for (unsigned z = 0; z < nr; z++) {
          float lo = get<float>(f), hi = get<float>(f);
          if (!std::isfinite(lo) || !std::isfinite(hi) || lo < 0 || hi > 1 ||
              lo > hi)
            throw std::runtime_error("source interval");
          int a = std::floor(double(lo) * 100), b = std::ceil(double(hi) * 100);
          if (!a && !b)
            b = 1;
          ranges.emplace_back(a, b);
        }
        std::sort(ranges.begin(), ranges.end());
        std::vector<std::pair<int, int> > merged;
        for (auto r : ranges) {
          if (!merged.empty() && r.first <= merged.back().second)
            merged.back().second = std::max(merged.back().second, r.second);
          else
            merged.push_back(r);
        }
        if (m.M && merged.size() > m.M)
          throw std::runtime_error("range slots");
        auto *edge = p + 4 + U(k) * (4 + 2 * m.M);
        memcpy(edge, &nb, 4);
        if (m.M)
          for (unsigned z = 0; z < merged.size(); z++) {
            edge[4 + 2 * z] = merged[z].first;
            edge[5 + 2 * z] = merged[z].second;
          }
      }
    }
    if (f.peek() != std::char_traits<char>::eof())
      throw std::runtime_error("graph trailing bytes");
    mean_degree = double(edges) / m.n;
    std::sort(degrees.begin(), degrees.end());
    median_degree = (double(degrees[(m.n - 1) / 2]) + degrees[m.n / 2]) / 2;
    max_degree = degrees.back();
    std::vector<uint8_t> visited(m.n, 0);
    std::vector<unsigned> todo;
    for (auto id : m.entries)
      if (!visited[id]) {
        visited[id] = 1;
        todo.push_back(id);
      }
    for (U j = 0; j < todo.size(); j++) {
      auto *p = rows.data() + U(todo[j]) * width;
      for (unsigned k = 0; k < read32(p); k++) {
        unsigned nb = read32(p + 4 + U(k) * (4 + 2 * m.M));
        if (!visited[nb]) {
          visited[nb] = 1;
          todo.push_back(nb);
        }
      }
    }
    reachable = todo.size();
  }
};
struct Answer {
  std::array<unsigned, 10> ids{};
  std::array<float, 10> distances{};
  U expanded = 0, evals = 0, edges = 0, rejected = 0, duplicate = 0,
    inserted = 0;
  unsigned hits = 0;
  double us = 0;
  std::vector<std::vector<unsigned> > batches;
};
struct Engine {
  Graph &g;
  std::vector<float> x, y;
  bool concat = std::getenv("PTR_CONCAT");
  std::unique_ptr<pipeann::Distance<float> > distance;
  Engine(Graph &graph, const std::string &px, const std::string &py)
      : g(graph) {
    unsigned n, d;
    x = matrix<float>(px, n, d);
    if (n != g.m.n || d != g.m.d1)
      throw std::runtime_error("base X");
    y = matrix<float>(py, n, d);
    if (n != g.m.n || d != g.m.d2)
      throw std::runtime_error("base Y");
    if (concat) {
      std::vector<float> joined(U(n) * (g.m.d1 + g.m.d2));
      for (U i = 0; i < n; i++) {
        memcpy(joined.data() + i * (g.m.d1 + g.m.d2), x.data() + i * g.m.d1,
               g.m.d1 * 4);
        memcpy(joined.data() + i * (g.m.d1 + g.m.d2) + g.m.d1,
               y.data() + i * g.m.d2, g.m.d2 * 4);
      }
      x.swap(joined);
      y.clear();
      y.shrink_to_fit();
    }
    distance.reset(pipeann::get_distance_function<float>(pipeann::Metric::L2));
  }
  Answer search(const float *qx, const float *qy, float alpha, unsigned L,
                bool filter, bool trace) const {
    Answer a;
    auto start = Clock::now();
    std::vector<float> joined;
    if (concat) {
      joined.insert(joined.end(), qx, qx + g.m.d1);
      joined.insert(joined.end(), qy, qy + g.m.d2);
    }
    auto score = [&](unsigned id) {
      if (concat) {
        ++a.evals;
        return std::sqrt(std::max(
            0.f, distance->compare(joined.data(),
                                   x.data() + U(id) * (g.m.d1 + g.m.d2),
                                   g.m.d1 + g.m.d2)));
      }
      float dx = distance->compare(qx, x.data() + U(id) * g.m.d1, g.m.d1);
      float dy = distance->compare(qy, y.data() + U(id) * g.m.d2, g.m.d2);
      a.evals += 2;
      return alpha * std::sqrt(std::max(0.f, dx)) +
             (1 - alpha) * std::sqrt(std::max(0.f, dy));
    };
    std::vector<pipeann::Neighbor> ret(
        std::max(U(L) * 10, g.m.entries.size() + 1));
    tsl::robin_set<U> visited;
    unsigned cur = 0;
    for (auto id : g.m.entries) {
      ret[cur++] = pipeann::Neighbor(id, score(id), true);
      visited.insert(id);
    }
    std::sort(ret.begin(), ret.begin() + cur);
    unsigned k = 0;
    while (k < cur) {
      unsigned nk = cur, marker = k, seen = 0;
      std::vector<unsigned> frontier;
      while (marker < cur && frontier.size() < 4 && seen < 4) {
        if (ret[marker].flag) {
          ++seen;
          frontier.push_back(ret[marker].id);
          ret[marker].flag = false;
        }
        ++marker;
      }
      if (trace)
        a.batches.push_back(frontier);
      a.expanded += frontier.size();
      for (auto id : frontier) {
        const auto *p = g.rows.data() + U(id) * g.width;
        std::vector<unsigned> nbrs;
        for (unsigned m = 0; m < read32(p); m++) {
          auto *edge = p + 4 + U(m) * (4 + 2 * g.m.M);
          unsigned nb = read32(edge);
          bool active = !filter || g.m.M == 0;
          ++a.edges;
          auto *ranges = reinterpret_cast<const int8_t *>(edge + 4);
          for (unsigned z = 0; filter && z < g.m.M; z++) {
            int lo = ranges[2 * z], hi = ranges[2 * z + 1];
            if (lo < 0 || hi < 0 || lo > 100 || hi > 100 || (!lo && !hi))
              break;
            float a100 = alpha * 100.f;
            if (a100 >= lo && a100 <= hi) {
              active = true;
              break;
            }
            if (a100 < lo)
              break;
          }
          if (!active) {
            ++a.rejected;
            continue;
          }
          if (visited.find(nb) != visited.end()) {
            ++a.duplicate;
            continue;
          }
          visited.insert(nb);
          nbrs.push_back(nb);
        }
        for (auto nb : nbrs) {
          float d = score(nb);
          if (cur < L || d < ret[cur - 1].distance) {
            unsigned pos = pipeann::InsertIntoPool(
                ret.data(), cur, pipeann::Neighbor(nb, d, true));
            ++a.inserted;
            if (cur < L)
              ++cur;
            if (pos < nk)
              nk = pos;
          }
        }
      }
      if (nk <= k)
        k = nk;
      else
        ++k;
    }
    if (cur < 10)
      throw std::runtime_error("fewer than ten reachable candidates");
    for (unsigned j = 0; j < 10; j++) {
      a.ids[j] = ret[j].id;
      a.distances[j] = ret[j].distance;
    }
    a.us =
        std::chrono::duration<double, std::micro>(Clock::now() - start).count();
    return a;
  }
};
double epoch() {
  return std::chrono::duration<double>(
      std::chrono::system_clock::now().time_since_epoch()).count();
}
int main(int argc, char **argv) {
  try {
    if (argc < 15)
      throw std::runtime_error("meta float-graph baseX baseY queryX queryY "
                               "alpha GT queryIDs output threads seconds "
                               "filter L...");
    Graph g(argv[1], argv[2]);
    Engine engine(g, argv[3], argv[4]);
    unsigned q, d, qb, db, na, da, ng, dg;
    auto qx = matrix<float>(argv[5], q, d), qy = matrix<float>(argv[6], qb, db),
         alpha = matrix<float>(argv[7], na, da);
    auto gt = matrix<unsigned>(argv[8], ng, dg);
    if (q != qb || q != na || q != ng || d != g.m.d1 || db != g.m.d2 ||
        da != 1 || dg != 10)
      throw std::runtime_error("query/GT shape");
    for (auto a : alpha)
      if (!std::isfinite(a) || a < 0 || a > 1)
        throw std::runtime_error("query weight");
    std::vector<unsigned> ids;
    if (std::string(argv[9]) == "-") {
      ids.resize(q);
      std::iota(ids.begin(), ids.end(), 0);
    } else {
      std::ifstream f(argv[9]);
      unsigned id;
      while (f >> id) {
        if (id >= q)
          throw std::runtime_error("query ID");
        ids.push_back(id);
      }
    }
    if (ids.empty())
      throw std::runtime_error("empty cohort");
    unsigned threads = std::stoul(argv[11]);
    double duration = std::stod(argv[12]);
    bool filter = std::stoi(argv[13]);
    if (!threads || threads > 64 || duration < 0 || duration > 60 ||
        (filter && !g.m.M))
      throw std::runtime_error("run configuration");
    omp_set_num_threads(threads);
    bool trace = std::getenv("PTR_TRACE");
    if (trace && duration)
      throw std::runtime_error("trace is diagnostic only");
    std::ofstream out(argv[10]);
    out << std::setprecision(12) << "[";
    bool first = true;
    for (int arg = 14; arg < argc; arg++) {
      unsigned L = std::stoul(argv[arg]);
      if (L < 10 || L > 6400 || g.m.entries.size() > L)
        throw std::runtime_error("L/entries");
      std::vector<Answer> answers(ids.size());
      std::vector<double> lat;
      auto batch = [&](bool measured) {
#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t j = 0; j < int64_t(ids.size()); j++) {
          unsigned id = ids[j];
          answers[j] =
              engine.search(qx.data() + U(id) * d, qy.data() + U(id) * db,
                            alpha[id], L, filter, trace);
          for (auto nb : answers[j].ids)
            for (unsigned k = 0; k < 10; k++)
              if (nb == gt[U(id) * 10 + k]) {
                ++answers[j].hits;
                break;
              }
        }
        if (measured)
          for (auto &a : answers)
            lat.push_back(a.us);
      };
      if (duration)
        batch(false);
      double begin = epoch();
      auto tick = Clock::now();
      unsigned batches = 0;
      double seconds;
      do {
        batch(true);
        ++batches;
        seconds = std::chrono::duration<double>(Clock::now() - tick).count();
      } while (seconds < duration);
      std::vector<double> sorted = lat;
      std::sort(sorted.begin(), sorted.end());
      double hits = 0, expanded = 0, evals = 0, edges = 0, rejected = 0,
             dups = 0, inserted = 0;
      for (auto &a : answers) {
        hits += a.hits;
        expanded += a.expanded;
        evals += a.evals;
        edges += a.edges;
        rejected += a.rejected;
        dups += a.duplicate;
        inserted += a.inserted;
      }
      double n = ids.size();
      if (!first)
        out << ",";
      first = false;
      out << "{\"engine\":\"exact-pointer\",\"L\":" << L
          << ",\"filter\":" << (filter ? "true" : "false")
          << ",\"threads\":" << threads << ",\"query_count\":" << ids.size()
          << ",\"samples\":" << lat.size() << ",\"point_start_unix\":" << begin
          << ",\"measured_seconds\":" << seconds
          << ",\"recall\":" << hits / (n * 10)
          << ",\"qps\":" << lat.size() / seconds << ",\"latency_us\":"
          << std::accumulate(lat.begin(), lat.end(), 0.) / lat.size()
          << ",\"p50_us\":" << sorted[sorted.size() / 2] << ",\"p95_us\":"
          << sorted[std::min(sorted.size() - 1,
                             U(std::ceil(.95 * sorted.size()) - 1))]
          << ",\"p99_us\":"
          << sorted[std::min(sorted.size() - 1,
                             U(std::ceil(.99 * sorted.size()) - 1))]
          << ",\"expanded\":" << expanded / n
          << ",\"exact_space_evals\":" << evals / n
          << ",\"inspected_edges\":" << edges / n
          << ",\"alpha_rejected\":" << rejected / n
          << ",\"duplicate_rejected\":" << dups / n
          << ",\"inserted\":" << inserted / n << ",\"nodes\":" << g.m.n
          << ",\"edges\":" << g.edges << ",\"mean_degree\":" << g.mean_degree
          << ",\"median_degree\":" << g.median_degree
          << ",\"max_degree\":" << g.max_degree
          << ",\"reachable\":" << g.reachable
          << ",\"ram_topology_bytes\":" << g.rows.size()
          << ",\"topology_submitted_pages\":0,\"coordinate_submitted_pages\":0,"
             "\"queries\":[";
      for (U j = 0; j < answers.size(); j++) {
        auto &a = answers[j];
        if (j)
          out << ",";
        out << "{\"query_id\":" << ids[j] << ",\"hits\":" << a.hits
            << ",\"expanded\":" << a.expanded
            << ",\"exact_space_evals\":" << a.evals
            << ",\"inspected_edges\":" << a.edges
            << ",\"alpha_rejected\":" << a.rejected << ",\"ids\":[";
        for (unsigned k = 0; k < 10; k++) {
          if (k)
            out << ",";
          out << a.ids[k];
        }
        out << "]}";
      }
      out << "]}";
      out.flush();
      if (const char *path = std::getenv("PTR_TRACE")) {
        std::string stem = std::string(path) + ".L" + std::to_string(L);
        std::ofstream f(stem, std::ios::binary),
            fi(stem + ".ids", std::ios::binary),
            fd(stem + ".dists", std::ios::binary);
        put<unsigned>(f, 0x48565331);
        put<unsigned>(f, answers.size());
        for (auto &a : answers) {
          put<unsigned>(f, a.batches.size());
          for (auto &b : a.batches) {
            put<unsigned>(f, b.size());
            for (auto id : b)
              put(f, id);
          }
          put<unsigned>(f, 0);
          fi.write(reinterpret_cast<char *>(a.ids.data()), 40);
          fd.write(reinterpret_cast<char *>(a.distances.data()), 40);
        }
      }
    }
    out << "]\n";
    if (!out)
      throw std::runtime_error("output write");
  }
  catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
