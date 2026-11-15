// Diagnostic candidate pools are unions of final neighbors, not recovered GPS
// histories. This separates ordering/budget/quantization from scalar
// predicates.
#include "index.h"
#include "parameters.h"
#include "fixed_pruning.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <limits>
#include <memory>
#include <numeric>
#include <omp.h>
#include <random>
#include <set>
#include <stdexcept>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
using U = uint64_t;
struct Map {
  int fd;
  size_t size;
  const unsigned char *p;
  Map(const char *path) {
    fd = open(path, O_RDONLY);
    struct stat s {};
    if (fd < 0 || fstat(fd, &s))
      throw std::runtime_error("open map");
    size = s.st_size;
    p = (const unsigned char *)mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd,
                                    0);
    if (p == MAP_FAILED)
      throw std::runtime_error("mmap");
  }
  ~Map() {
    munmap((void *)p, size);
    close(fd);
  }
};
template <class T> T take(const unsigned char *&p, const unsigned char *end) {
  if (size_t(end - p) < sizeof(T))
    throw std::runtime_error("truncated graph");
  T x;
  memcpy(&x, p, sizeof(x));
  p += sizeof(x);
  return x;
}
void neighbors(const char *path,
               std::map<unsigned, std::vector<unsigned> > &chosen, unsigned n,
               unsigned d) {
  Map m(path);
  const auto *p = m.p;
  const auto *end = p + m.size;
  if (take<U>(p, end) != n || take<unsigned>(p, end) != d ||
      take<unsigned>(p, end) != d)
    throw std::runtime_error("graph identity");
  for (unsigned node = 0; node < n; node++) {
    unsigned count = take<unsigned>(p, end);
    auto it = chosen.find(node);
    for (unsigned j = 0; j < count; j++) {
      unsigned id = take<unsigned>(p, end), nr = take<unsigned>(p, end);
      if (id >= n || nr > 64 || size_t(end - p) < nr * 8)
        throw std::runtime_error("edge bounds");
      if (it != chosen.end())
        it->second.push_back(id);
      p += nr * 8;
    }
  }
  if (p != end)
    throw std::runtime_error("trailing graph");
}
void load(const char *path, float *target, unsigned n, unsigned dim) {
  std::ifstream f(path, std::ios::binary);
  unsigned nn, dd;
  f.read((char *)&nn, 4);
  f.read((char *)&dd, 4);
  if (nn != n || dd != dim || !f.read((char *)target, U(n) * dim * 4))
    throw std::runtime_error("base identity");
}
std::set<unsigned> ids(const std::vector<fixed_pruning::Candidate> &v) {
  std::set<unsigned> s;
  for (auto p : v)
    s.insert(p.id);
  return s;
}
std::set<unsigned> active(const std::vector<pipeann::DEGNeighbor> &v,
                          bool quant) {
  std::set<unsigned> s;
  for (auto p : v)
    for (auto r : p.available_range) {
      double lo = quant ? std::floor(double(r.first) * 100) : r.first,
             hi = quant ? std::ceil(double(r.second) * 100) : r.second;
      double a = quant ? 50 : .5;
      if (a >= lo && a <= hi) {
        s.insert(p.id_);
        break;
      }
    }
  return s;
}
struct Counts {
  U pools = 0, a = 0, b = 0, c = 0, cq = 0, d = 0, ab = 0, ac = 0, bc = 0,
    cd = 0, cquant = 0, predicates = 0, hard_mismatches = 0, tie_mismatches = 0;
};
int main(int argc, char **argv) {
  try {
    if (argc != 6)
      throw std::runtime_error("unified-float fixed-float baseX baseY output");
    const unsigned n = 2000000, dim = 512, samples = 1000;
    std::mt19937 rng(20261113);
    std::map<unsigned, std::vector<unsigned> > chosen;
    while (chosen.size() < samples)
      chosen.emplace(std::uniform_int_distribution<unsigned>(0, n - 1)(rng),
                     std::vector<unsigned>{});
    neighbors(argv[1], chosen, n, dim);
    neighbors(argv[2], chosen, n, dim);
    pipeann::HybridIndex<float> index(pipeann::Metric::L2, dim, dim, n, false,
                                      false);
    load(argv[3], index._data1, n, dim);
    load(argv[4], index._data2, n, dim);
    std::unique_ptr<pipeann::Distance<float> > kernel(
        pipeann::get_distance_function<float>(pipeann::Metric::L2));
    std::vector<std::pair<unsigned, std::vector<unsigned> > > pools(
        chosen.begin(), chosen.end());
    std::vector<Counts> all(4);
    omp_set_num_threads(4);
#pragma omp parallel for schedule(dynamic, 1)
    for (int k = 0; k < int(pools.size()); k++) {
      auto &c = all[omp_get_thread_num()];
      auto node = pools[k].first;
      auto list = pools[k].second;
      std::sort(list.begin(), list.end());
      list.erase(std::unique(list.begin(), list.end()), list.end());
      list.erase(std::remove(list.begin(), list.end(), node), list.end());
      const unsigned m = list.size();
      if (!m)
        continue;
      std::map<unsigned, unsigned> at;
      std::vector<float> x(m), y(m), pairx(m * m), pairy(m * m);
      std::vector<pipeann::NeighborH> pool;
      std::vector<fixed_pruning::Candidate> candidate;
      auto component = [&](unsigned p, unsigned q, bool second) {
        const float *base = second ? index._data2 : index._data1;
        return std::sqrt(std::max(
            0.f, kernel->compare(base + U(p) * dim, base + U(q) * dim, dim)));
      };
      for (unsigned i = 0; i < m; i++) {
        at[list[i]] = i;
        x[i] = component(node, list[i], false);
        y[i] = component(node, list[i], true);
        pool.emplace_back(list[i], x[i], y[i], true, 0);
        candidate.push_back({ list[i], .5f * x[i] + .5f * y[i] });
      }
      for (unsigned i = 0; i < m; i++)
        for (unsigned j = 0; j <= i; j++) {
          pairx[i * m + j] = pairx[j * m + i] =
              component(list[i], list[j], false);
          pairy[i * m + j] = pairy[j * m + i] =
              component(list[i], list[j], true);
        }
      auto score = [&](unsigned i) { return .5f * x[i] + .5f * y[i]; };
      auto distance = [&](unsigned p, unsigned q) {
        unsigned i = at.at(p), j = at.at(q);
        return .5f * pairx[i * m + j] + .5f * pairy[i * m + j];
      };
      for (unsigned i = 0; i < m; i++)
        for (unsigned j = 0; j < m; j++)
          if (i != j) {
            ++c.predicates;
            float s1 = x[j] - x[i] + y[i] - y[j], r1 = y[i] - y[j],
                  s2 = pairx[i * m + j] - x[i] + y[i] - pairy[i * m + j],
                  r2 = y[i] - pairy[i * m + j];
            bool direct =
                score(j) <= score(i) && distance(list[j], list[i]) <= score(i);
            bool affine = .5f * s1 <= r1 && .5f * s2 <= r2;
            if (direct != affine) {
              double eps = 1e-6 * std::max(1.f, score(i));
              if (std::abs(double(score(i)) - score(j)) > eps &&
                  std::abs(double(score(i)) - distance(list[j], list[i])) > eps)
                ++c.hard_mismatches;
              else
                ++c.tie_mismatches;
            }
          }
      auto a = ids(fixed_pruning::select(candidate, 40, distance));
      auto pending = pool;
      std::sort(pending.begin(), pending.end());
      std::vector<unsigned> order;
      while (!pending.empty()) {
        float minimum = std::numeric_limits<float>::max();
        std::vector<pipeann::NeighborH> remain;
        for (auto p : pending) {
          if (p.distance1 < minimum) {
            minimum = p.distance1;
            order.push_back(p.id);
          } else
            remain.push_back(p);
        }
        pending.swap(remain);
      }
      std::vector<unsigned> selected;
      for (auto q : order) {
        bool covered = false;
        for (auto p : selected)
          if (score(at[p]) <= score(at[q]) && distance(p, q) <= score(at[q])) {
            covered = true;
            break;
          }
        if (!covered)
          selected.push_back(q);
        if (selected.size() == 40)
          break;
      }
      std::set<unsigned> b(selected.begin(), selected.end());
      pipeann::Parameters p;
      p.Set<unsigned>("R", 40);
      std::vector<pipeann::DEGNeighbor> result;
      auto copy = pool;
      index.prune_neighbors_no_opt(copy, p, result);
      auto cf = active(result, false), cq = active(result, true);
      p.Set<unsigned>("R", m + 1);
      copy = pool;
      std::vector<pipeann::DEGNeighbor> no_cap;
      index.prune_neighbors_no_opt(copy, p, no_cap);
      auto d = active(no_cap, false);
      ++c.pools;
      c.a += a.size();
      c.b += b.size();
      c.c += cf.size();
      c.cq += cq.size();
      c.d += d.size();
      c.ab += a != b;
      c.ac += a != cf;
      c.bc += b != cf;
      c.cd += cf != d;
      c.cquant += cf != cq;
    }
    Counts c;
    for (auto v : all) {
#define ADD(x) c.x += v.x
      ADD(pools);
      ADD(a);
      ADD(b);
      ADD(c);
      ADD(cq);
      ADD(d);
      ADD(ab);
      ADD(ac);
      ADD(bc);
      ADD(cd);
      ADD(cquant);
      ADD(predicates);
      ADD(hard_mismatches);
      ADD(tie_mismatches);
#undef ADD
    }
    std::ofstream out(argv[5]);
    out << std::setprecision(12)
        << "{\"scope\":\"1000 seeded diagnostic pools: unions of final U/S "
           "neighbors, not historical GPS candidate pools\",\"pools\":"
        << c.pools << ",\"mean_fixed_fused_order\":" << double(c.a) / c.pools
        << ",\"mean_pointwise_pareto_order\":" << double(c.b) / c.pools
        << ",\"mean_actual_unified_float\":" << double(c.c) / c.pools
        << ",\"mean_actual_unified_quantized\":" << double(c.cq) / c.pools
        << ",\"mean_actual_unified_no_R_cap\":" << double(c.d) / c.pools
        << ",\"pools_order_changes_set\":" << c.ab
        << ",\"pools_fixed_vs_actual_changes_set\":" << c.ac
        << ",\"pools_pareto_point_vs_actual_changes_set\":" << c.bc
        << ",\"pools_R_cap_changes_active_set\":" << c.cd
        << ",\"pools_quantization_changes_active_set\":" << c.cquant
        << ",\"pair_predicates\":" << c.predicates
        << ",\"positive_margin_predicate_mismatches\":" << c.hard_mismatches
        << ",\"near_tie_or_rounding_mismatches\":" << c.tie_mismatches << "}\n";
    if (!out || c.hard_mismatches)
      throw std::runtime_error(
          "output failure or positive-margin predicate mismatch");
  }
  catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
  return 0;
}
