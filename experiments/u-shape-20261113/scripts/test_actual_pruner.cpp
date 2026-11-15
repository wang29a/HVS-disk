#include "index.h"
#include "parameters.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
static void require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
int main(int argc, char **argv) {
  try {
    require(argc == 3, "fixed|unified alpha");
    bool fixed = std::string(argv[1]) == "fixed";
    float a = std::stof(argv[2]);
    const unsigned n = 12, dim = 16;
    pipeann::HybridIndex<float> index(pipeann::Metric::L2, dim, dim, n, false,
                                      false);
    pipeann::Parameters params;
    params.Set<unsigned>("R", 8);
    std::mt19937 rng(20261113);
    for (unsigned trial = 0; trial < 1000; trial++) {
      int x[n], y[n];
      for (unsigned i = 0; i < n; i++) {
        x[i] = int(rng() % 41) - 20;
        y[i] = int(rng() % 41) - 20;
      }
      if (trial % 4 == 0) {
        x[1] = x[2];
        y[1] = y[2];
      }
      if (trial % 4 == 1) {
        x[1] = x[0];
        x[2] = x[0];
      }
      if (!fixed || trial % 4 == 2)
        for (unsigned i = 0; i < n; i++)
          y[i] = x[i];
      int xx[n] [dim] {}, yy[n] [dim] {};
      for (unsigned i = 0; i < n; i++) {
        xx[i][0] = x[i];
        yy[i][0] = y[i];
        if (fixed && trial % 4 == 3)
          for (unsigned j = 1; j < 3; j++) {
            xx[i][j] = int(rng() % 41) - 20;
            yy[i][j] = int(rng() % 41) - 20;
          }
        std::fill(index._data1 + size_t(i) * dim,
                  index._data1 + size_t(i + 1) * dim, 0.f);
        std::fill(index._data2 + size_t(i) * dim,
                  index._data2 + size_t(i + 1) * dim, 0.f);
        for (unsigned j = 0; j < dim; j++) {
          index._data1[size_t(i) * dim + j] = xx[i][j];
          index._data2[size_t(i) * dim + j] = yy[i][j];
        }
      }
      auto dx = [&](unsigned p, unsigned q) {
        double sum = 0;
        for (unsigned j = 0; j < dim; j++) {
          double z = xx[p][j] - xx[q][j];
          sum += z * z;
        }
        return float(std::sqrt(sum));
      };
      auto dy = [&](unsigned p, unsigned q) {
        double sum = 0;
        for (unsigned j = 0; j < dim; j++) {
          double z = yy[p][j] - yy[q][j];
          sum += z * z;
        }
        return float(std::sqrt(sum));
      };
      auto dist = [&](unsigned p,
                      unsigned q) { return a * dx(p, q) + (1 - a) * dy(p, q); };
      std::vector<pipeann::NeighborH> pool;
      for (unsigned i = 1; i < n; i++)
        pool.emplace_back(i, fixed ? dist(0, i) : dx(0, i),
                          fixed ? dist(0, i) : dy(0, i), true, 0);
      std::shuffle(pool.begin(), pool.end(), rng);
      std::vector<unsigned> order;
      if (fixed) {
        order.resize(n - 1);
        std::iota(order.begin(), order.end(), 1);
        std::sort(order.begin(), order.end(), [&](unsigned p, unsigned q) {
          return dist(0, p) < dist(0, q) || (dist(0, p) == dist(0, q) && p < q);
        });
      } else {
        // Identical spaces remove thresholds/interval-order confounds. Preserve
        // the legacy comparator's chosen tie order for an exact semantic test.
        auto copy = pool;
        std::sort(copy.begin(), copy.end());
        for (auto p : copy)
          order.push_back(p.id);
      }
      std::vector<unsigned> reference;
      for (auto q : order) {
        bool covered = false;
        for (auto p : reference)
          if (dist(0, p) <= dist(0, q) && dist(p, q) <= dist(0, q))
            covered = true;
        if (!covered)
          reference.push_back(q);
        if (reference.size() == 8)
          break;
      }
      std::vector<pipeann::DEGNeighbor> got;
      index.prune_neighbors_no_opt(pool, params, got);
      require(got.size() == reference.size(),
              "actual/reference degree mismatch");
      for (unsigned i = 0; i < got.size(); i++) {
        require(got[i].id_ == reference[i],
                "actual/reference edge ID mismatch");
        require(got[i].available_range ==
                    std::vector<std::pair<float, float> >{ { 0, 1 } },
                "expected complete interval");
        for (unsigned j = 0; j < i; j++)
          require(
              !(dist(0, got[j].id_) <= dist(0, got[i].id_) &&
                dist(got[j].id_, got[i].id_) <= dist(0, got[i].id_)),
              "retained neighbor is covered by an earlier retained witness");
      }
    }
    std::cout << "PASS 1000 actual member-pruner/reference cases mode="
              << argv[1] << " alpha=" << a << "\n";
  }
  catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
  return 0;
}
