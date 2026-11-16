#include "index.h"
#include "parameters.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
using Edge = pipeann::DEGNeighbor;
using Candidate = pipeann::NeighborH;
unsigned bits(float x) {
  unsigned v;
  std::memcpy(&v, &x, 4);
  return v;
}
bool same_ranges(const Edge &a, const Edge &b) {
  if (a.available_range.size() != b.available_range.size())
    return false;
  for (size_t i = 0; i < a.available_range.size(); ++i)
    if (bits(a.available_range[i].first) != bits(b.available_range[i].first) ||
        bits(a.available_range[i].second) != bits(b.available_range[i].second))
      return false;
  return true;
}
void dump_edges(std::ostream &out, const std::vector<Edge> &edges) {
  out << "[";
  bool first = true;
  for (const auto &e : edges) {
    if (!first)
      out << ",";
    first = false;
    out << "{\"id\":" << e.id_ << ",\"layer\":" << e.layer_
        << ",\"d1\":" << e.emb_distance_ << ",\"d2\":" << e.geo_distance_
        << ",\"ranges\":[";
    for (size_t i = 0; i < e.available_range.size(); ++i) {
      if (i)
        out << ",";
      out << "[" << e.available_range[i].first << ","
          << e.available_range[i].second << "]";
    }
    out << "],\"endpoint_bits\":[";
    for (size_t i = 0; i < e.available_range.size(); ++i) {
      if (i)
        out << ",";
      out << "[" << bits(e.available_range[i].first) << ","
          << bits(e.available_range[i].second) << "]";
    }
    out << "]}";
  }
  out << "]";
}
int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::runtime_error("output-directory required");
    const unsigned n = 751, dim = 16, R = 40, trials = 2400, seed = 20261115;
    const char *names[] = { "general_independent_spaces", "duplicate_vectors",
                            "identical_spaces",           "zero_first_space",
                            "cross_space_mirror",         "all_zero_distances",
                            "equal_radius",               "zero_second_space" };
    pipeann::HybridIndex<float> index(pipeann::Metric::L2, dim, dim, n, false,
                                      false);
    std::unique_ptr<pipeann::Distance<float> > distance(
        pipeann::get_distance_function<float>(pipeann::Metric::L2));
    pipeann::Parameters params;
    params.Set<unsigned>("R", R);
    std::mt19937 rng(seed);
    unsigned tested[8]{}, failed[8]{}, changed_edges = 0, deleted_edges = 0,
                                       order_changes = 0, examples = 0,
                                       changed_layers = 0;
    unsigned max_changed_edges = 0;
    for (unsigned trial = 0; trial < trials; ++trial) {
      unsigned kind = trial % 8,
               count = (trial % 7 == 6 ? 750 : unsigned(16u << (trial % 5)));
      count = std::min(count, n - 1);
      for (unsigned i = 0; i < n; ++i) {
        std::fill(index._data1 + size_t(i) * dim,
                  index._data1 + size_t(i + 1) * dim, 0.f);
        std::fill(index._data2 + size_t(i) * dim,
                  index._data2 + size_t(i + 1) * dim, 0.f);
        for (unsigned d = 0; d < 3; ++d) {
          index._data1[size_t(i) * dim + d] = float(int(rng() % 41) - 20);
          index._data2[size_t(i) * dim + d] = float(int(rng() % 41) - 20);
        }
      }
      std::fill(index._data1, index._data1 + dim, 0.f);
      std::fill(index._data2, index._data2 + dim, 0.f);
      if (kind == 1)
        for (unsigned i = 2; i <= count; i += 3) {
          std::copy(index._data1 + dim, index._data1 + 2 * dim,
                    index._data1 + size_t(i) * dim);
          std::copy(index._data2 + dim, index._data2 + 2 * dim,
                    index._data2 + size_t(i) * dim);
        }
      if (kind == 2)
        std::copy(index._data1, index._data1 + size_t(n) * dim, index._data2);
      if (kind == 3 || kind == 5)
        std::fill(index._data1, index._data1 + size_t(n) * dim, 0.f);
      if (kind == 7 || kind == 5)
        std::fill(index._data2, index._data2 + size_t(n) * dim, 0.f);
      if (kind == 4)
        for (unsigned i = 1; i <= count; ++i) {
          index._data1[size_t(i) * dim] = float(i % 17);
          index._data2[size_t(i) * dim] = float(16 - i % 17);
          index._data1[size_t(i) * dim + 1] =
              index._data1[size_t(i) * dim + 2] = 0;
          index._data2[size_t(i) * dim + 1] =
              index._data2[size_t(i) * dim + 2] = 0;
        }
      if (kind == 6)
        for (unsigned i = 1; i <= count; ++i) {
          index._data1[size_t(i) * dim] = (i % 2 ? 3.f : -3.f);
          index._data1[size_t(i) * dim + 1] = 4.f;
          index._data1[size_t(i) * dim + 2] = 0;
        }
      std::vector<Candidate> pool;
      for (unsigned i = 1; i <= count; ++i) {
        float dx = std::sqrt(std::max(
            0.f, distance->compare(index._data1, index._data1 + size_t(i) * dim,
                                   dim)));
        float dy = std::sqrt(std::max(
            0.f, distance->compare(index._data2, index._data2 + size_t(i) * dim,
                                   dim)));
        pool.emplace_back(i, dx, dy, true, 0);
      }
      std::shuffle(pool.begin(), pool.end(), rng);
      const auto original = pool;
      std::vector<Edge> first, second;
      index.prune_neighbors_no_opt(pool, params, first);
      std::vector<Candidate> again;
      // Exactly the saved output order, same distances, no reverse insertion.
      for (const auto &e : first)
        again.emplace_back(e.id_, e.emb_distance_, e.geo_distance_, true,
                           int(e.layer_));
      index.prune_neighbors_no_opt(again, params, second);
      std::map<unsigned, Edge> a, b;
      for (const auto &e : first)
        a.emplace(e.id_, e);
      for (const auto &e : second)
        b.emplace(e.id_, e);
      bool same = a.size() == b.size(), order = first.size() == second.size();
      unsigned local_changed = 0, local_deleted = 0, layer_changed = 0;
      for (const auto &p : a) {
        auto it = b.find(p.first);
        if (it == b.end()) {
          same = false;
          ++local_deleted;
          ++local_changed;
        } else {
          if (!same_ranges(p.second, it->second)) {
            same = false;
            ++local_changed;
          }
          layer_changed += p.second.layer_ != it->second.layer_;
        }
      }
      for (size_t j = 0; j < std::min(first.size(), second.size()); ++j)
        order = order && first[j].id_ == second[j].id_;
      ++tested[kind];
      order_changes += !order;
      changed_edges += local_changed;
      deleted_edges += local_deleted;
      changed_layers += layer_changed;
      max_changed_edges = std::max(max_changed_edges, local_changed);
      if (!same) {
        ++failed[kind];
        if (examples < 8) {
          std::ofstream out(std::string(argv[1]) + "/counterexample-" +
                            std::to_string(examples) + ".json");
          out << std::setprecision(17) << "{\"trial\":" << trial
              << ",\"seed\":" << seed << ",\"kind\":\"" << names[kind]
              << "\",\"R\":" << R << ",\"input_count\":" << count
              << ",\"changed_edges\":" << local_changed
              << ",\"deleted_edges\":" << local_deleted
              << ",\"order_changed\":" << (!order ? "true" : "false")
              << ",\"layer_changed\":" << layer_changed << ",\"input_order\":[";
          for (size_t j = 0; j < original.size(); ++j) {
            if (j)
              out << ",";
            out << original[j].id;
          }
          out << "],\"vectors\":[";
          for (unsigned i = 0; i <= count; ++i) {
            if (i)
              out << ",";
            out << "{\"id\":" << i << ",\"x\":[";
            for (unsigned d = 0; d < dim; ++d) {
              if (d)
                out << ",";
              out << index._data1[size_t(i) * dim + d];
            }
            out << "],\"y\":[";
            for (unsigned d = 0; d < dim; ++d) {
              if (d)
                out << ",";
              out << index._data2[size_t(i) * dim + d];
            }
            out << "]}";
          }
          out << "],\"first\":";
          dump_edges(out, first);
          out << ",\"second\":";
          dump_edges(out, second);
          out << "}\n";
          ++examples;
        }
      }
    }
    unsigned failures = std::accumulate(failed, failed + 8, 0u);
    std::ofstream out(std::string(argv[1]) + "/result.json");
    out << "{\"status\":\"" << (failures ? "FAIL" : "PASS")
        << "\",\"tests\":" << trials << ",\"failed\":" << failures
        << ",\"R\":" << R << ",\"distance_dim\":" << dim << ",\"seed\":" << seed
        << ",\"changed_edges\":" << changed_edges
        << ",\"deleted_edges\":" << deleted_edges
        << ",\"order_changes\":" << order_changes
        << ",\"changed_layers\":" << changed_layers
        << ",\"max_changed_edges_in_case\":" << max_changed_edges
        << ",\"counterexamples\":" << examples << ",\"groups\":[";
    for (unsigned k = 0; k < 8; ++k) {
      if (k)
        out << ",";
      out << "{\"kind\":\"" << names[k] << "\",\"tested\":" << tested[k]
          << ",\"failed\":" << failed[k] << "}";
    }
    out << "],\"scope\":\"two calls to exact formal pruner; no inter_insert "
           "invoked; supplied saved order retained at API entry; range "
           "endpoint equality is bitwise; not a dataset edge-error "
           "estimate\"}\n";
    out.close();
    std::cout << (failures ? "FAIL" : "PASS") << " idempotence cases=" << trials
              << " failed=" << failures << " changed_edges=" << changed_edges
              << " deleted_edges=" << deleted_edges << "\n";
    return failures ? 2 : 0;
  }
  catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
