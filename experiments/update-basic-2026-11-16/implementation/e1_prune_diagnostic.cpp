#include "deg_prune.h"
#include "pq_table.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <unordered_map>
using namespace hvs_update;
static void check(bool b, const char *s) {
  if (!b)
    throw std::runtime_error(s);
}
template <class T>
static std::vector<T> matrix(const std::string &p, unsigned &n, unsigned &d) {
  std::ifstream f(p, std::ios::binary);
  f.read(reinterpret_cast<char *>(&n), 4);
  f.read(reinterpret_cast<char *>(&d), 4);
  check(bool(f) && n && d, "matrix header");
  std::vector<T> v(uint64_t(n) * d);
  f.read(reinterpret_cast<char *>(v.data()), v.size() * sizeof(T));
  check(bool(f) && f.peek() == EOF, "matrix size");
  return v;
}
static float l2(const float *a, const float *b, unsigned d) {
  double sum = 0;
  for (unsigned j = 0; j < d; ++j) {
    double v = double(a[j]) - b[j];
    sum += v * v;
  }
  return float(std::sqrt(sum));
}
int main(int argc, char **argv) {
  try {
    check(argc == 3, "diagnostic-directory frozen-pq-prefix");
    std::string root = argv[1], prefix = argv[2];
    unsigned n, d, n2, d2, nc, c, nc2, c2;
    auto x = matrix<float>(root + "/vectors-x.bin", n, d),
         y = matrix<float>(root + "/vectors-y.bin", n2, d2);
    auto cx = matrix<uint8_t>(root + "/codes-x.bin", nc, c),
         cy = matrix<uint8_t>(root + "/codes-y.bin", nc2, c2);
    check(n == n2 && n == nc && n == nc2 && c == c2, "matrix shapes");
    std::ifstream mapping(root + "/rows.txt");
    unsigned id, row, tag;
    std::unordered_map<unsigned, unsigned> rows;
    std::vector<unsigned> ids(n);
    while (mapping >> id >> row >> tag) {
      check(row < n && rows.emplace(id, row).second, "ID mapping");
      ids[row] = id;
    }
    check(rows.size() == n, "mapping count");
    pipeann::FixedChunkPQTable<float> pq1, pq2;
    pq1.load_pq_centroid_bin((prefix + "emb_pq_pivots.bin").c_str(), c);
    pq2.load_pq_centroid_bin((prefix + "loc_pq_pivots.bin").c_str(), c2);
    unsigned encoded = 0, mismatches = 0;
    for (unsigned j = 0; j < n; ++j)
      if (ids[j] >= 1800000) {
        std::vector<uint8_t> a(c), b(c2);
        pq1.deflate_vec(x.data() + uint64_t(j) * d, a.data());
        pq2.deflate_vec(y.data() + uint64_t(j) * d2, b.data());
        mismatches +=
            !std::equal(a.begin(), a.end(), cx.data() + uint64_t(j) * c) ||
            !std::equal(b.begin(), b.end(), cy.data() + uint64_t(j) * c2);
        std::copy(a.begin(), a.end(), cx.data() + uint64_t(j) * c);
        std::copy(b.begin(), b.end(), cy.data() + uint64_t(j) * c2);
        ++encoded;
      }
    std::ifstream cases(root + "/cases.txt");
    std::ofstream out(root + "/prune-diagnostic.json"),
        distances(root + "/candidate-distances.csv");
    distances << "target,source,candidate,source_exact_x,source_exact_y,target_"
                 "exact_x,target_exact_y,source_PQ_x,source_PQ_y,target_PQ_x,"
                 "target_PQ_y\n" << std::setprecision(17);
    out << "{\"scope\":\"constructed candidate pools from before/after graphs; "
           "not captured historical runtime "
           "pools\",\"new_vectors_encoded_with_actual_update_method\":"
        << encoded
        << ",\"differences_from_frozen_full_data_codes\":" << mismatches
        << ",\"cases\":[";
    bool first = true;
    unsigned target, source, old_count, added_count;
    while (cases >> target >> source >> old_count >> added_count) {
      std::vector<unsigned> old(old_count), added(added_count);
      for (auto &v : old)
        cases >> v;
      for (auto &v : added)
        cases >> v;
      check(bool(cases), "case row");
      PairDistance exact = [&](uint32_t a, uint32_t b) {
        auto i = rows.at(a), j = rows.at(b);
        return DistancePair(
            l2(x.data() + uint64_t(i) * d, x.data() + uint64_t(j) * d, d),
            l2(y.data() + uint64_t(i) * d2, y.data() + uint64_t(j) * d2, d2));
      };
      PairDistance pq = [&](uint32_t a, uint32_t b) {
        float u, v;
        auto i = rows.at(a), j = rows.at(b);
        pq1.compute_distances_alltoall(cx.data() + uint64_t(i) * c,
                                       cx.data() + uint64_t(j) * c, &u, 1);
        pq2.compute_distances_alltoall(cy.data() + uint64_t(i) * c2,
                                       cy.data() + uint64_t(j) * c2, &v, 1);
        return DistancePair(std::sqrt(std::max(0.f, u)),
                            std::sqrt(std::max(0.f, v)));
      };
      if (!first)
        out << ',';
      first = false;
      out << "{\"target\":" << target << ",\"source\":" << source
          << ",\"before_candidates\":" << old_count
          << ",\"added_final_neighbors\":" << added_count << ",\"tests\":[";
      bool first_test = true;
      for (bool include_new : { false, true })
        for (bool precise : { false, true }) {
          auto pool = old;
          if (include_new)
            pool.insert(pool.end(), added.begin(), added.end());
          std::sort(pool.begin(), pool.end());
          pool.erase(std::unique(pool.begin(), pool.end()), pool.end());
          PairDistance provider = precise ? exact : pq;
          std::vector<pipeann::NeighborH> candidates;
          for (auto id : pool) {
            auto v = provider(source, id);
            candidates.emplace_back(id, v.first, v.second, true, 0);
          }
          auto selected = prune(std::move(candidates), 40, provider);
          if (!first_test)
            out << ',';
          first_test = false;
          out << "{\"pool\":\""
              << (include_new ? "before_plus_newly_retained" : "before_only")
              << "\",\"distance\":\"" << (precise ? "exact" : "pq")
              << "\",\"selected_count\":" << selected.size()
              << ",\"target_kept\":"
              << (std::find(selected.begin(), selected.end(), target) !=
                          selected.end()
                      ? "true"
                      : "false") << ",\"selected\":[";
          for (unsigned j = 0; j < selected.size(); ++j) {
            if (j)
              out << ',';
            out << selected[j];
          }
          out << "]}";
        }
      out << "]}";
      old.insert(old.end(), added.begin(), added.end());
      std::sort(old.begin(), old.end());
      old.erase(std::unique(old.begin(), old.end()), old.end());
      for (auto id : old) {
        auto a = exact(source, id), b = exact(target, id), u = pq(source, id),
             v = pq(target, id);
        distances << target << ',' << source << ',' << id << ',' << a.first
                  << ',' << a.second << ',' << b.first << ',' << b.second << ','
                  << u.first << ',' << u.second << ',' << v.first << ','
                  << v.second << '\n';
      }
    }
    out << "]}\n";
    check(bool(out) && bool(distances), "output write");
    std::cout << "OFFLINE_PRUNE_DIAGNOSTIC_COMPLETE new_codes=" << encoded
              << " mismatches=" << mismatches << std::endl;
  }
  catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
