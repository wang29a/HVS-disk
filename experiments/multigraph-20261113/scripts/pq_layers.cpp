// Diagnostic translation unit; original measured executor and binary unchanged.
#define main original_multigraph_main
#include "multigraph.cpp"
#undef main
#include <atomic>
#include <limits>

using Ranked = std::vector<std::pair<unsigned, float> >;
Ranked top_scores(const std::vector<float> &scores, unsigned k) {
  if (!k || scores.size() < k)
    throw std::runtime_error("exhaustive candidate budget");
  std::vector<unsigned> ids(scores.size());
  std::iota(ids.begin(), ids.end(), 0);
  auto less = [&](unsigned a, unsigned b) {
    return scores[a] < scores[b] || (scores[a] == scores[b] && a < b);
  };
  if (k < ids.size())
    std::nth_element(ids.begin(), ids.begin() + k, ids.end(), less);
  ids.resize(k);
  std::sort(ids.begin(), ids.end(), less);
  Ranked out;
  for (auto id : ids)
    out.emplace_back(id, scores[id]);
  return out;
}
Ranked first(const std::vector<pipeann::Neighbor> &ret, unsigned k) {
  if (ret.size() < k)
    throw std::runtime_error("graph returned fewer than k candidates");
  Ranked out;
  for (unsigned i = 0; i < k; i++)
    out.emplace_back(ret[i].id, ret[i].distance);
  return out;
}
void pairs(std::ostream &o, const Ranked &v) {
  o << "[";
  for (U j = 0; j < v.size(); j++) {
    if (j)
      o << ",";
    o << "[" << v[j].first << "," << v[j].second << "]";
  }
  o << "]";
}
void ids_json(std::ostream &o, const std::array<unsigned, 10> &ids) {
  o << "[";
  for (unsigned j = 0; j < 10; j++) {
    if (j)
      o << ",";
    o << ids[j];
  }
  o << "]";
}
std::array<unsigned, 10> rerank(Engine &e, Workspace &w, const Ranked &x,
                                const Ranked &y, float alpha, const float *qx,
                                const float *qy) {
  std::vector<unsigned> ids;
  for (auto p : x)
    ids.push_back(p.first);
  for (auto p : y)
    ids.push_back(p.first);
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  Ranked exact;
  for (auto id : ids) {
    float a = e.distance->compare(qx, e.vector(id, 0, w), e.d1),
          b = e.distance->compare(qy, e.vector(id, 1, w), e.d2);
    exact.emplace_back(id, e.fuse(a, b, alpha));
  }
  std::sort(exact.begin(), exact.end(), [](auto a, auto b) {
    return a.second < b.second || (a.second == b.second && a.first < b.first);
  });
  if (exact.size() < 10)
    throw std::runtime_error("rerank top10");
  std::array<unsigned, 10> out{};
  for (unsigned j = 0; j < 10; j++)
    out[j] = exact[j].first;
  return out;
}
struct ScoreAudit {
  U compared = 0, bit_mismatches = 0;
  double max_relative = 0, max_absolute = 0;
};
Ranked canonical_scores(const Ranked &c, const std::vector<float> &scores,
                        ScoreAudit &audit) {
  Ranked out;
  for (auto p : c) {
    float expected = scores.at(p.first);
    ++audit.compared;
    if (memcmp(&p.second, &expected, 4))
      ++audit.bit_mismatches;
    double error = std::abs(double(p.second) - expected);
    audit.max_absolute = std::max(audit.max_absolute, error);
    if (expected == 0) {
      if (p.second != 0)
        throw std::runtime_error("nonzero PQ score for exact zero reference");
    } else
      audit.max_relative =
          std::max(audit.max_relative, error / std::abs(double(expected)));
    out.emplace_back(p.first, expected);
  }
  if (audit.max_relative > 5e-6)
    throw std::runtime_error("PQ score numerical audit exceeds 5e-6");
  return out;
}
void run_query(Engine &e, unsigned qi, const float *qx, const float *qy,
               unsigned L, unsigned k, const std::string &path) {
  Workspace w;
  w.reset();
  e.pq1.populate_chunk_distances(qx, w.t1.data());
  e.pq2.populate_chunk_distances(qy, w.t2.data());
  std::vector<float> sx(e.n), sy(e.n), score(e.n);
  std::vector<unsigned> ids;
  for (U at = 0; at < e.n; at += 4096) {
    unsigned n = std::min(U(4096), U(e.n) - at);
    ids.resize(n);
    std::iota(ids.begin(), ids.end(), at);
    aggregate_coords(ids.data(), n, e.code1.data(), 64, w.c1.data());
    aggregate_coords(ids.data(), n, e.code2.data(), 64, w.c2.data());
    pq_dist_lookup(w.c1.data(), n, 64, w.t1.data(), w.de.data());
    pq_dist_lookup(w.c2.data(), n, 64, w.t2.data(), w.dy.data());
    std::copy(w.de.begin(), w.de.begin() + n, sx.begin() + at);
    std::copy(w.dy.begin(), w.dy.begin() + n, sy.begin() + at);
  }
  ScoreAudit audit;
  for (U j = 0; j < e.n; j++)
    score[j] = std::sqrt(std::max(0.f, sx[j]));
  auto bx = top_scores(score, k);
  Answer a;
  auto cx = canonical_scores(
      first(e.traverse(0, 0, .5f, L, false, qx, qy, w, a, false), k), score,
      audit);
  for (U j = 0; j < e.n; j++)
    score[j] = std::sqrt(std::max(0.f, sy[j]));
  auto by = top_scores(score, k);
  auto cy = canonical_scores(
      first(e.traverse(1, 1, .5f, L, false, qx, qy, w, a, false), k), score,
      audit);
  std::ofstream o(path);
  o << std::setprecision(17) << "{\"query_id\":" << qi << ",\"k\":" << k
    << ",\"L\":" << L << ",\"image\":{\"B\":";
  pairs(o, bx);
  o << ",\"C\":";
  pairs(o, cx);
  o << "},\"text\":{\"B\":";
  pairs(o, by);
  o << ",\"C\":";
  pairs(o, cy);
  o << "},\"weights\":[";
  const float alphas[] = { .1f, .3f, .5f, .7f, .9f };
  for (unsigned z = 0; z < 5; z++) {
    float alpha = alphas[z];
    if (z)
      o << ",";
    for (U j = 0; j < e.n; j++)
      score[j] = e.fuse(sx[j], sy[j], alpha);
    auto b = top_scores(score, k);
    o << "{\"alpha\":" << alpha << ",\"M_native_C_ids\":";
    ids_json(o, rerank(e, w, cx, cy, alpha, qx, qy));
    o << ",\"unified_B\":";
    pairs(o, b);
    for (bool filter : { false, true }) {
      Answer a;
      auto c = canonical_scores(
          first(e.traverse(7, -1, alpha, L, filter, qx, qy, w, a, false), k),
          score, audit);
      o << (filter ? ",\"unified_on\":{\"C\":" : ",\"unified_off\":{\"C\":");
      pairs(o, c);
      o << ",\"native_ids\":";
      ids_json(o, rerank(e, w, c, {}, alpha, qx, qy));
      o << ",\"expanded\":" << a.graph[7].expanded
        << ",\"pq_space_evals\":" << a.graph[7].pq << "}";
    }
    o << "}";
  }
  o << "],\"pq_score_audit\":{\"compared\":" << audit.compared
    << ",\"bit_mismatches\":" << audit.bit_mismatches
    << ",\"max_relative_error\":" << audit.max_relative
    << ",\"max_absolute_error\":" << audit.max_absolute << "}}\n";
  if (!o)
    throw std::runtime_error("diagnostic write");
}
int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--self-test") {
      auto v = top_scores({ 2, 1, 1, 0, 3 }, 3);
      if (v != Ranked{ { 3, 0 }, { 1, 1 }, { 2, 1 } })
        throw std::runtime_error("canonical ties");
      std::cout << "PASS canonical PQ top-k\n";
      return 0;
    }
    if (argc != 9)
      throw std::runtime_error(
          "manifest qx qy queryIDs output-directory L k threads");
    unsigned n, d, n2, d2;
    auto qx = matrix<float>(argv[2], n, d), qy = matrix<float>(argv[3], n2, d2);
    if (n != n2)
      throw std::runtime_error("query rows");
    std::ifstream f(argv[4]);
    std::vector<unsigned> queries;
    unsigned id;
    while (f >> id) {
      if (id >= n)
        throw std::runtime_error("query ID");
      queries.push_back(id);
    }
    if (queries.empty())
      throw std::runtime_error("empty queries");
    unsigned L = std::stoul(argv[6]), k = std::stoul(argv[7]),
             threads = std::stoul(argv[8]);
    if (!threads || threads > 4 || L < k || k < 10)
      throw std::runtime_error("diagnostic parameters");
    omp_set_num_threads(threads);
    Engine e(argv[1], "dram", "pq", { 0, 1, 7 });
    if (e.d1 != d || e.d2 != d2)
      throw std::runtime_error("dimensions");
    std::string error;
    std::atomic<unsigned> done{ 0 };
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t j = 0; j < int64_t(queries.size()); j++)
      try {
        auto qi = queries[j];
        run_query(e, qi, qx.data() + U(qi) * d, qy.data() + U(qi) * d2, L, k,
                  std::string(argv[5]) + "/q-" + std::to_string(qi) + ".json");
        std::cerr << "queries_done=" << ++done << "/" << queries.size() << "\n";
      }
    catch (const std::exception &ex) {
#pragma omp critical
      error = ex.what();
    }
    if (!error.empty())
      throw std::runtime_error(error);
    return 0;
  }
  catch (const std::exception &ex) {
    std::cerr << ex.what() << "\n";
    return 1;
  }
}
