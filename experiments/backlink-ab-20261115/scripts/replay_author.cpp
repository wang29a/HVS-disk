#include "component.h"
#include "function_io.h"
#include <iostream>
static fnio::Oracle *current_oracle = nullptr;
float hvs_author_oracle(const float *a, const float *b, unsigned d) {
  if (!current_oracle)
    throw std::runtime_error("no oracle");
  return std::sqrt(std::max(0.f, current_oracle->raw(a, b, d)));
}
int main(int argc, char **argv) {
  try {
    if (argc != 5)
      throw std::runtime_error("trace base1 base2 output");
    unsigned n, d, n2, d2;
    auto x = fnio::matrix(argv[2], n, d), y = fnio::matrix(argv[3], n2, d2);
    if (n != n2)
      throw std::runtime_error("paired bases");
    auto *index = new stkq::Index(1.f, 1.f);
    index->setBaseEmbData(x.data());
    index->setBaseLocData(y.data());
    index->setBaseLen(n);
    index->setBaseEmbDim(d);
    index->setBaseLocDim(d2);
    fnio::Oracle oracle;
    oracle.x = x.data();
    oracle.y = y.data();
    oracle.n = n;
    oracle.d1 = d;
    oracle.d2 = d2;
    current_oracle = &oracle;
    stkq::ComponentDEGPruneHeuristic prune(index);
    std::ifstream input(argv[1], std::ios::binary);
    fnio::header(input, "HVSREV2");
    std::ofstream out(argv[4], std::ios::binary);
    out.write("HVSOUT2", 8);
    fnio::Record r;
    unsigned count = 0;
    while (fnio::read(input, r)) {
      oracle.select(r);
      std::vector<stkq::Index::DEGNNDescentNeighbor> pool;
      for (auto v : r.candidates)
        pool.emplace_back(v.id, v.e, v.g, true, 0);
      std::vector<stkq::Index::DEGNeighbor> selected;
      prune.PruneInner(pool, 40, selected);
      fnio::put(out, r.ordinal);
      std::vector<fnio::Edge> result;
      for (auto v : selected)
        result.push_back({ v.id_,    v.emb_distance_,  v.geo_distance_,
                           v.layer_, v.available_range });
      fnio::edges(out, result);
      ++count;
    }
    out.close();
    if (!out)
      throw std::runtime_error("output write");
    std::cout << "REPLAY_AUTHOR cases=" << count << "\n";
    std::exit(0);
  }
  catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
