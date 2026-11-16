#include "index.h"
#include "function_io.h"
#include <iostream>
struct ReplayDistance : pipeann::Distance<float> {
  fnio::Oracle *oracle;
  explicit ReplayDistance(fnio::Oracle *o) : oracle(o) {}
  float compare(const float *a, const float *b, unsigned d) const override {
    return oracle->raw(a, b, d);
  }
};
int main(int argc, char **argv) {
  try {
    if (argc != 5)
      throw std::runtime_error("trace base1 base2 output");
    unsigned n, d, n2, d2;
    auto x = fnio::matrix(argv[2], n, d), y = fnio::matrix(argv[3], n2, d2);
    if (n != n2)
      throw std::runtime_error("paired bases");
    auto *index = new pipeann::HybridIndex<float>(pipeann::Metric::L2, d, d2, n,
                                                  false, false, false);
    std::memcpy(index->_data1, x.data(), x.size() * 4);
    std::memcpy(index->_data2, y.data(), y.size() * 4);
    fnio::Oracle oracle;
    oracle.x = index->_data1;
    oracle.y = index->_data2;
    oracle.n = n;
    oracle.d1 = d;
    oracle.d2 = d2;
    delete index->_distance;
    index->_distance = new ReplayDistance(&oracle);
    pipeann::Parameters p;
    p.Set<unsigned>("R", 40);
    p.Set<unsigned>("C", 750);
    p.Set<unsigned>("L", 200);
    std::ifstream input(argv[1], std::ios::binary);
    fnio::header(input, "HVSREV2");
    std::ofstream out(argv[4], std::ios::binary);
    out.write("HVSOUT2", 8);
    fnio::Record r;
    unsigned count = 0;
    while (fnio::read(input, r)) {
      oracle.select(r);
      std::vector<pipeann::NeighborH> pool;
      for (auto v : r.candidates)
        pool.emplace_back(v.id, v.e, v.g, true, 0);
      std::vector<pipeann::DEGNeighbor> selected;
      index->prune_neighbors_no_opt(pool, p, selected);
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
    std::cout << "REPLAY_B cases=" << count << "\n";
    std::exit(0);
  }
  catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
