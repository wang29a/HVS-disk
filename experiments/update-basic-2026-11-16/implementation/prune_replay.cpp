#include "deg_prune.h"
#include "function_io.h"
#include <cmath>
#include <iostream>
#include <cstring>
static bool same_float(float a, float b) {
  return std::memcmp(&a, &b, sizeof(float)) == 0;
}
int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::runtime_error("prune_replay capture.bin");
    std::ifstream input(argv[1], std::ios::binary);
    fnio::header(input, "HVSREV2");
    fnio::Record r;
    unsigned count = 0;
    while (fnio::read(input, r)) {
      std::vector<pipeann::NeighborH> pool;
      std::unordered_map<unsigned, unsigned> local;
      for (unsigned i = 0; i < r.candidates.size(); ++i) {
        const auto &v = r.candidates[i];
        local[v.id] = i;
        pool.emplace_back(v.id, v.e, v.g, true, 0);
      }
      auto result =
          hvs_update::prune_detailed(pool, 40, [&](unsigned a, unsigned b) {
            auto k = (size_t(local.at(a)) * pool.size() + local.at(b)) * 2;
            return std::make_pair(std::sqrt(std::max(0.f, r.table[k])),
                                  std::sqrt(std::max(0.f, r.table[k + 1])));
          });
      if (result.size() != r.expected.size())
        throw std::runtime_error("degree mismatch");
      for (size_t i = 0; i < result.size(); ++i) {
        const auto &a = result[i];
        const auto &b = r.expected[i];
        if (a.id_ != b.id || a.layer_ != b.layer ||
            !same_float(a.emb_distance_, b.e) ||
            !same_float(a.geo_distance_, b.g) ||
            a.available_range.size() != b.ranges.size())
          throw std::runtime_error("selection mismatch case " +
                                   std::to_string(r.ordinal));
        for (size_t j = 0; j < b.ranges.size(); ++j)
          if (!same_float(a.available_range[j].first, b.ranges[j].first) ||
              !same_float(a.available_range[j].second, b.ranges[j].second))
            throw std::runtime_error("interval mismatch");
      }
      ++count;
    }
    if (count != 10000)
      throw std::runtime_error("incomplete semantic replay");
    std::cout << "PRUNE_REPLAY_PASS cases=" << count << " bit_exact=true\n";
    return 0;
  }
  catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
