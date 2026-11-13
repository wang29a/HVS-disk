#define main graph_builder_main
#include "build_graph.cpp"
#undef main
#include "metric_contract.h"

int main() {
  if (hvs_metric_distance(25, pipeann::Metric::L2) != 5)
    return 1;
  const double root_a = .5 * std::sqrt(1.) + .5 * std::sqrt(9.);
  const double root_b = .5 * std::sqrt(4.84) + .5 * std::sqrt(4.84);
  if (!(root_a < root_b && .5 * 1 + .5 * 9 > 4.84))
    return 2;
  for (Intervals intervals : { Intervals{},
                               { { 0, 0 } },
                               { { 0, 1 } },
                               { { .101f, .109f } },
                               { { 0, .01f }, { .9f, 1 } },
                               { { .1f, .2f }, { .7f, .9f } } }) {
    const auto q = quantize(intervals);
    for (auto p : intervals) {
      bool contains = false;
      for (auto r : q)
        contains |= r.first / 100. <= p.first && r.second / 100. >= p.second;
      if (!contains)
        return 3;
    }
  }
  std::cout << "PASS metric reversal, squared-kernel contract, conservative "
               "interval boundaries\n";
  return 0;
}
