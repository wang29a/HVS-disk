#include "aux_utils.h"
#include "index.h"
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

using Ranges = std::vector<std::pair<float, float> >;

int main() {
  pipeann::HybridIndex<float> index(pipeann::Metric::L2, 8, 8, 1, false, false);
  const std::vector<std::pair<Ranges, Ranges> > cases = {
    { {}, {} },
    { { { 0, 1 } }, { { 0, 1 } } },
    { { { .1f, .2f }, { .7f, .9f } }, { { .1f, .2f }, { .7f, .9f } } },
    { { { .1f, .4f }, { .2f, .8f } }, { { .1f, .8f } } },
    { { { .1f, .8f }, { .2f, .3f } }, { { .1f, .8f } } },
    { { { 0, .5f }, { .5f, 1 } }, { { 0, 1 } } },
    { { { .8f, 1 }, { .1f, .3f }, { .2f, .9f } }, { { .1f, 1 } } }
  };
  for (size_t i = 0; i < cases.size(); ++i) {
    auto input = cases[i].first;
    if (pipeann::mergeIntervals(input) != cases[i].second)
      throw std::runtime_error("free function failed case " +
                               std::to_string(i));
    input = cases[i].first;
    if (index.mergeIntervals(input) != cases[i].second)
      throw std::runtime_error("HybridIndex failed case " + std::to_string(i));
  }
  Ranges pruned{ { .1f, .2f }, { .7f, .9f } };
  Ranges expected{ { 0, .1f }, { .2f, .7f }, { .9f, 1 } };
  auto merged = pipeann::mergeIntervals(pruned);
  if (pipeann::get_use_range(merged) != expected)
    throw std::runtime_error("free complement failed");
  merged = index.mergeIntervals(pruned);
  if (index.get_use_range(merged) != expected)
    throw std::runtime_error("HybridIndex complement failed");
  Ranges available;
  index.get_use_range(merged, available);
  if (available != expected)
    throw std::runtime_error("output-parameter complement failed");
  std::cout << "PASS seven union cases and three prune-complement checks\n";
}
