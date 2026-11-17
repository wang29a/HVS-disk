#pragma once
#include "neighbor.h"
#include <functional>
#include <utility>
#include <vector>

namespace hvs_update {
using DistancePair = std::pair<float, float>;
using PairDistance = std::function<DistancePair(uint32_t, uint32_t)>;
// Both components are rooted L2, never squared L2. Ranges are temporary
// selection state; neither the graph store nor the query path serializes them.
std::vector<pipeann::DEGNeighbor>
prune_detailed(std::vector<pipeann::NeighborH> candidates, unsigned degree,
               const PairDistance &distance);
std::vector<uint32_t> prune(std::vector<pipeann::NeighborH> candidates,
                            unsigned degree, const PairDistance &distance);
} // namespace hvs_update
