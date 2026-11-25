#pragma once
#include "utils.h"
#include <algorithm>
#include <cmath>
inline float hvs_metric_distance(float squared, pipeann::Metric metric) {
  return metric == pipeann::Metric::L2 ? std::sqrt(std::max(0.0f, squared)) : squared;
}
