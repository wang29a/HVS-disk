#pragma once
#include <cmath>
#include <cstdlib>
#include <stdexcept>
inline float hvs_unified_rho() {
  static const float rho = [] {
    const char *s = std::getenv("HVS_UNIFIED_RHO");
    if (!s) throw std::runtime_error("HVS_UNIFIED_RHO required");
    char *end = nullptr;
    const float v = std::strtof(s, &end);
    if (!end || *end || !std::isfinite(v) || v < 1 || v > 1.05f)
      throw std::runtime_error("HVS_UNIFIED_RHO outside frozen [1,1.05]");
    return v;
  }();
  return rho;
}
