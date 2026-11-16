#pragma once
#include "function_io.h"
#include <cstdlib>
namespace hvs_capture {
template <class Candidates, class Selected, class Compare>
void record(unsigned target, unsigned new_id, const Candidates &c,
            const Selected &selected, Compare compare) {
  static const char *path = std::getenv("HVS_FUNCTION_CAPTURE");
  if (!path)
    return;
  static uint64_t ordinal = 0, saved = 0;
  uint64_t k = ordinal++;
  if (k % 23 || saved >= 10000)
    return;
  static std::ofstream out(path, std::ios::binary);
  if (saved == 0)
    out.write("HVSREV2", 8);
  fnio::put(out, k);
  fnio::put(out, target);
  fnio::put(out, new_id);
  fnio::put(out, unsigned(c.size()));
  for (auto &v : c) {
    fnio::put(out, v.id);
    fnio::put(out, v.distance1);
    fnio::put(out, v.distance2);
  }
  for (auto &a : c)
    for (auto &b : c)
      for (unsigned space = 0; space < 2; ++space)
        fnio::put(out, compare(a.id, b.id, space));
  fnio::put(out, unsigned(selected.size()));
  for (const auto &v : selected)
    fnio::edge(out, { v.id_,    v.emb_distance_,  v.geo_distance_,
                      v.layer_, v.available_range });
  out.flush();
  if (!out)
    throw std::runtime_error("capture write");
  ++saved;
}
}
