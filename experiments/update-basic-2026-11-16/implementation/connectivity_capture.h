#pragma once
#include "index.h"
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <stdexcept>

inline std::string hvs_connectivity_path(const char *name) {
  const char *p = std::getenv("HVS_CONNECTIVITY_DIAG_DIR");
  if (!p) throw std::runtime_error("missing capture output directory");
  return std::string(p)+"/"+name;
}
inline void hvs_diag_candidates(unsigned owner,unsigned origin,const char *kind,
                                 const std::vector<pipeann::NeighborH> &pool) {
  if (owner != 118 && owner != 8947) return;
  static std::ofstream out(hvs_connectivity_path("construction-candidates.csv"));
  static bool header = false;
  if (!header) { out << "owner,origin,kind,candidate,dist_x,dist_y\n";header = true; }
  for (const auto &v : pool)
    out << owner << ',' << origin << ',' << kind << ',' << v.id << ','
        << std::setprecision(17) << v.distance1 << ',' << v.distance2 << '\n';
  out.flush();
  if (!out) throw std::runtime_error("candidate capture write");
}
inline void hvs_diag_selected(unsigned owner,unsigned origin,const char *kind,
                               const std::vector<pipeann::DEGNeighbor> &pool) {
  if (owner != 118 && owner != 8947) return;
  static std::ofstream out(hvs_connectivity_path("construction-selected.csv"));
  static bool header = false;
  if (!header) { out << "owner,origin,kind,selected,dist_x,dist_y\n";header = true; }
  for (const auto &v : pool)
    out << owner << ',' << origin << ',' << kind << ',' << v.id_ << ','
        << std::setprecision(17) << v.emb_distance_ << ',' << v.geo_distance_ << '\n';
  out.flush();
  if (!out) throw std::runtime_error("selected capture write");
}
