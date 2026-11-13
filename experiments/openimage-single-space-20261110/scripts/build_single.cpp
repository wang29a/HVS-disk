#include "aux_utils.h"
#include "index.h"
#include "omp.h"
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: build-single <float32 base bin> <index prefix>\n";
    return 2;
  }
  const std::string base = argv[1], prefix = argv[2];
  const std::string graph = prefix + "_mem.index";
  const std::string disk = prefix + "_disk.index";
  omp_set_num_threads(2);
  size_t nodes, dim;
  pipeann::get_bin_metadata(base, nodes, dim);
  pipeann::Parameters parameters;
  parameters.Set<unsigned>("L", 100);
  parameters.Set<unsigned>("R", 40);
  parameters.Set<unsigned>("C", 750);
  parameters.Set<unsigned>("num_threads", 2);
  parameters.Set<float>("alpha", 1.2f);
  parameters.Set<bool>("saturate_graph", true);
  parameters.Set<std::string>("save_path", graph);
  // Leave out index_prefix_path to disable the legacy optional page reorder.
  pipeann::Index<float> index(pipeann::Metric::L2, dim, nodes, false, false);
  index.build(base.c_str(), nodes, parameters);
  index.save(graph.c_str());
  pipeann::create_disk_layout<float>(graph, base, "", prefix + "_pq_pivots.bin",
                                     prefix + "_pq_compressed.bin", false,
                                     disk);
  std::cout << "SINGLE_SPACE_BUILD_COMPLETE\n";
}
