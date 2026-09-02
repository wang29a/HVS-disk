#pragma once
#include <cstdint>
#include <fcntl.h>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <malloc.h>

#include <unistd.h>

#include "neighbor.h"
#include "parameters.h"
#include "tsl/robin_set.h"
#include "utils.h"

namespace pipeann {
  const size_t MAX_PQ_TRAINING_SET_SIZE = 256000;
  const size_t MAX_SAMPLE_POINTS_FOR_WARMUP = 1000000;
  const double PQ_TRAINING_SET_FRACTION = 0.1;
  const double SPACE_FOR_CACHED_NODES_IN_GB = 0.25;
  const double THRESHOLD_FOR_CACHING_IN_GB = 1.0;
  const uint32_t WARMUP_L = 20;

  template<typename T, typename TagT>
  class SSDIndex;

  double get_memory_budget(const std::string &mem_budget_str);
  double get_memory_budget(double search_ram_budget_in_gb);
  void add_new_file_to_single_index(std::string index_file, std::string new_file);

  size_t calculate_num_pq_chunks(double final_index_ram_limit, size_t points_num, uint32_t dim);

  double calculate_recall(unsigned num_queries, unsigned *gold_std, float *gs_dist, unsigned dim_gs,
                          unsigned *our_results, unsigned dim_or, unsigned recall_at);

  double calculate_recall(unsigned num_queries, unsigned *gold_std, float *gs_dist, unsigned dim_gs,
                          unsigned *our_results, unsigned dim_or, unsigned recall_at,
                          const tsl::robin_set<unsigned> &active_tags);

  void read_idmap(const std::string &fname, std::vector<unsigned> &ivecs);

  int merge_shards(const std::string &vamana_prefix, const std::string &vamana_suffix, const std::string &idmaps_prefix,
                   const std::string &idmaps_suffix, const _u64 nshards, unsigned max_degree,
                   const std::string &output_vamana, const std::string &medoids_file);

  template<typename T>
  int merge_shards_h(const std::string &vamana_prefix, const std::string &vamana_suffix, const std::string &idmaps_prefix,
                   const std::string &idmaps_suffix, const _u64 nshards, unsigned max_degree,
                   const std::string &output_vamana, const std::string &medoids_file,
                   pipeann::Metric m, const pipeann::Parameters &pars);

  template<typename T>
  int build_merged_vamana_index(std::string base_file, pipeann::Metric _compareMetric, bool single_index_file,
                                unsigned L, unsigned R, double sampling_rate, double ram_budget,
                                std::string mem_index_path, std::string medoids_file, std::string centroids_file,
                                const char *tag_file = nullptr);

  template<typename T>
  int build_merged_vamana_index(std::string base1_file, std::string base2_file, pipeann::Metric _compareMetric,
                                bool single_index_file, unsigned L, unsigned R, double sampling_rate, double ram_budget,
                                std::string mem_index_path1, std::string medoids_file1, std::string centroids_file1,
                                std::string mem_index_path2, std::string medoids_file2, std::string centroids_file2,
                                std::string mem_index_path, std::string medoids_file,
                                const char *tag_file = nullptr);

  template<typename T, typename TagT = uint32_t>
  bool build_disk_index(const char *dataFilePath, const char *indexFilePath, const char *indexBuildParameters,
                        pipeann::Metric _compareMetric, bool single_file_index, const char *tag_file = nullptr);

  template<typename T, typename TagT = uint32_t>
  bool build_disk_index(const char *data1FilePath, const char *data2FilePath, const char *indexFilePath,
                        const char *indexBuildParameters, pipeann::Metric _compareMetric, bool single_file_index,
                        const char *tag_file = nullptr);

  template<typename T, typename TagT = uint32_t>
  bool build_disk_index_py(const char *dataPath, const char *indexFilePath, uint32_t R, uint32_t L, uint32_t M,
                           uint32_t num_threads, uint32_t PQ_bytes, pipeann::Metric _compareMetric,
                           bool single_file_index, const char *tag_file);

  template<typename T, typename TagT = uint32_t>
  void create_disk_layout(const std::string &mem_index_file, const std::string &base_file, const std::string &tag_file,
                          const std::string &pq_pivots_file, const std::string &pq_compressed_vectors_file,
                          bool single_file_index, const std::string &output_file);

  template<typename T, typename TagT = uint32_t>
  void create_disk_layout(const std::string &mem_index_file, const std::string &base1_file,
                          const std::string &base2_file, const std::string &medoids_file,
                          bool single_file_index, const std::string &meta_file,
                          const std::string &topo_file, const std::string &data_file,
                          int merge_method = 0);

  template<typename T, typename TagT = uint32_t>
  void create_disk_layout_single_file_aligned(
                const std::string &mem_index_file, const std::string &base1_file,
                const std::string &base2_file, const std::string &medoids_file,
                const std::string &output_file, int merge_method = 0);

  // 将带 alpha range 的 topo 索引转换为无 alpha 版本
  void convert_topo_remove_alpha(const std::string &input_meta_file,
                                  const std::string &input_topo_file,
                                  const std::string &output_meta_file,
                                  const std::string &output_topo_file);

  // 将带 alpha range 的 topo 索引合并压缩到最多 kMergedMaxAlphaRangeLen pairs
  // merge_method: 2=greedy gap merge, 3=max-coverage
  void convert_topo_merge_alpha(const std::string &input_meta_file,
                                 const std::string &input_topo_file,
                                 const std::string &output_meta_file,
                                 const std::string &output_topo_file,
                                 int merge_method);

  template<typename T>
  void UpdateEnterpointSet(std::vector<NeighborH> &enterpoints_skyeline, std::vector<uint32_t> &enterpoints,
                           T* emb_center, T* loc_center, uint32_t id,
                           Metric m, const Parameters &pars);

  template<typename T>
  int merge_two_graphs(const std::string &graph1_file, const std::string &graph2_file,
                      const std::string &medoids1_file, const std::string &medoids2_file,
                       unsigned max_degree, unsigned L, const std::string &output_graph_file,
                       const std::string &output_medoids_file, pipeann::Metric m, const pipeann::Parameters &paras);

  template<typename T>
  void prune_neighbors(std::vector<NeighborH> pool, unsigned R, pipeann::Metric m,
                        std::vector<DEGNeighbor> &pruned_list, const pipeann::Parameters &pars);
  template<typename T>
  void prune_neighbors_no_opt(std::vector<NeighborH> pool, unsigned R, pipeann::Metric m,
                        std::vector<DEGNeighbor> &pruned_list, const pipeann::Parameters &pars);

  void intersection(const std::vector<std::pair<float, float>> &picked_available_range,
                    const std::pair<float, float> &use_range,
                    std::vector<std::pair<float, float>> &shared_use_range);
  
  void get_use_range(std::vector<std::pair<float, float>> &prune_range, std::vector<std::pair<float, float>> &after_pruned_use_range);

  std::vector<std::pair<float, float>> get_use_range(std::vector<std::pair<float, float>> &intervals);

  std::vector<std::pair<float, float>> mergeIntervals(std::vector<std::pair<float, float>> &intervals);
}  // namespace pipeann
