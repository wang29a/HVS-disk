#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>
#ifdef USE_MKL
    #include <mkl_cblas.h>
#else
    #include <cblas.h>
#endif
#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>

#include "aux_utils.h"
#include "cached_io.h"
#include "index.h"
#include "log.h"
#include "neighbor.h"
#include "omp.h"
#include "parameters.h"
#include "partition_and_pq.h"
#include "percentile_stats.h"
#include "ssd_index.h"
#include "utils.h"

#include "ssd_index.h"
#include "tsl/robin_set.h"
#include "utils.h"
#include "global_stats.h"

#define NUM_KMEANS 15

namespace pipeann {

  void add_new_file_to_single_index(std::string index_file, std::string new_file) {
    std::unique_ptr<_u64[]> metadata;
    _u64 nr, nc;
    pipeann::load_bin<_u64>(index_file, metadata, nr, nc, 0);
    if (nc != 1) {
      LOG(ERROR) << "Error, index file specified does not have correct metadata. ";
      crash();
    }
    size_t index_ending_offset = metadata[nr - 1];
    _u64 read_blk_size = 64 * 1024 * 1024;
    cached_ofstream writer(index_file, read_blk_size, index_ending_offset);
    _u64 check_file_size = get_file_size(index_file);
    if (check_file_size != index_ending_offset) {
      LOG(ERROR) << "Error, index file specified does not have correct metadata "
                    "(last entry must match the filesize). ";
      crash();
    }

    cached_ifstream reader(new_file, read_blk_size);
    size_t fsize = reader.get_file_size();
    if (fsize == 0) {
      LOG(ERROR) << "Error, new file specified is empty. Not appending.";
      crash();
    }

    size_t num_blocks = DIV_ROUND_UP(fsize, read_blk_size);
    char *dump = new char[read_blk_size];
    for (_u64 i = 0; i < num_blocks; i++) {
      size_t cur_block_size = read_blk_size > fsize - (i * read_blk_size) ? fsize - (i * read_blk_size) : read_blk_size;
      reader.read(dump, cur_block_size);
      writer.write(dump, cur_block_size);
    }
    reader.close();
    writer.close();

    delete[] dump;
    std::vector<_u64> new_meta;
    for (_u64 i = 0; i < nr; i++)
      new_meta.push_back(metadata[i]);
    new_meta.push_back(metadata[nr - 1] + fsize);

    pipeann::save_bin<_u64>(index_file, new_meta.data(), new_meta.size(), 1, 0);
  }

  double get_memory_budget(double search_ram_budget) {
    double final_index_ram_limit = search_ram_budget;
    if (search_ram_budget - SPACE_FOR_CACHED_NODES_IN_GB > THRESHOLD_FOR_CACHING_IN_GB) {  // slack for space used by
                                                                                           // cached nodes
      final_index_ram_limit = search_ram_budget - SPACE_FOR_CACHED_NODES_IN_GB;
    }
    return final_index_ram_limit * 1024 * 1024 * 1024;
  }

  double get_memory_budget(const std::string &mem_budget_str) {
    double search_ram_budget = atof(mem_budget_str.c_str());
    return get_memory_budget(search_ram_budget);
  }

  size_t calculate_num_pq_chunks(double final_index_ram_limit, size_t points_num, uint32_t dim) {
    size_t num_pq_chunks = (size_t) (std::floor)(_u64(final_index_ram_limit / (double) points_num));

    LOG(INFO) << "Calculated num_pq_chunks :" << num_pq_chunks;
    num_pq_chunks = num_pq_chunks <= 0 ? 1 : num_pq_chunks;
    num_pq_chunks = num_pq_chunks > dim ? dim : num_pq_chunks;
    num_pq_chunks = num_pq_chunks > MAX_PQ_CHUNKS ? MAX_PQ_CHUNKS : num_pq_chunks;

    LOG(INFO) << "Compressing " << dim << "-dimensional data into " << num_pq_chunks << " bytes per vector.";
    return num_pq_chunks;
  }

  double calculate_recall(unsigned num_queries, unsigned *gold_std, float *gs_dist, unsigned dim_gs,
                          unsigned *our_results, unsigned dim_or, unsigned recall_at) {
    double total_recall = 0;
    std::set<unsigned> gt, res;

    for (size_t i = 0; i < num_queries; i++) {
      gt.clear();
      res.clear();
      unsigned *gt_vec = gold_std + dim_gs * i;
      unsigned *res_vec = our_results + dim_or * i;
      size_t tie_breaker = recall_at;
      if (gs_dist != nullptr) {
        float *gt_dist_vec = gs_dist + dim_gs * i;
        tie_breaker = recall_at - 1;
        while (tie_breaker < dim_gs && gt_dist_vec[tie_breaker] == gt_dist_vec[recall_at - 1])
          tie_breaker++;
      }

      gt.insert(gt_vec, gt_vec + tie_breaker);
      res.insert(res_vec, res_vec + recall_at);

      unsigned cur_recall = 0;
      for (auto &v : res) {
        if (gt.find(v) != gt.end()) {
          cur_recall++;
        }
      }
      total_recall += cur_recall;
    }
    return total_recall / (num_queries) * (100.0 / recall_at);
  }

  double calculate_recall(unsigned num_queries, unsigned *gold_std, float *gs_dist, unsigned dim_gs,
                          unsigned *our_results, unsigned dim_or, unsigned recall_at,
                          const tsl::robin_set<unsigned> &active_tags) {
    double total_recall = 0;
    std::set<unsigned> gt, res;
    bool printed = false;
    for (size_t i = 0; i < num_queries; i++) {
      gt.clear();
      res.clear();
      unsigned *gt_vec = gold_std + dim_gs * i;
      unsigned *res_vec = our_results + dim_or * i;
      size_t tie_breaker = recall_at;
      unsigned active_points_count = 0;
      unsigned cur_counter = 0;
      while (active_points_count < recall_at && cur_counter < dim_gs) {
        if (active_tags.find(*(gt_vec + cur_counter)) != active_tags.end()) {
          active_points_count++;
        }
        cur_counter++;
      }
      if (active_tags.empty())
        cur_counter = recall_at;

      if ((active_points_count < recall_at && !active_tags.empty()) && !printed) {
        LOG(INFO) << "Warning: Couldn't find enough closest neighbors " << active_points_count << "/" << recall_at
                  << " from truthset for query # " << i << ". Will result in under-reported value of recall.";
        printed = true;
      }
      if (gs_dist != nullptr) {
        tie_breaker = cur_counter - 1;
        float *gt_dist_vec = gs_dist + dim_gs * i;
        while (tie_breaker < dim_gs && gt_dist_vec[tie_breaker] == gt_dist_vec[cur_counter - 1])
          tie_breaker++;
      }

      gt.insert(gt_vec, gt_vec + tie_breaker);
      res.insert(res_vec, res_vec + recall_at);
      unsigned cur_recall = 0;
      for (auto &v : res) {
        if (gt.find(v) != gt.end()) {
          cur_recall++;
        }
      }
      total_recall += cur_recall;
    }
    return ((double) (total_recall / (num_queries))) * ((double) (100.0 / recall_at));
  }

  /***************************************************
      Support for Merging Many Vamana Indices
   ***************************************************/

  void read_idmap(const std::string &fname, std::vector<unsigned> &ivecs) {
    uint32_t npts32, dim;
    size_t actual_file_size = get_file_size(fname);
    std::ifstream reader(fname.c_str(), std::ios::binary);
    reader.read((char *) &npts32, sizeof(uint32_t));
    reader.read((char *) &dim, sizeof(uint32_t));
    if (dim != 1 || actual_file_size != ((size_t) npts32) * sizeof(uint32_t) + 2 * sizeof(uint32_t)) {
      LOG(ERROR) << "Error reading idmap file. Check if the file is bin file with 1 dimensional data. Actual: "
                 << actual_file_size << ", expected: " << (size_t) npts32 + 2 * sizeof(uint32_t);

      crash();
    }
    ivecs.resize(npts32);
    reader.read((char *) ivecs.data(), ((size_t) npts32) * sizeof(uint32_t));
    reader.close();
  }

  int merge_shards(const std::string &vamana_prefix, const std::string &vamana_suffix, const std::string &idmaps_prefix,
                   const std::string &idmaps_suffix, const _u64 nshards, unsigned max_degree,
                   const std::string &output_vamana, const std::string &medoids_file) {
    // Read ID maps
    std::vector<std::string> vamana_names(nshards);
    std::vector<std::vector<unsigned>> idmaps(nshards);
    for (_u64 shard = 0; shard < nshards; shard++) {
      vamana_names[shard] = vamana_prefix + std::to_string(shard) + vamana_suffix;
      read_idmap(idmaps_prefix + std::to_string(shard) + idmaps_suffix, idmaps[shard]);
    }

    // find max node id
    _u64 nnodes = 0;
    _u64 nelems = 0;
    for (auto &idmap : idmaps) {
      for (auto &id : idmap) {
        nnodes = std::max(nnodes, (_u64) id);
      }
      nelems += idmap.size();
    }
    nnodes++;
    LOG(INFO) << "# nodes: " << nnodes << ", max. degree: " << max_degree;

    // compute inverse map: node -> shards
    std::vector<std::pair<unsigned, unsigned>> node_shard;
    node_shard.reserve(nelems);
    for (_u64 shard = 0; shard < nshards; shard++) {
      LOG(INFO) << "Creating inverse map -- shard #" << shard;
      for (_u64 idx = 0; idx < idmaps[shard].size(); idx++) {
        _u64 node_id = idmaps[shard][idx];
        node_shard.push_back(std::make_pair((_u32) node_id, (_u32) shard));
      }
    }
    std::sort(node_shard.begin(), node_shard.end(), [](const auto &left, const auto &right) {
      return left.first < right.first || (left.first == right.first && left.second < right.second);
    });
    LOG(INFO) << "Finished computing node -> shards map";

    // create cached vamana readers
    std::vector<cached_ifstream> vamana_readers(nshards);
    for (_u64 i = 0; i < nshards; i++) {
      vamana_readers[i].open(vamana_names[i], 1024 * 1048576);
      size_t expected_file_size;
      vamana_readers[i].read((char *) &expected_file_size, sizeof(uint64_t));
    }

    size_t merged_index_size = 24;
    size_t merged_index_frozen = 0;
    // create cached vamana writers
    cached_ofstream diskann_writer(output_vamana, 1024 * 1048576);
    diskann_writer.write((char *) &merged_index_size, sizeof(uint64_t));

    unsigned output_width = max_degree;
    unsigned max_input_width = 0;
    // read width from each vamana to advance buffer by sizeof(unsigned) bytes
    for (auto &reader : vamana_readers) {
      unsigned input_width;
      reader.read((char *) &input_width, sizeof(unsigned));
      max_input_width = input_width > max_input_width ? input_width : max_input_width;
    }

    LOG(INFO) << "Max input width: " << max_input_width << ", output width: " << output_width;

    diskann_writer.write((char *) &output_width, sizeof(unsigned));
    std::ofstream medoid_writer(medoids_file.c_str(), std::ios::binary);
    _u32 nshards_u32 = (_u32) nshards;
    _u32 one_val = 1;
    medoid_writer.write((char *) &nshards_u32, sizeof(uint32_t));
    medoid_writer.write((char *) &one_val, sizeof(uint32_t));

    _u64 vamana_index_frozen = 0;
    for (_u64 shard = 0; shard < nshards; shard++) {
      unsigned medoid;
      // read medoid
      vamana_readers[shard].read((char *) &medoid, sizeof(unsigned));
      vamana_readers[shard].read((char *) &vamana_index_frozen, sizeof(_u64));
      assert(vamana_index_frozen == false);
      // rename medoid
      medoid = idmaps[shard][medoid];

      medoid_writer.write((char *) &medoid, sizeof(uint32_t));
      // write renamed medoid
      if (shard == (nshards - 1))  //--> uncomment if running hierarchical
        diskann_writer.write((char *) &medoid, sizeof(unsigned));
    }
    diskann_writer.write((char *) &merged_index_frozen, sizeof(_u64));
    medoid_writer.close();

    LOG(INFO) << "Starting merge";

    // random_shuffle() is deprecated.
    std::random_device rng;
    std::mt19937 urng(rng());

    std::vector<bool> nhood_set(nnodes, 0);
    std::vector<unsigned> final_nhood;

    unsigned nnbrs = 0, shard_nnbrs = 0;
    unsigned cur_id = 0;
    for (const auto &id_shard : node_shard) {
      unsigned node_id = id_shard.first;
      unsigned shard_id = id_shard.second;
      if (cur_id < node_id) {
        // random_shuffle() is deprecated.
        std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
        nnbrs = (unsigned) (std::min)(final_nhood.size(), (uint64_t) max_degree);
        // write into merged ofstream
        diskann_writer.write((char *) &nnbrs, sizeof(unsigned));
        diskann_writer.write((char *) final_nhood.data(), nnbrs * sizeof(unsigned));
        merged_index_size += (sizeof(unsigned) + nnbrs * sizeof(unsigned));
        if (cur_id % 499999 == 1) {
          LOG(INFO) << cur_id << "...";
        }
        cur_id = node_id;
        nnbrs = 0;
        for (auto &p : final_nhood)
          nhood_set[p] = 0;
        final_nhood.clear();
      }
      // read from shard_id ifstream
      vamana_readers[shard_id].read((char *) &shard_nnbrs, sizeof(unsigned));
      std::vector<unsigned> shard_nhood(shard_nnbrs);
      vamana_readers[shard_id].read((char *) shard_nhood.data(), shard_nnbrs * sizeof(unsigned));

      // rename nodes
      for (_u64 j = 0; j < shard_nnbrs; j++) {
        if (nhood_set[idmaps[shard_id][shard_nhood[j]]] == 0) {
          nhood_set[idmaps[shard_id][shard_nhood[j]]] = 1;
          final_nhood.emplace_back(idmaps[shard_id][shard_nhood[j]]);
        }
      }
    }

    // random_shuffle() is deprecated.
    std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
    nnbrs = (unsigned) (std::min)(final_nhood.size(), (uint64_t) max_degree);
    // write into merged ofstream
    diskann_writer.write((char *) &nnbrs, sizeof(unsigned));
    diskann_writer.write((char *) final_nhood.data(), nnbrs * sizeof(unsigned));
    merged_index_size += (sizeof(unsigned) + nnbrs * sizeof(unsigned));
    for (auto &p : final_nhood)
      nhood_set[p] = 0;
    final_nhood.clear();

    LOG(INFO) << "Expected size: " << merged_index_size;

    diskann_writer.reset();
    diskann_writer.write((char *) &merged_index_size, sizeof(uint64_t));

    LOG(INFO) << "Finished merge";
    return 0;
  }

  template<typename T>
  int merge_shards_h(const std::string &vamana_prefix, const std::string &vamana_suffix, const std::string &idmaps_prefix,
                   const std::string &idmaps_suffix, const _u64 nshards, unsigned max_degree,
                   const std::string &output_vamana, const std::string &medoids_file,
                   pipeann::Metric m, const Parameters &parameter) {
    // Read ID maps
    std::vector<std::string> vamana_names(nshards);
    std::vector<std::vector<unsigned>> idmaps(nshards);
    for (_u64 shard = 0; shard < nshards; shard++) {
      vamana_names[shard] = vamana_prefix + std::to_string(shard) + vamana_suffix;
      read_idmap(idmaps_prefix + std::to_string(shard) + idmaps_suffix, idmaps[shard]);
    }

    // find max node id
    _u64 nnodes = 0;
    _u64 nelems = 0;
    for (auto &idmap : idmaps) {
      for (auto &id : idmap) {
        nnodes = std::max(nnodes, (_u64) id);
      }
      nelems += idmap.size();
    }
    nnodes++;
    LOG(INFO) << "# nodes: " << nnodes << ", max. degree: " << max_degree;

    // compute inverse map: node -> shards
    std::vector<std::pair<unsigned, unsigned>> node_shard;
    node_shard.reserve(nelems);
    for (_u64 shard = 0; shard < nshards; shard++) {
      LOG(INFO) << "Creating inverse map -- shard #" << shard;
      for (_u64 idx = 0; idx < idmaps[shard].size(); idx++) {
        _u64 node_id = idmaps[shard][idx];
        node_shard.push_back(std::make_pair((_u32) node_id, (_u32) shard));
      }
    }
    std::sort(node_shard.begin(), node_shard.end(), [](const auto &left, const auto &right) {
      return left.first < right.first || (left.first == right.first && left.second < right.second);
    });
    LOG(INFO) << "Finished computing node -> shards map";

    // create cached vamana readers
    std::vector<cached_ifstream> vamana_readers(nshards);
    for (_u64 i = 0; i < nshards; i++) {
      vamana_readers[i].open(vamana_names[i], 1024 * 1048576);
      size_t expected_file_size;
      vamana_readers[i].read((char *) &expected_file_size, sizeof(uint64_t));
      LOG(INFO) << expected_file_size;
    }

    size_t merged_index_size = 24;
    size_t merged_index_frozen = 0;
    // create cached vamana writers
    cached_ofstream diskann_writer(output_vamana, 1024 * 1048576);
    diskann_writer.write((char *) &merged_index_size, sizeof(uint64_t));

    unsigned output_width = max_degree;
    unsigned max_input_width = 0;
    // read width from each vamana to advance buffer by sizeof(unsigned) bytes
    for (auto &reader : vamana_readers) {
      unsigned input_width;
      reader.read((char *) &input_width, sizeof(unsigned));
      max_input_width = input_width > max_input_width ? input_width : max_input_width;
    }

    LOG(INFO) << "Max input width: " << max_input_width << ", output width: " << output_width;

    // diskann_writer.write((char *) &output_width, sizeof(unsigned));
    std::ofstream medoid_writer(medoids_file.c_str(), std::ios::binary);
    _u32 nshards_u32 = (_u32) nshards;
    _u32 one_val = 1;
    medoid_writer.write((char *) &nshards_u32, sizeof(uint32_t));
    medoid_writer.write((char *) &one_val, sizeof(uint32_t));

    _u64 vamana_index_frozen = 0;
    for (_u64 shard = 0; shard < nshards; shard++) {
      unsigned medoid;
      // read medoid
      vamana_readers[shard].read((char *) &medoid, sizeof(unsigned));
      vamana_readers[shard].read((char *) &vamana_index_frozen, sizeof(_u64));
      assert(vamana_index_frozen == false);
      // rename medoid
      medoid = idmaps[shard][medoid];
      LOG(INFO) << medoid ;

      // medoid_writer.write((char *) &medoid, sizeof(uint32_t));
      // write renamed medoid
      // if (shard == (nshards - 1))  //--> uncomment if running hierarchical
      //   diskann_writer.write((char *) &medoid, sizeof(unsigned));
    }
    // diskann_writer.write((char *) &merged_index_frozen, sizeof(_u64));
    // medoid_writer.close();

    LOG(INFO) << "Starting merge";

    // random_shuffle() is deprecated.
    std::random_device rng;
    std::mt19937 urng(rng());

    std::vector<bool> nhood_set(nnodes, 0);
    std::vector<NeighborH> final_nhood;

    unsigned nnbrs = 0, shard_nnbrs = 0;
    unsigned cur_id = 0;
    size_t max = 0, min = 1 << 30, total = 0, cnt = 0;
    for (const auto &id_shard : node_shard) {
      unsigned node_id = id_shard.first;
      unsigned shard_id = id_shard.second;
      if (cur_id < node_id) {
        // random_shuffle() is deprecated.
        // std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
        std::vector<DEGNeighbor> prune_list;
        // prune_neighbors<T>(final_nhood, max_degree, m, prune_list, parameter);
        prune_neighbors_no_opt<T>(final_nhood, max_degree, m, prune_list, parameter);
        // write into merged ofstream
        nnbrs = prune_list.size();
        diskann_writer.write((char *) &nnbrs, sizeof(unsigned));
        max = std::max(max, prune_list.size());
        min = (std::min)(min, prune_list.size());
        total += prune_list.size();
        if (prune_list.size() < 2)
          cnt++;
        for (unsigned k = 0; k < nnbrs; k++) {
          DEGNeighbor &neighbor = prune_list[k];
          unsigned neighbor_id = neighbor.id_;
          diskann_writer.write((char *)&neighbor_id, sizeof(unsigned));
          float dist1 = neighbor.emb_distance_;
          float dist2 = neighbor.geo_distance_;
          diskann_writer.write((char *)&dist1, sizeof(float));
          diskann_writer.write((char *)&dist2, sizeof(float));
          // std::vector<std::pair<float, float>> &use_range = neighbor.available_range;
          // unsigned range_size = use_range.size();
          // diskann_writer.write((char *)&range_size, sizeof(unsigned));
          // for (unsigned t = 0; t < range_size; t++) {
          //   int8_t x = static_cast<int8_t>(use_range[t].first * 100);
          //   int8_t y = static_cast<int8_t>(use_range[t].second * 100);
          //   diskann_writer.write((char *) &x, sizeof(int8_t));
          //   diskann_writer.write((char *) &y, sizeof(int8_t));
          // }
        }
        // diskann_writer.write((char *) final_nhood.data(), nnbrs * sizeof(unsigned));
        merged_index_size += (sizeof(unsigned) + nnbrs * sizeof(unsigned));
        if (cur_id % 499999 == 1) {
          LOG(INFO) << cur_id << "...";
        }
        cur_id = node_id;
        nnbrs = 0;
        for (auto &p : final_nhood)
          nhood_set[p.id] = 0;
        final_nhood.clear();
      }
      // read from shard_id ifstream
      vamana_readers[shard_id].read((char *) &shard_nnbrs, sizeof(unsigned));
      std::vector<NeighborH> shard_nhood;
      shard_nhood.reserve(shard_nnbrs);
      for (unsigned k = 0; k < shard_nnbrs; k++)
      {
        unsigned neighbor_id;
        vamana_readers[shard_id].read((char *)&neighbor_id, sizeof(unsigned));
        float dist1 = 0, dist2 = 0;
        vamana_readers[shard_id].read((char *)&dist1, sizeof(float));
        vamana_readers[shard_id].read((char *)&dist2, sizeof(float));
        // unsigned range_size;
        // vamana_readers[shard_id].read((char *)&range_size, sizeof(unsigned));
        // for (unsigned t = 0; t < range_size; t++)
        // {
        //   int8_t range_start, range_end;
        //   vamana_readers[shard_id].read((char *)&range_start, sizeof(int8_t));
        //   vamana_readers[shard_id].read((char *)&range_end, sizeof(int8_t));
        // }
        shard_nhood.push_back(NeighborH(neighbor_id, dist1, dist2, false, 0));
      }

      // rename nodes
      for (_u64 j = 0; j < shard_nnbrs; j++) {
        if (nhood_set[idmaps[shard_id][shard_nhood[j].id]] == 0) {
          nhood_set[idmaps[shard_id][shard_nhood[j].id]] = 1;
          final_nhood.emplace_back(idmaps[shard_id][shard_nhood[j].id], shard_nhood[j].distance1, shard_nhood[j].distance2, false, 0);
        }
      }
    }

    // random_shuffle() is deprecated.
    // std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
    nnbrs = (unsigned) (std::min)(final_nhood.size(), (uint64_t) max_degree);
    std::vector<DEGNeighbor> prune_list;
    // prune_neighbors<T>(final_nhood, max_degree, m, prune_list, parameter);
    prune_neighbors_no_opt<T>(final_nhood, max_degree, m, prune_list, parameter);
    nnbrs = prune_list.size();
    max = std::max(max, prune_list.size());
    min = (std::min)(min, prune_list.size());
    total += prune_list.size();
    if (prune_list.size() < 2)
      cnt++;
    // write into merged ofstream
    diskann_writer.write((char *) &nnbrs, sizeof(unsigned));
    for (unsigned k = 0; k < nnbrs; k++) {
      DEGNeighbor &neighbor = prune_list[k];
      unsigned neighbor_id = neighbor.id_;
      diskann_writer.write((char *)&neighbor_id, sizeof(unsigned));
      float dist1 = neighbor.emb_distance_;
      float dist2 = neighbor.geo_distance_;
      diskann_writer.write((char *)&dist1, sizeof(float));
      diskann_writer.write((char *)&dist2, sizeof(float));
      // std::vector<std::pair<float, float>> &use_range = neighbor.available_range;
      // unsigned range_size = use_range.size();
      // diskann_writer.write((char *)&range_size, sizeof(unsigned));
      // for (unsigned t = 0; t < range_size; t++) {
      //   int8_t x = static_cast<int8_t>(use_range[t].first * 100);
      //   int8_t y = static_cast<int8_t>(use_range[t].second * 100);
      //   diskann_writer.write((char *) &x, sizeof(int8_t));
      //   diskann_writer.write((char *) &y, sizeof(int8_t));
      // }
    }
    // diskann_writer.write((char *) final_nhood.data(), nnbrs * sizeof(unsigned));
    merged_index_size += (sizeof(unsigned) + nnbrs * sizeof(unsigned));
    for (auto &p : final_nhood)
      nhood_set[p.id] = 0;
    final_nhood.clear();

    LOG(INFO) << "Expected size: " << merged_index_size;

    diskann_writer.reset();
    diskann_writer.write((char *) &nnodes, sizeof(uint64_t));
    diskann_writer.close();
    LOG(INFO) << "Index built with degree: max:" << max << "  avg:" << (float) total / (float) (nnodes)
              << "  min:" << min << "  count(deg<2):" << cnt;

    LOG(INFO) << "Finished merge";
    return 0;
  }
  template int merge_shards_h<float>(const std::string &vamana_prefix, const std::string &vamana_suffix, const std::string &idmaps_prefix,
                   const std::string &idmaps_suffix, const _u64 nshards, unsigned max_degree,
                   const std::string &output_vamana, const std::string &medoids_file,
                   pipeann::Metric m, const Parameters &parameter);
  template int merge_shards_h<int8_t>(const std::string &vamana_prefix, const std::string &vamana_suffix, const std::string &idmaps_prefix,
                   const std::string &idmaps_suffix, const _u64 nshards, unsigned max_degree,
                   const std::string &output_vamana, const std::string &medoids_file,
                   pipeann::Metric m, const Parameters &parameter);
  template int merge_shards_h<uint8_t>(const std::string &vamana_prefix, const std::string &vamana_suffix, const std::string &idmaps_prefix,
                   const std::string &idmaps_suffix, const _u64 nshards, unsigned max_degree,
                   const std::string &output_vamana, const std::string &medoids_file,
                   pipeann::Metric m, const Parameters &parameter);

  template<typename T>
  int build_merged_vamana_index(std::string base_file, pipeann::Metric _compareMetric, bool single_file_index,
                                unsigned L, unsigned R, double sampling_rate, double ram_budget,
                                std::string mem_index_path, std::string medoids_file, std::string centroids_file,
                                const char *tag_file) {
    if (unlikely(single_file_index)) {
      LOG(INFO) << "Single file index is not supported for merged Vamana index, setting to false.";
      single_file_index = false;
    }

    size_t base_num, base_dim;
    pipeann::get_bin_metadata(base_file, base_num, base_dim);

    double full_index_ram = estimate_ram_usage(base_num, base_dim, sizeof(T), R);
    if (full_index_ram < ram_budget * 1024 * 1024 * 1024) {
      LOG(INFO) << "Full index fits in RAM, building in one shot";
      pipeann::Parameters paras;
      paras.Set<unsigned>("L", (unsigned) L);
      paras.Set<unsigned>("R", (unsigned) R);
      paras.Set<unsigned>("C", 750);
      paras.Set<float>("alpha", 1.2f);
      paras.Set<bool>("saturate_graph", 1);  // was 0 earlier.
      paras.Set<std::string>("save_path", mem_index_path);
      size_t pos = mem_index_path.find_last_of('/');
      std::string index_prefix_path = (pos != std::string::npos) ? mem_index_path.substr(0, pos) : mem_index_path;
      LOG(ERROR) << "index_prefix_path: " << index_prefix_path;
      paras.Set<std::string>("index_prefix_path", index_prefix_path);

      bool tags_enabled;
      if (tag_file == nullptr)
        tags_enabled = false;
      else
        tags_enabled = true;

      std::unique_ptr<pipeann::Index<T>> _pvamanaIndex = std::unique_ptr<pipeann::Index<T>>(
          new pipeann::Index<T>(_compareMetric, base_dim, base_num, false, single_file_index, tags_enabled));
      if (tags_enabled)
        _pvamanaIndex->build(base_file.c_str(), base_num, paras, tag_file);
      else
        _pvamanaIndex->build(base_file.c_str(), base_num, paras);

      _pvamanaIndex->save(mem_index_path.c_str());
      std::remove(medoids_file.c_str());
      std::remove(centroids_file.c_str());
      return 0;
    }

    if (single_file_index || tag_file != nullptr) {
      LOG(INFO) << "Cannot build merged index if single_file_index is "
                   "required or if tags are specified.";
      return 1;
    }

    std::string merged_index_prefix = mem_index_path + "_tempFiles";
    int num_parts =
        partition_with_ram_budget<T>(base_file, sampling_rate, ram_budget, 2 * R / 3, merged_index_prefix, 2);

    std::string cur_centroid_filepath = merged_index_prefix + "_centroids.bin";
    std::rename(cur_centroid_filepath.c_str(), centroids_file.c_str());

    for (int p = 0; p < num_parts; p++) {
      std::string shard_base_file = merged_index_prefix + "_subshard-" + std::to_string(p) + ".bin";
      std::string shard_index_file = merged_index_prefix + "_subshard-" + std::to_string(p) + "_mem.index";

      pipeann::Parameters paras;
      paras.Set<unsigned>("L", L);
      paras.Set<unsigned>("R", (2 * (R / 3)));
      paras.Set<unsigned>("C", 750);
      paras.Set<float>("alpha", 1.2f);
      paras.Set<bool>("saturate_graph", 0);
      paras.Set<std::string>("save_path", shard_index_file);

      _u64 shard_base_dim, shard_base_pts;
      get_bin_metadata(shard_base_file, shard_base_pts, shard_base_dim);
      std::unique_ptr<pipeann::Index<T>> _pvamanaIndex = std::unique_ptr<pipeann::Index<T>>(
          new pipeann::Index<T>(_compareMetric, shard_base_dim, shard_base_pts, false,
                                single_file_index));  // TODO: Single?
      _pvamanaIndex->build(shard_base_file.c_str(), shard_base_pts, paras);
      _pvamanaIndex->save(shard_index_file.c_str());
    }

    pipeann::merge_shards(merged_index_prefix + "_subshard-", "_mem.index", merged_index_prefix + "_subshard-",
                          "_ids_uint32.bin", num_parts, R, mem_index_path, medoids_file);

    // delete tempFiles
    for (int p = 0; p < num_parts; p++) {
      std::string shard_base_file = merged_index_prefix + "_subshard-" + std::to_string(p) + ".bin";
      std::string shard_id_file = merged_index_prefix + "_subshard-" + std::to_string(p) + "_ids_uint32.bin";
      std::string shard_index_file = merged_index_prefix + "_subshard-" + std::to_string(p) + "_mem.index";
      // Required if Index.cpp thinks we are building a multi-file index.
      std::string shard_index_file_data = shard_index_file + ".data";

      std::remove(shard_base_file.c_str());
      std::remove(shard_id_file.c_str());
      std::remove(shard_index_file.c_str());
      std::remove(shard_index_file_data.c_str());
    }
    return 0;
  }

  template<typename T>
  int build_merged_vamana_index(std::string base1_file, std::string base2_file, pipeann::Metric _compareMetric,
                                bool single_file_index, unsigned L, unsigned R, double sampling_rate, double ram_budget,
                                std::string mem_index_path1, std::string medoids_file1, std::string centroids_file1,
                                std::string mem_index_path2, std::string medoids_file2, std::string centroids_file2,
                                std::string mem_index_path, std::string medoids_file,
                                const char *tag_file) {
    if (unlikely(single_file_index)) {
      LOG(INFO) << "Single file index is not supported for merged Vamana index, setting to false.";
      single_file_index = false;
    }

    size_t base_num1, base_dim1;
    size_t base_num2, base_dim2;
    pipeann::get_bin_metadata(base1_file, base_num1, base_dim1);
    pipeann::get_bin_metadata(base2_file, base_num2, base_dim2);

    double full_index_ram = estimate_ram_usage(base_num1, base_dim1, base_dim2, sizeof(T), R);
    LOG(INFO) << full_index_ram/1024/1024/1024 << " " << ram_budget;
    if (full_index_ram < ram_budget * 1024 * 1024 * 1024) {
      LOG(INFO) << "Full index fits in RAM, building in one shot";
      pipeann::Parameters paras;
      paras.Set<unsigned>("L", (unsigned) L);
      paras.Set<unsigned>("R", (unsigned) R);
      paras.Set<unsigned>("C", 750);
      paras.Set<float>("alpha", 1.2f);
      paras.Set<bool>("saturate_graph", 1);  // was 0 earlier.
      paras.Set<std::string>("save_path", mem_index_path);
      size_t pos = mem_index_path1.find_last_of('/');
      std::string index_prefix_path = (pos != std::string::npos) ? mem_index_path1.substr(0, pos) : mem_index_path1;
      LOG(ERROR) << "index_prefix_path: " << index_prefix_path;
      paras.Set<std::string>("index_prefix_path", index_prefix_path);

      bool tags_enabled;
      if (tag_file == nullptr)
        tags_enabled = false;
      else
        tags_enabled = true;

      std::unique_ptr<pipeann::HybridIndex<T>> _pvamanaIndex =
          std::unique_ptr<pipeann::HybridIndex<T>>(new pipeann::HybridIndex<T>(
              _compareMetric, base_dim1, base_dim2, base_num1, false, single_file_index, tags_enabled));
      _pvamanaIndex->build(base1_file.c_str(), base2_file.c_str(), base_num1, paras);

      _pvamanaIndex->save_single(mem_index_path.c_str());
      std::ofstream medoid_writer(medoids_file.c_str(), std::ios::binary);
      _u32 nshards_u32 = (_u32) _pvamanaIndex->_init_ids.size();
      _u32 one_val = 1;
      medoid_writer.write((char *) &nshards_u32, sizeof(uint32_t));
      medoid_writer.write((char *) &one_val, sizeof(uint32_t));
      for (auto medoid : _pvamanaIndex->_init_ids)
        medoid_writer.write((char *) &medoid, sizeof(uint32_t));
      medoid_writer.close();
      return 0;
    }

    if (single_file_index || tag_file != nullptr) {
      LOG(INFO) << "Cannot build merged index if single_file_index is "
                   "required or if tags are specified.";
      return 1;
    }

    std::string merged_index_prefix1 = mem_index_path1 + "_tempFiles";
    std::string merged_index_prefix2 = mem_index_path2 + "_tempFiles";
    int num_parts1 = partition_two_vectors_with_ram_budget<T>(base1_file, base2_file, base_dim1, base_dim2, sampling_rate, ram_budget, (2*(R/3)),
                                                  merged_index_prefix1, 1);

    std::string cur_centroid_filepath1 = merged_index_prefix1 + "_centroids.bin";
    std::string cur_centroid_filepath2 = merged_index_prefix2 + "_centroids.bin";
    std::rename(cur_centroid_filepath1.c_str(), centroids_file1.c_str());
    std::rename(cur_centroid_filepath2.c_str(), centroids_file2.c_str());

    for (int p = 0; p < num_parts1; p++) {
      std::string shard_base_file1 = merged_index_prefix1 + "_subshard-" + std::to_string(p) + ".bin";
      std::string shard_index_file = merged_index_prefix1 + "_subshard-" + std::to_string(p) + "_mem.index";
      std::string shard_base_file2 = merged_index_prefix1 + "_subshard_loc-" + std::to_string(p) + ".bin";

      pipeann::Parameters paras;
      paras.Set<unsigned>("L", L);
      paras.Set<unsigned>("R", (2 * (R / 3)));
      paras.Set<unsigned>("C", 750);
      paras.Set<float>("alpha", 1.2f);
      paras.Set<bool>("saturate_graph", 0);
      paras.Set<std::string>("save_path", shard_index_file);

      _u64 shard_base_dim1, shard_base_pts1;
      get_bin_metadata(shard_base_file1, shard_base_pts1, shard_base_dim1);
      _u64 shard_base_dim2, shard_base_pts2;
      get_bin_metadata(shard_base_file2, shard_base_pts2, shard_base_dim2);
      std::unique_ptr<pipeann::HybridIndex<T>> _pvamanaIndex = std::unique_ptr<pipeann::HybridIndex<T>>(
          new pipeann::HybridIndex<T>(_compareMetric, shard_base_dim1, shard_base_dim2, shard_base_pts1, false,
                                      single_file_index));  // TODO: Single?
      _pvamanaIndex->build(shard_base_file1.c_str(), shard_base_file2.c_str(), shard_base_pts1, paras);
      _pvamanaIndex->save(shard_index_file.c_str());
    }

    pipeann::Parameters paras;
    paras.Set<unsigned>("dim1", base_dim1);
    paras.Set<unsigned>("dim2", base_dim2);
    paras.Set<std::string>("data1_file", base1_file);
    paras.Set<std::string>("data2_file", base2_file);
    std::unique_ptr<boost::interprocess::file_mapping> m1_file_ptr;
    std::unique_ptr<boost::interprocess::mapped_region> m1_region_ptr;
    std::unique_ptr<boost::interprocess::file_mapping> m2_file_ptr;
    std::unique_ptr<boost::interprocess::mapped_region> m2_region_ptr;
    m1_file_ptr = std::make_unique<boost::interprocess::file_mapping>(base1_file.c_str(), boost::interprocess::read_only);
    m1_region_ptr = std::make_unique<boost::interprocess::mapped_region>(*m1_file_ptr, boost::interprocess::read_only);
    m2_file_ptr = std::make_unique<boost::interprocess::file_mapping>(base2_file.c_str(), boost::interprocess::read_only);
    m2_region_ptr = std::make_unique<boost::interprocess::mapped_region>(*m2_file_ptr, boost::interprocess::read_only);
    T *_data1 = (T *)((char*)m1_region_ptr->get_address() + sizeof(int) * 2);
    T *_data2 = (T *)((char*)m2_region_ptr->get_address() + sizeof(int) * 2);
    paras.Set("data1", _data1);
    paras.Set("data2", _data2);

    auto start = std::chrono::high_resolution_clock::now();
    pipeann::merge_shards_h<T>(merged_index_prefix1 + "_subshard-", "_mem.index", merged_index_prefix1 + "_subshard-",
                          "_ids_uint32.bin", num_parts1, R, mem_index_path1, medoids_file1, _compareMetric, paras);
    auto end = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "merge time: " << std::chrono::duration<double>(end - start).count();

    // delete tempFiles
    for (int p = 0; p < num_parts1; p++) {
      std::string shard_base_file = merged_index_prefix1 + "_subshard-" + std::to_string(p) + ".bin";
      std::string shard_base_file2 = merged_index_prefix1 + "_subshard_loc-" + std::to_string(p) + ".bin";
      std::string shard_id_file = merged_index_prefix1 + "_subshard-" + std::to_string(p) + "_ids_uint32.bin";
      std::string shard_index_file = merged_index_prefix1 + "_subshard-" + std::to_string(p) + "_mem.index";
      // Required if Index.cpp thinks we are building a multi-file index.
      std::string shard_index_file_data = shard_index_file + ".data";

      std::remove(shard_base_file.c_str());
      std::remove(shard_base_file2.c_str());
      std::remove(shard_id_file.c_str());
      std::remove(shard_index_file.c_str());
      std::remove(shard_index_file_data.c_str());
    }

    int num_parts2 = partition_two_vectors_with_ram_budget<T>(base2_file, base1_file, base_dim2, base_dim1, sampling_rate, ram_budget, (2*(R/3)),
                                                  merged_index_prefix2, 1);

    for (int p = 0; p < num_parts2; p++) {
      std::string shard_base_file2 = merged_index_prefix2 + "_subshard-" + std::to_string(p) + ".bin";
      std::string shard_base_file1 = merged_index_prefix2 + "_subshard_loc-" + std::to_string(p) + ".bin";
      std::string shard_index_file = merged_index_prefix2 + "_subshard-" + std::to_string(p) + "_mem.index";

      pipeann::Parameters paras;
      paras.Set<unsigned>("L", L);
      paras.Set<unsigned>("R", (2 * (R / 3)));
      paras.Set<unsigned>("C", 750);
      paras.Set<float>("alpha", 1.2f);
      paras.Set<bool>("saturate_graph", 0);
      paras.Set<std::string>("save_path", shard_index_file);

      _u64 shard_base_dim1, shard_base_pts1;
      get_bin_metadata(shard_base_file1, shard_base_pts1, shard_base_dim1);
      _u64 shard_base_dim2, shard_base_pts2;
      get_bin_metadata(shard_base_file2, shard_base_pts2, shard_base_dim2);
      std::unique_ptr<pipeann::HybridIndex<T>> _pvamanaIndex = std::unique_ptr<pipeann::HybridIndex<T>>(
          new pipeann::HybridIndex<T>(_compareMetric, shard_base_dim1, shard_base_dim2, shard_base_pts1, false,
                                      single_file_index));  // TODO: Single?
      _pvamanaIndex->build(shard_base_file1.c_str(), shard_base_file2.c_str(), shard_base_pts1, paras);
      _pvamanaIndex->save(shard_index_file.c_str());
    }


    start = std::chrono::high_resolution_clock::now();
    pipeann::merge_shards_h<T>(merged_index_prefix2 + "_subshard-", "_mem.index", merged_index_prefix2 + "_subshard-",
                          "_ids_uint32.bin", num_parts2, R, mem_index_path2, medoids_file2, _compareMetric, paras);
    end = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "merge time: " << std::chrono::duration<double>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    pipeann::merge_two_graphs<T>(mem_index_path1, mem_index_path2, medoids_file1, medoids_file2,
                          R, L, mem_index_path, medoids_file, _compareMetric, paras);
    end = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "merge time: " << std::chrono::duration<double>(end - start).count();

    // delete tempFiles
    for (int p = 0; p < num_parts2; p++) {
      std::string shard_base_file = merged_index_prefix2 + "_subshard-" + std::to_string(p) + ".bin";
      std::string shard_base_file2 = merged_index_prefix2 + "_subshard_loc-" + std::to_string(p) + ".bin";
      std::string shard_id_file = merged_index_prefix2 + "_subshard-" + std::to_string(p) + "_ids_uint32.bin";
      std::string shard_index_file = merged_index_prefix2 + "_subshard-" + std::to_string(p) + "_mem.index";
      // Required if Index.cpp thinks we are building a multi-file index.
      std::string shard_index_file_data = shard_index_file + ".data";

      std::remove(shard_base_file.c_str());
      std::remove(shard_base_file2.c_str());
      std::remove(shard_id_file.c_str());
      std::remove(shard_index_file.c_str());
      std::remove(shard_index_file_data.c_str());
    }
    return 0;
  }

  // if single_index format is true, we assume that the entire mem index is in
  // mem_index_file, and the entire disk index will be in output_file.
  template<typename T, typename TagT>
  void create_disk_layout(const std::string &mem_index_file, const std::string &base_file, const std::string &tag_file,
                          const std::string &pq_pivots_file, const std::string &pq_vectors_file, bool single_file_index,
                          const std::string &output_file) {
    unsigned npts, ndims;

    // amount to read or write in one shot
    _u64 read_blk_size = 64 * 1024 * 1024;
    _u64 write_blk_size = read_blk_size;
    cached_ifstream base_reader;
    std::ifstream vamana_reader;
    _u64 base_offset = 0, vamana_offset = 0, tags_offset = 0;
    bool tags_enabled = false;

    base_reader.open(base_file, read_blk_size);
    vamana_reader.open(mem_index_file, std::ios::binary);
    tags_enabled = tag_file != "";

    base_reader.read((char *) &npts, sizeof(uint32_t));
    base_reader.read((char *) &ndims, sizeof(uint32_t));

    size_t npts_64, ndims_64;
    npts_64 = npts;
    ndims_64 = ndims;

    // create cached reader + writer
    //    size_t          actual_file_size = get_file_size(mem_index_file);
    std::remove(output_file.c_str());
    cached_ofstream diskann_writer;
    diskann_writer.open(output_file, write_blk_size);

    // metadata: width, medoid
    unsigned width_u32, medoid_u32;
    size_t index_file_size;

    vamana_reader.read((char *) &index_file_size, sizeof(uint64_t));

    _u64 vamana_frozen_num = false, vamana_frozen_loc = 0;
    vamana_reader.read((char *) &width_u32, sizeof(unsigned));
    vamana_reader.read((char *) &medoid_u32, sizeof(unsigned));
    vamana_reader.read((char *) &vamana_frozen_num, sizeof(_u64));
    // compute
    _u64 medoid, max_node_len, nnodes_per_sector;
    npts_64 = (_u64) npts;
    medoid = (_u64) medoid_u32;
    if (vamana_frozen_num == 1)
      vamana_frozen_loc = medoid;
    max_node_len = (((_u64) width_u32 + 1) * sizeof(unsigned)) + (ndims_64 * sizeof(T));
    nnodes_per_sector = SECTOR_LEN / max_node_len;  // 0 if max_node_len > SECTOR_LEN

    LOG(INFO) << "medoid: " << medoid << "B";
    LOG(INFO) << "max_node_len: " << max_node_len << "B";
    LOG(INFO) << "nnodes_per_sector: " << nnodes_per_sector << "B";

    // SECTOR_LEN buffer for each sector
    std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(SECTOR_LEN);
    std::unique_ptr<char[]> multisector_buf = std::make_unique<char[]>(ROUND_UP(max_node_len, SECTOR_LEN));
    std::unique_ptr<char[]> node_buf = std::make_unique<char[]>(max_node_len);
    unsigned &nnbrs = *(unsigned *) (node_buf.get() + ndims_64 * sizeof(T));
    unsigned *nhood_buf = (unsigned *) (node_buf.get() + (ndims_64 * sizeof(T)) + sizeof(unsigned));

    // number of sectors (1 for meta data)
    _u64 n_sectors = nnodes_per_sector > 0 ? ROUND_UP(npts_64, nnodes_per_sector) / nnodes_per_sector
                                           : npts_64 * DIV_ROUND_UP(max_node_len, SECTOR_LEN);
    _u64 disk_index_file_size = (n_sectors + 1) * SECTOR_LEN;

    std::vector<_u64> output_file_meta;
    output_file_meta.push_back(npts_64);
    output_file_meta.push_back(ndims_64);
    output_file_meta.push_back(medoid);
    output_file_meta.push_back(max_node_len);
    output_file_meta.push_back(nnodes_per_sector);
    output_file_meta.push_back(vamana_frozen_num);
    output_file_meta.push_back(vamana_frozen_loc);
    output_file_meta.push_back(disk_index_file_size);

    diskann_writer.write(sector_buf.get(), SECTOR_LEN);  // write out the empty
                                                         // first sector, will
                                                         // be populated at the
                                                         // end.

    std::unique_ptr<T[]> cur_node_coords = std::make_unique<T[]>(ndims_64);
    LOG(INFO) << "# sectors: " << n_sectors;
    _u64 cur_node_id = 0;

    if (nnodes_per_sector > 0) {
      for (_u64 sector = 0; sector < n_sectors; sector++) {
        if (sector % 100000 == 0) {
          LOG(INFO) << "Sector #" << sector << "written";
        }
        memset(sector_buf.get(), 0, SECTOR_LEN);
        for (_u64 sector_node_id = 0; sector_node_id < nnodes_per_sector && cur_node_id < npts_64; sector_node_id++) {
          memset(node_buf.get(), 0, max_node_len);
          // read cur node's nnbrs
          vamana_reader.read((char *) &nnbrs, sizeof(unsigned));

          // sanity checks on nnbrs
          if (nnbrs == 0) {
            LOG(INFO) << "ERROR. Found point with no out-neighbors; Point#: " << cur_node_id;
            exit(-1);
          }

          // read node's nhood
          vamana_reader.read((char *) nhood_buf, (std::min)(nnbrs, width_u32) * sizeof(unsigned));
          if (nnbrs > width_u32) {
            vamana_reader.seekg((nnbrs - width_u32) * sizeof(unsigned), vamana_reader.cur);
          }

          // write coords of node first
          //  T *node_coords = data + ((_u64) ndims_64 * cur_node_id);
          base_reader.read((char *) cur_node_coords.get(), sizeof(T) * ndims_64);
          memcpy(node_buf.get(), cur_node_coords.get(), ndims_64 * sizeof(T));

          // write nnbrs
          *(unsigned *) (node_buf.get() + ndims_64 * sizeof(T)) = (std::min)(nnbrs, width_u32);

          // write nhood next
          memcpy(node_buf.get() + ndims_64 * sizeof(T) + sizeof(unsigned), nhood_buf,
                 (std::min)(nnbrs, width_u32) * sizeof(unsigned));

          // get offset into sector_buf
          char *sector_node_buf = sector_buf.get() + (sector_node_id * max_node_len);

          // copy node buf into sector_node_buf
          memcpy(sector_node_buf, node_buf.get(), max_node_len);
          cur_node_id++;
        }
        // flush sector to disk
        diskann_writer.write(sector_buf.get(), SECTOR_LEN);
      }
    } else {
      uint64_t nsectors_per_node = DIV_ROUND_UP(max_node_len, SECTOR_LEN);
      for (uint64_t i = 0; i < npts_64; i++) {
        if ((i * nsectors_per_node) % 100000 == 0) {
          LOG(INFO) << "Sector #" << i * nsectors_per_node << "written";
        }
        memset(multisector_buf.get(), 0, nsectors_per_node * SECTOR_LEN);

        memset(node_buf.get(), 0, max_node_len);
        // read cur node's nnbrs
        vamana_reader.read((char *) &nnbrs, sizeof(uint32_t));

        // read node's nhood
        vamana_reader.read((char *) nhood_buf, (std::min)(nnbrs, width_u32) * sizeof(uint32_t));
        if (nnbrs > width_u32) {
          vamana_reader.seekg((nnbrs - width_u32) * sizeof(uint32_t), vamana_reader.cur);
        }

        // write coords of node first
        //  T *node_coords = data + ((uint64_t) ndims_64 * cur_node_id);
        base_reader.read((char *) cur_node_coords.get(), sizeof(T) * ndims_64);
        memcpy(multisector_buf.get(), cur_node_coords.get(), ndims_64 * sizeof(T));

        // write nnbrs
        *(uint32_t *) (multisector_buf.get() + ndims_64 * sizeof(T)) = (std::min)(nnbrs, width_u32);

        // write nhood next
        memcpy(multisector_buf.get() + ndims_64 * sizeof(T) + sizeof(uint32_t), nhood_buf,
               (std::min)(nnbrs, width_u32) * sizeof(uint32_t));

        // flush sector to disk
        diskann_writer.write(multisector_buf.get(), nsectors_per_node * SECTOR_LEN);
      }
    }

    diskann_writer.close();
    size_t tag_bytes_written = 0;

    // frozen point implies dynamic index which must have tags
    if (vamana_frozen_num > 0) {
      std::unique_ptr<TagT[]> mem_index_tags;
      size_t nr, nc;
      pipeann::load_bin<TagT>(tag_file, mem_index_tags, nr, nc, tags_offset);

      if (nr != npts_64 && nc != 1) {
        LOG(ERROR) << "Error loading tags file. File dims are " << nr << ", " << nc << ", but expecting " << npts_64
                   << " tags in 1 dimension (bin format).";

        crash();
      }

      pipeann::save_bin<TagT>(output_file + std::string(".tags"), mem_index_tags.get(), nr, nc);
    } else {
      if (tags_enabled) {
        std::unique_ptr<TagT[]> mem_index_tags;
        size_t nr, nc;

        if (!file_exists(tag_file)) {
          LOG(INFO) << "Static vamana index, tag file " << tag_file << "does not exist. Exiting....";
          exit(-1);
        }

        pipeann::load_bin<TagT>(tag_file, mem_index_tags, nr, nc, tags_offset);

        if (nr != npts_64 && nc != 1) {
          LOG(ERROR) << "Error loading tags file. File dims are " << nr << ", " << nc << ", but expecting " << npts_64
                     << " tags in 1 dimension (bin format).";
          crash();
        }

        pipeann::save_bin<TagT>(output_file + std::string(".tags"), mem_index_tags.get(), nr, nc);
      }
    }

    output_file_meta.push_back(output_file_meta[output_file_meta.size() - 1] + tag_bytes_written);
    pipeann::save_bin<_u64>(output_file, output_file_meta.data(), output_file_meta.size(), 1, 0);
    LOG(INFO) << "Output file written.";
  }

  // =========================================================================
  // Alpha Range Merge Utilities
  // =========================================================================
  constexpr uint32_t kMergedMaxAlphaRangeLen = 2;

  // Method 2: Greedy Merge by Gap
  // Sort by start, greedily merge adjacent pairs with smallest gap,
  // until <= target_max pairs remain.
  std::vector<std::pair<int8_t, int8_t>>
  merge_ranges_greedy(const std::vector<std::pair<int8_t, int8_t>>& ranges,
                      uint32_t target_max) {
    if (ranges.size() <= target_max || ranges.empty()) return ranges;

    auto result = ranges;
    std::sort(result.begin(), result.end());

    while (result.size() > target_max) {
      size_t best_idx = 0;
      int16_t best_gap = INT16_MAX;
      for (size_t i = 0; i + 1 < result.size(); i++) {
        int16_t gap = (int16_t)result[i + 1].first - (int16_t)result[i].second;
        if (gap < best_gap) {
          best_gap = gap;
          best_idx = i;
        }
      }
      // Merge result[best_idx] and result[best_idx+1]
      result[best_idx].second = std::max(result[best_idx].second, result[best_idx + 1].second);
      result.erase(result.begin() + best_idx + 1);
    }
    return result;
  }

  // Method 3: Max-Coverage (optimal)
  // Given a coverage bitmap [0..100], find <= target_max intervals
  // that maximize the number of covered points.
  std::vector<std::pair<int8_t, int8_t>>
  merge_ranges_max_coverage(const std::vector<std::pair<int8_t, int8_t>>& ranges,
                            uint32_t target_max) {
    if (ranges.size() <= target_max || ranges.empty()) return ranges;
    if (target_max == 0) return {};

    // Build coverage bitmap for alpha 0..100
    bool covered[101] = {false};
    for (const auto& r : ranges) {
      int8_t lo = std::max<int8_t>(0, r.first);
      int8_t hi = std::min<int8_t>(100, r.second);
      for (int i = lo; i <= hi; i++) covered[i] = true;
    }

    // Find first and last covered point
    int first = -1, last = -1;
    for (int i = 0; i <= 100; i++) {
      if (covered[i]) {
        if (first < 0) first = i;
        last = i;
      }
    }
    if (first < 0) return {};  // no covered points

    if (target_max == 1) {
      return {{(int8_t)first, (int8_t)last}};
    }

    // target_max >= 2: find the largest contiguous uncovered gap
    // and split there to get 2 intervals
    int best_gap_start = -1, best_gap_end = -1;
    int best_gap_len = -1;
    int gap_start = -1;
    for (int i = first; i <= last; i++) {
      if (!covered[i]) {
        if (gap_start < 0) gap_start = i;
      } else {
        if (gap_start >= 0) {
          int gap_len = i - gap_start;
          if (gap_len > best_gap_len) {
            best_gap_len = gap_len;
            best_gap_start = gap_start;
            best_gap_end = i - 1;
          }
          gap_start = -1;
        }
      }
    }
    // Check trailing gap
    if (gap_start >= 0) {
      int gap_len = last + 1 - gap_start;
      if (gap_len > best_gap_len) {
        best_gap_len = gap_len;
        best_gap_start = gap_start;
        best_gap_end = last;
      }
    }

    if (best_gap_len <= 0) {
      // No uncovered gap — single interval covers everything
      return {{(int8_t)first, (int8_t)last}};
    }

    // Split at the largest gap
    std::vector<std::pair<int8_t, int8_t>> result;
    if (best_gap_start > first) {
      result.push_back({(int8_t)first, (int8_t)(best_gap_start - 1)});
    }
    if (best_gap_end < last) {
      result.push_back({(int8_t)(best_gap_end + 1), (int8_t)last});
    }
    if (result.empty()) {
      result.push_back({(int8_t)first, (int8_t)last});
    }
    return result;
  }

  // Dispatcher
  std::vector<std::pair<int8_t, int8_t>>
  merge_alpha_ranges(const std::vector<std::pair<int8_t, int8_t>>& ranges,
                     int merge_method) {
    if (ranges.size() <= kMergedMaxAlphaRangeLen) return ranges;
    switch (merge_method) {
      case 2:
        return merge_ranges_greedy(ranges, kMergedMaxAlphaRangeLen);
      case 3:
        return merge_ranges_max_coverage(ranges, kMergedMaxAlphaRangeLen);
      default:
        return ranges;  // no merge
    }
  }

  // if single_index format is true, we assume that the entire mem index is in
  // mem_index_file, and the entire disk index will be in output_file.
  template<typename T, typename TagT>
  void create_disk_layout(const std::string &mem_index_file, const std::string &base1_file,
                          const std::string &base2_file, const std::string &medoids_file,
                          bool single_file_index, const std::string &output_meta_file,
                          const std::string &output_graph_file, const std::string &output_data_file,
                          int merge_method) {
    const size_t PAGE_SIZE = 8192; // 定义页大小
    unsigned npts1, ndims1;
    unsigned npts2, ndims2;

    // amount to read or write in one shot
    _u64 read_blk_size = 64 * 1024 * 1024;
    cached_ifstream base_reader1;
    cached_ifstream base_reader2;
    cached_ifstream vamana_reader;
    cached_ifstream medoids_reader;
    _u64 base_offset = 0, vamana_offset = 0, tags_offset = 0;

    base_reader1.open(base1_file, read_blk_size);
    base_reader2.open(base2_file, read_blk_size);
    vamana_reader.open(mem_index_file, read_blk_size);
    medoids_reader.open(medoids_file, read_blk_size);

    base_reader1.read((char *) &npts1, sizeof(uint32_t));
    base_reader1.read((char *) &ndims1, sizeof(uint32_t));
    base_reader2.read((char *) &npts2, sizeof(uint32_t));
    base_reader2.read((char *) &ndims2, sizeof(uint32_t));

    size_t npts1_64, ndims1_64;
    npts1_64 = npts1;
    ndims1_64 = ndims1;

    size_t npts2_64, ndims2_64;
    npts2_64 = npts2;
    ndims2_64 = ndims2;


    uint64_t node_num = 0;
    uint32_t max_nbr_len = 0;
    uint32_t max_alpha_range_len = 0;
    uint32_t enterpoint_set_size = 0;
    uint32_t emb_dim = ndims1;
    uint32_t loc_dim = ndims2;
    std::vector<uint32_t> enterpoint_set;

    vamana_reader.read((char *) &node_num, sizeof(uint64_t));
    vamana_reader.read((char *) &max_nbr_len, sizeof(uint32_t));
    vamana_reader.read((char *) &max_alpha_range_len, sizeof(uint32_t));

    medoids_reader.read((char *) &enterpoint_set_size, sizeof(uint32_t));
    uint32_t one_val = 1;
    medoids_reader.read((char *) &one_val, sizeof(uint32_t));
    for (size_t i = 0; i < enterpoint_set_size; i ++) {
      uint32_t up = 0;
      medoids_reader.read((char *) &up, sizeof(uint32_t));
      enterpoint_set.emplace_back(up);
    }
    // enterpoint_set_size = 1;
    // enterpoint_set.emplace_back(0);

    const bool compress_alpha = merge_method == 2 || merge_method == 3;
    uint32_t output_max_alpha_range_len =
        compress_alpha ? kMergedMaxAlphaRangeLen : max_alpha_range_len;
    uint32_t single_neighbor_size =
        sizeof(uint32_t) + (2 * output_max_alpha_range_len * sizeof(int8_t));
    size_t fixed_topo_size = sizeof(uint32_t) + (max_nbr_len * single_neighbor_size);
    uint64_t nnodes_per_sector = PAGE_SIZE / fixed_topo_size;

    // 写入 Meta Data (按照 save_graph_disk 的对齐逻辑)
    std::remove(output_meta_file.c_str());
    std::ofstream out_meta(output_meta_file, std::ios::binary);
    size_t raw_meta_size =
            sizeof(node_num) + sizeof(max_nbr_len) + sizeof(output_max_alpha_range_len) +
            sizeof(enterpoint_set_size) + sizeof(emb_dim) + sizeof(loc_dim) +sizeof(nnodes_per_sector) +
            (sizeof(uint32_t) * enterpoint_set_size);

    size_t aligned_meta_size = (raw_meta_size + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
    std::vector<char> meta_buffer(aligned_meta_size, 0);

    char* m_ptr = meta_buffer.data();
    auto copy_meta = [&](const void* src, size_t sz) { std::memcpy(m_ptr, src, sz); m_ptr += sz; };
    copy_meta(&node_num, sizeof(uint32_t));
    copy_meta(&emb_dim, sizeof(uint32_t));
    copy_meta(&loc_dim, sizeof(uint32_t));
    copy_meta(&max_nbr_len, sizeof(uint32_t));
    copy_meta(&output_max_alpha_range_len, sizeof(uint32_t));
    copy_meta(&nnodes_per_sector, sizeof(uint64_t));
    copy_meta(&enterpoint_set_size, sizeof(uint32_t));
    copy_meta(enterpoint_set.data(), sizeof(uint32_t) * enterpoint_set_size);
    LOG(INFO) << "nnodes " << node_num << " emb dim " << emb_dim << " loc dim " << loc_dim
              << " nbrs " << max_nbr_len << " alpha len " << output_max_alpha_range_len
              << " nodes/page " << nnodes_per_sector;
    LOG(INFO) << "ep size: " << enterpoint_set_size ;
    
    out_meta.write(meta_buffer.data(), aligned_meta_size);
    out_meta.close();

    // ==========================================
    // 通用 Buffer 写入 Lambda (核心逻辑)
    // [Image of buffer flushing mechanism to disk storage]
    // ==========================================
    // 参数: 文件流, Buffer vector, 当前Buffer偏移量(引用), 数据源, 数据长度
    auto buffered_write = [&](std::fstream& fs, std::vector<char>& buf, size_t& buf_off, const void* src, size_t size) {
        const char* src_ptr = (const char*)src;
        size_t written = 0;
        while (written < size) {
            size_t space_left = PAGE_SIZE - buf_off;
            size_t chunk = std::min(space_left, size - written);
            
            std::memcpy(buf.data() + buf_off, src_ptr + written, chunk);
            buf_off += chunk;
            written += chunk;

            // 满页落盘
            if (buf_off == PAGE_SIZE) {
                fs.write(buf.data(), PAGE_SIZE);
                buf_off = 0; // 重置偏移量
                // std::fill(buf.begin(), buf.end(), 0); // 性能优化可省略fill，因为会被覆盖
            }
        }
    };
    auto buffered_write_aligned = [&](std::ofstream& fs, std::vector<char>& buf, size_t& buf_off, const void* src, size_t size) {
        // 1. 安全检查：如果单条数据比一整页还大，这种逻辑是无法处理的
        if (size > PAGE_SIZE) {
            throw std::runtime_error("Data too large for a single page");
        }

        // 2. 判断当前页是否放得下
        if (buf_off + size > PAGE_SIZE) {
            // 放不下 -> 也就是“当前写入超过page”
            
            // A. 将当前页剩余空间补 0 (Padding)
            std::memset(buf.data() + buf_off, 0, PAGE_SIZE - buf_off);
            
            // B. 前一个 Page 落盘
            fs.write(buf.data(), PAGE_SIZE);
            
            // C. 开一个新的 Page (重置偏移)
            buf_off = 0;
        }

        // 3. 将数据写入 Buffer (此时 Buffer 空间一定足够)
        std::memcpy(buf.data() + buf_off, src, size);
        buf_off += size;
    };

    // 结束时的 Flush Lambda (处理不足一页的数据)
    auto final_flush = [&](std::ofstream& fs, std::vector<char>& buf, size_t& buf_off) {
        if (buf_off > 0) {
            // 剩余部分补 0
            std::memset(buf.data() + buf_off, 0, PAGE_SIZE - buf_off);
            fs.write(buf.data(), PAGE_SIZE);
        }
    };

    // 写入双向量 Data 文件
    // 格式：[Node 0 Vector 1][Node 0 Vector 2] | [Node 1 Vector 1][Node 1 Vector 2] ...
    std::ofstream out_data(output_data_file, std::ios::binary);
    std::vector<char> data_page_buf(PAGE_SIZE, 0);
    size_t data_off = 0;
    size_t single_node_data_size = (ndims1 + ndims2) * sizeof(T);
    
    std::vector<char> node_data_tmp(single_node_data_size, 0);

    for (uint32_t i = 0; i < npts1; ++i) {
        // 分别从两个基础文件中读取向量并拼接到同一个 Buffer
        base_reader1.read(node_data_tmp.data(), sizeof(T) * ndims1);
        base_reader2.read(node_data_tmp.data() + sizeof(T) * ndims1, sizeof(T) * ndims2);

        buffered_write_aligned(out_data, data_page_buf, data_off, node_data_tmp.data(), single_node_data_size);
    }
    if (data_off > 0) {
        std::memset(data_page_buf.data() + data_off, 0, PAGE_SIZE - data_off);
        out_data.write(data_page_buf.data(), PAGE_SIZE);
    }
    out_data.close();

    // 写入 Graph Topology (拓扑分离)
    std::ofstream out_graph(output_graph_file, std::ios::binary);
    std::vector<char> graph_page_buf(PAGE_SIZE, 0);
    size_t graph_off = 0;
    
    // 临时存储单个节点的定长 Buffer (对应 fixed_topo_size)
    // vamana_reader: 原始变长格式的输入流
    // out_graph: 目标定长格式的输出文件流
    // fixed_topo_size: 提前计算好的定长节点大小

    std::vector<char> node_topo_buffer(fixed_topo_size, 0); 
    size_t graph_buf_off = 0;

    // 记录单个邻居结构在定长布局中的总字节数
    // [ID(4B)] + [Max_Alpha_Range_Len * 2 * int8_t]
    const size_t range_total_bytes_per_nbr =
        2 * output_max_alpha_range_len * sizeof(int8_t);
    const size_t single_nbr_fixed_size = sizeof(uint32_t) + range_total_bytes_per_nbr;

    for (size_t i = 0; i < node_num; i++) {
        // 1. 初始化当前节点的定长 Buffer (Padding 为 0)
        std::memset(node_topo_buffer.data(), 0, fixed_topo_size);
        char* ptr = node_topo_buffer.data();

        // 2. 从原始流中读取当前节点的邻居数量 (GK/neighbor_size)
        unsigned neighbor_size = 0;
        vamana_reader.read((char*)&neighbor_size, sizeof(unsigned));

        // 写入目标定长格式的邻居计数 (由于是定长，若超过 max_degree 需要截断)
        unsigned actual_to_write = std::min(neighbor_size, (unsigned)max_nbr_len);
        std::memcpy(ptr, &actual_to_write, sizeof(uint32_t));
        ptr += sizeof(uint32_t);

        // 3. 逐个处理邻居
        for (unsigned k = 0; k < neighbor_size; k++) {
            // A. 读取原始 ID
            unsigned neighbor_id = 0;
            vamana_reader.read((char*)&neighbor_id, sizeof(unsigned));

            // B. 读取原始 Range Size
            unsigned range_size = 0;
            vamana_reader.read((char*)&range_size, sizeof(unsigned));

            // C. 读取原始的 int8 对
            std::vector<std::pair<int8_t, int8_t>> temp_ranges;
            for (unsigned t = 0; t < range_size; t++) {
                int8_t x, y;
                vamana_reader.read((char*)&x, sizeof(int8_t));
                vamana_reader.read((char*)&y, sizeof(int8_t));
                temp_ranges.push_back({x, y});
            }

            // D. 如果当前邻居超出了定长允许的最大邻居数，则跳过写入（但必须读完原始流）
            if (k < actual_to_write) {
                // 写入定长 ID
                std::memcpy(ptr, &neighbor_id, sizeof(uint32_t));
                char* range_ptr = ptr + sizeof(uint32_t);

                auto output_ranges = compress_alpha
                    ? merge_alpha_ranges(temp_ranges, merge_method)
                    : temp_ranges;
                unsigned actual_ranges_to_copy =
                    std::min((unsigned)output_ranges.size(), output_max_alpha_range_len);
                for (unsigned r = 0; r < actual_ranges_to_copy; r++) {
                    std::memcpy(range_ptr, &output_ranges[r].first, sizeof(int8_t));
                    range_ptr += sizeof(int8_t);
                    std::memcpy(range_ptr, &output_ranges[r].second, sizeof(int8_t));
                    range_ptr += sizeof(int8_t);
                }
                // 指针移动到下一个邻居的起始位置 (single_nbr_fixed_size 包含了所有的 Padding)
                ptr += single_nbr_fixed_size;
            }
        }

        // 4. 将构建好的定长节点写入 Page Buffer
        buffered_write_aligned(out_graph, graph_page_buf, graph_buf_off, node_topo_buffer.data(), fixed_topo_size);

        if (i % 100000 == 0) {
            std::cerr << "Transformed and saved " << i << " nodes...";
        }
    }

    // 5. 刷新最后不满一页的数据
    final_flush(out_graph, graph_page_buf, graph_buf_off);
    out_graph.close();

    LOG(INFO) << "Disk layout creation finished with 3 separate files.";
  }

  template<typename T, typename TagT>
  void create_disk_layout_single_file_aligned(
      const std::string &mem_index_file, const std::string &base1_file,
      const std::string &base2_file, const std::string &medoids_file,
      const std::string &output_file, int merge_method) {
      
      const size_t PAGE_SIZE = 8192;
      _u64 read_blk_size = 64 * 1024 * 1024;

      // 1. 初始化读取流
      cached_ifstream base_reader1, base_reader2, vamana_reader, medoids_reader;
      base_reader1.open(base1_file, read_blk_size);
      base_reader2.open(base2_file, read_blk_size);
      vamana_reader.open(mem_index_file, read_blk_size);
      medoids_reader.open(medoids_file, read_blk_size);

      // 2. 读取原始元数据
      unsigned npts1, ndims1, npts2, ndims2;
      base_reader1.read((char *)&npts1, sizeof(uint32_t));
      base_reader1.read((char *)&ndims1, sizeof(uint32_t));
      base_reader2.read((char *)&npts2, sizeof(uint32_t));
      base_reader2.read((char *)&ndims2, sizeof(uint32_t));

      uint64_t node_num = 0;
      uint32_t max_nbr_len = 0, max_alpha_range_len = 0, ep_size = 0;
      vamana_reader.read((char *)&node_num, sizeof(uint64_t));
      vamana_reader.read((char *)&max_nbr_len, sizeof(uint32_t));
      vamana_reader.read((char *)&max_alpha_range_len, sizeof(uint32_t));

      medoids_reader.read((char *)&ep_size, sizeof(uint32_t));
      uint32_t dummy; medoids_reader.read((char *)&dummy, sizeof(uint32_t));
      std::vector<uint32_t> enterpoint_set(ep_size);
      for (uint32_t i = 0; i < ep_size; i++) {
          medoids_reader.read((char *)&enterpoint_set[i], sizeof(uint32_t));
      }

      // 3. 计算布局参数
      uint32_t emb_dim = ndims1, loc_dim = ndims2;
      size_t data_size = (emb_dim + loc_dim) * sizeof(float);
      uint32_t output_max_alpha_range_len = kMergedMaxAlphaRangeLen;
      uint32_t single_nbr_size = sizeof(uint32_t) + (2 * kMergedMaxAlphaRangeLen * sizeof(int8_t));
      size_t topo_size = sizeof(uint32_t) + (max_nbr_len * single_nbr_size);
      
      size_t total_node_content_size = data_size + topo_size;
      // 每个节点占用的对齐大小（Page 的倍数）
      size_t aligned_node_size = (total_node_content_size + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
      uint64_t nnodes_per_sector = ((PAGE_SIZE / total_node_content_size) <= 1) ? 0 : (PAGE_SIZE / total_node_content_size);

      // 计算 Meta (Header) 需要占用多少个 Page
      size_t raw_meta_bytes = sizeof(uint64_t) * 4 + sizeof(uint32_t) * 5 + (sizeof(uint32_t) * ep_size);
      size_t header_pages = (raw_meta_bytes + PAGE_SIZE - 1) / PAGE_SIZE;
      size_t header_size = header_pages * PAGE_SIZE;

      // 4. 开始写入单文件
      std::ofstream out(output_file, std::ios::binary);

      // --- A. 写入 Header Section ---
      std::vector<char> header_buffer(header_size, 0);
      char* m_ptr = header_buffer.data();
      auto copy_m = [&](const void* s, size_t z) { std::memcpy(m_ptr, s, z); m_ptr += z; };
      
      copy_m(&node_num, sizeof(uint32_t));
      copy_m(&emb_dim, sizeof(uint32_t));
      copy_m(&loc_dim, sizeof(uint32_t));
      copy_m(&max_nbr_len, sizeof(uint32_t));
      copy_m(&output_max_alpha_range_len, sizeof(uint32_t));
      copy_m(&nnodes_per_sector, sizeof(uint64_t)); // 关键：每行跨度
      // copy_m(&aligned_node_size, sizeof(uint64_t)); // 关键：每行跨度
      // copy_m(&header_size, sizeof(uint64_t));       // 关键：数据区起始偏移
      copy_m(&ep_size, sizeof(uint32_t));
      copy_m(enterpoint_set.data(), sizeof(uint32_t) * ep_size);
      LOG(INFO) << "nnodes " << node_num << " emb dim " << emb_dim << " loc dim " << loc_dim << " nbrs " << max_nbr_len << " alpha len " << nnodes_per_sector;
      LOG(INFO) << "ep size: " << ep_size << " aligned node size" << aligned_node_size << " header size" << header_size;
      
      out.write(header_buffer.data(), header_size);

      // --- B. 循环写入每个节点 (Row-Aligned) ---
      std::vector<char> row_buffer(aligned_node_size, 0);

      for (size_t i = 0; i < node_num; i++) {
          std::memset(row_buffer.data(), 0, aligned_node_size);
          char* ptr = row_buffer.data();

          // 1. 写入向量数据
          base_reader1.read(ptr, sizeof(T) * ndims1);
          ptr += sizeof(T) * ndims1;
          base_reader2.read(ptr, sizeof(T) * ndims2);
          ptr += sizeof(T) * ndims2;

          // 2. 写入拓扑结构
          unsigned nbr_size = 0;
          vamana_reader.read((char*)&nbr_size, sizeof(unsigned));

          unsigned actual_to_write = std::min(nbr_size, (unsigned)max_nbr_len);
          std::memcpy(ptr, &actual_to_write, sizeof(uint32_t));
          char* nbr_ptr = ptr + sizeof(uint32_t);

          for (unsigned k = 0; k < nbr_size; k++) {
              unsigned nbr_id, r_size;
              vamana_reader.read((char*)&nbr_id, sizeof(unsigned));
              vamana_reader.read((char*)&r_size, sizeof(unsigned));
              
              // 预读 Range 数据并读完流
              std::vector<int8_t> r_data(r_size * 2);
              vamana_reader.read((char*)r_data.data(), r_size * 2);

              if (k < actual_to_write) {
                  std::memcpy(nbr_ptr, &nbr_id, sizeof(uint32_t));
                  // Convert flat int8_t vector to pairs, merge, write
                  std::vector<std::pair<int8_t, int8_t>> temp_ranges;
                  for (unsigned t = 0; t < r_size; t++) {
                      temp_ranges.push_back({r_data[t * 2], r_data[t * 2 + 1]});
                  }
                  auto merged = merge_alpha_ranges(temp_ranges, merge_method);
                  unsigned to_copy = std::min((unsigned)merged.size(), kMergedMaxAlphaRangeLen);
                  for (unsigned r = 0; r < to_copy; r++) {
                      std::memcpy(nbr_ptr + sizeof(uint32_t) + r * 2, &merged[r].first, sizeof(int8_t));
                      std::memcpy(nbr_ptr + sizeof(uint32_t) + r * 2 + 1, &merged[r].second, sizeof(int8_t));
                  }
                  nbr_ptr += single_nbr_size;
              }
          }

          // 3. 将整行数据（已对齐且包含 Padding）写入文件
          out.write(row_buffer.data(), aligned_node_size);

          if (i % 100000 == 0) {
              std::cerr << "Written " << i << " nodes into single file index..." << std::endl;
          }
      }

      out.close();
      LOG(INFO) << "Single-file aligned index created at: " << output_file;
      LOG(INFO) << "Header size: " << header_size << ", Node aligned size: " << aligned_node_size;
  }

  template<typename T, typename TagT>
  bool build_disk_index(const char *dataPath, const char *indexFilePath, const char *indexBuildParameters,
                        pipeann::Metric _compareMetric, bool single_file_index, const char *tag_file) {
    std::stringstream parser;
    parser << std::string(indexBuildParameters);
    std::string cur_param;
    std::vector<std::string> param_list;
    while (parser >> cur_param)
      param_list.push_back(cur_param);

    if (param_list.size() != 5 && param_list.size() != 6) {
      LOG(INFO) << "Correct usage of parameters is: R (max degree)"
                   " L (indexing list size, should be >= R) "
                   " B (RAM limit of final index in GB) "
                   " M (memory limit while indexing in GB)"
                   " T (number of threads for indexing) "
                   " [C (compression ratio for PQ. Overrides parameter value B)] ";
      return false;
    }

    std::string dataFilePath(dataPath);
    std::string index_prefix_path(indexFilePath);
    std::string pq_pivots_path = index_prefix_path + "_pq_pivots.bin";
    std::string pq_compressed_vectors_path = index_prefix_path + "_pq_compressed.bin";
    std::string mem_index_path = index_prefix_path + "_mem.index";
    std::string disk_index_path = index_prefix_path + "_disk.index";
    std::string medoids_path = disk_index_path + "_medoids.bin";
    std::string centroids_path = disk_index_path + "_centroids.bin";

    unsigned R = (unsigned) atoi(param_list[0].c_str());
    unsigned L = (unsigned) atoi(param_list[1].c_str());

    double final_index_ram_limit = get_memory_budget(param_list[2]);
    if (final_index_ram_limit <= 0) {
      LOG(ERROR) << "Insufficient memory budget (or string was not in right "
                    "format). Should be > 0.";
      return false;
    }
    double indexing_ram_budget = (float) atof(param_list[3].c_str());
    if (indexing_ram_budget <= 0) {
      LOG(ERROR) << "Not building index. Please provide more RAM budget";
      return false;
    }
    _u32 num_threads = (_u32) atoi(param_list[4].c_str());

    if (num_threads != 0) {
      omp_set_num_threads(num_threads);
    }

    LOG(INFO) << "Starting index build: R=" << R << " L=" << L << " Query RAM budget: " << final_index_ram_limit
              << " Indexing RAM budget: " << indexing_ram_budget << " T: " << num_threads << " Final index will be in "
              << (single_file_index ? "single file" : "multiple files");

    std::string normalized_file_path = dataFilePath;
    if (_compareMetric == pipeann::Metric::COSINE) {
      if (std::is_floating_point<T>::value) {
        LOG(INFO) << "Cosine metric chosen. Normalizing vectors and "
                     "changing distance to L2 to boost accuracy.";

        normalized_file_path = std::string(indexFilePath) + "_data.normalized.bin";
        normalize_data_file(dataFilePath, normalized_file_path);
        _compareMetric = pipeann::Metric::L2;
      } else {
        LOG(ERROR) << "WARNING: Cannot normalize integral data types."
                   << " Using cosine distance with integer data types may "
                      "result in poor recall."
                   << " Consider using L2 distance with integral data types.";
      }
    }

    auto s = std::chrono::high_resolution_clock::now();

    size_t points_num, dim;

    pipeann::get_bin_metadata(normalized_file_path, points_num, dim);
    auto training_set_size = PQ_TRAINING_SET_FRACTION * points_num > MAX_PQ_TRAINING_SET_SIZE
                                 ? MAX_PQ_TRAINING_SET_SIZE
                                 : (_u32) std::round(PQ_TRAINING_SET_FRACTION * points_num);
    training_set_size = (training_set_size == 0) ? 1 : training_set_size;
    LOG(INFO) << "(Normalized, if required) file : " << normalized_file_path << " has: " << points_num
              << " points. Changing training set size to " << training_set_size << " points";

    size_t num_pq_chunks = 64;  // calculate_num_pq_chunks(final_index_ram_limit, points_num, dim);

    size_t train_size, train_dim;
    float *train_data = nullptr;  // maximum: 256000 * dim * data_size, 1GB for 1024-dim float vector.

    auto start = std::chrono::high_resolution_clock::now();
    double p_val = ((double) training_set_size / (double) points_num);
    // generates random sample and sets it to train_data and updates train_size
    gen_random_slice<T>(normalized_file_path, p_val, train_data, train_size, train_dim);

    LOG(INFO) << "Generating PQ pivots with training data of size: " << train_size
              << " num PQ chunks: " << num_pq_chunks;
    generate_pq_pivots(train_data, train_size, (uint32_t) dim, 256, (uint32_t) num_pq_chunks, NUM_KMEANS,
                       pq_pivots_path);
    auto end = std::chrono::high_resolution_clock::now();

    LOG(INFO) << "Pivots generated in " << std::chrono::duration<double>(end - start).count() << "s.";
    start = std::chrono::high_resolution_clock::now();
    generate_pq_data_from_pivots<T>(normalized_file_path, 256, (uint32_t) num_pq_chunks, pq_pivots_path,
                                    pq_compressed_vectors_path);  // 64MB.

    if (1) {  // start to refine pq pivots
      std::cout << "start to refine pq pivots" << std::endl;
      std::string pq_pivots_refined_path = index_prefix_path + "_pq_pivots_refined.bin";
      std::string pq_compressed_vectors_refined_path = index_prefix_path + "_pq_compressed_refined.bin";
      std::string pq_pivots_1_path = index_prefix_path + "_pq_pivots_1.bin";
      std::string pq_compressed_vectors_1_path = index_prefix_path + "_pq_compressed_1.bin";
      std::string pq_pivots_2_path = index_prefix_path + "_pq_pivots_2.bin";
      std::string pq_compressed_vectors_2_path = index_prefix_path + "_pq_compressed_2.bin";
      // 512
      int type = 1;
      if (type == 1) {
        refine_pq_pivots_from_errors_toppercent_in_one_file<T>(
            normalized_file_path, pq_pivots_path, pq_pivots_refined_path, index_prefix_path + "_map.bin",
            pq_compressed_vectors_refined_path, 256, (uint32_t) num_pq_chunks, NUM_KMEANS, 0.20);
        // refine_pq_pivots_from_errors_toppercent<T>(normalized_file_path, pq_pivots_path, pq_pivots_refined_path,
        // index_prefix_path + "_map.bin", pq_compressed_vectors_refined_path, 256, (uint32_t) num_pq_chunks,
        // NUM_KMEANS, 0.20);
      } else if (type == 2) {
        refine_pq_pivots_from_errors<T>(normalized_file_path, pq_pivots_path, pq_pivots_refined_path,
                                        pq_compressed_vectors_refined_path, 256, (uint32_t) num_pq_chunks, NUM_KMEANS,
                                        0.20);
      } else if (type == 3) {
        build_dual_pq_512_from_scratch<T>(normalized_file_path, pq_pivots_path, pq_pivots_1_path, pq_pivots_2_path,
                                          index_prefix_path + "_map.bin", pq_compressed_vectors_1_path,
                                          pq_compressed_vectors_2_path, 256, (uint32_t) num_pq_chunks, NUM_KMEANS);
      }
      // 双PQ策略
      // 质心合并策略
      std::cout << "refine pq pivots done" << std::endl;
    }
    // exit(0);
    std::string pq_pivots_path_2 = index_prefix_path + "_pq_pivots_2.bin";
    std::string pq_compressed_vectors_path_2 = index_prefix_path + "_pq_compressed_2.bin";
    // std::string pq_pivots_path_tmp = index_prefix_path + "_pq_pivots_tmp.bin";

    std::string shuffled_normalized_file_path = normalized_file_path;
    int use_double_pq = 0;
    if (use_double_pq) {
      LOG(INFO) << "double pq";
      gen_random_slice<T>(shuffled_normalized_file_path, p_val, train_data, train_size, train_dim);
      // train_size = points_num * 0.25;
      // train_data = new float[train_size * dim];
      // LOG(INFO) << "gen_random_slice done";
      // generate_pq_pivots(train_data, train_size, (uint32_t)dim, 256, (uint32_t)num_pq_chunks, NUM_KMEANS,
      // pq_pivots_path_tmp); LOG(INFO) << "generate_pq_pivots done"; generate_train_data<T>(normalized_file_path,
      // train_data, train_size, 256, (uint32_t)num_pq_chunks, pq_pivots_path_tmp);
      LOG(INFO) << "generate_train_data done";
      generate_pq_pivots(train_data, train_size, (uint32_t) dim, 256, (uint32_t) num_pq_chunks, NUM_KMEANS,
                         pq_pivots_path_2);
      LOG(INFO) << "generate_pq_pivots done";
      generate_pq_data_from_pivots<T>(shuffled_normalized_file_path, 256, (uint32_t) num_pq_chunks, pq_pivots_path_2,
                                      pq_compressed_vectors_path_2);
      LOG(INFO) << "generate_pq_data_from_pivots done";
    }

    // delete[] train_data;
    // exit(0);

    train_data = nullptr;
    end = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "Compressed data generated and written in: " << std::chrono::duration<double>(end - start).count()
              << "s.";
    start = std::chrono::high_resolution_clock::now();
    pipeann::build_merged_vamana_index<T>(normalized_file_path, _compareMetric, single_file_index, L, R, p_val,
                                          indexing_ram_budget, mem_index_path, medoids_path, centroids_path, tag_file);
    end = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "Vamana index built in: " << std::chrono::duration<double>(end - start).count() << "s.";

    if (tag_file == nullptr) {
      pipeann::create_disk_layout<T, TagT>(mem_index_path, normalized_file_path, "", pq_pivots_path,
                                           pq_compressed_vectors_path, single_file_index, disk_index_path);
    } else {
      std::string tag_filename = std::string(tag_file);
      pipeann::create_disk_layout<T, TagT>(mem_index_path, normalized_file_path, tag_filename, pq_pivots_path,
                                           pq_compressed_vectors_path, single_file_index, disk_index_path);
    }

    LOG(INFO) << "Deleting memory index file: " << mem_index_path;
    std::remove(mem_index_path.c_str());
    // TODO: This is poor design. The decision to add the ".data" prefix
    // is taken by build_vamana_index. So, we shouldn't repeate it here.
    // Checking to see if we can merge the data and index into one file.
    std::remove((mem_index_path + ".data").c_str());
    if (normalized_file_path != dataFilePath) {
      // then we created a normalized vector file. Delete it.
      LOG(INFO) << "Deleting normalized vector file: " << normalized_file_path;
      std::remove(normalized_file_path.c_str());
    }

    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    LOG(INFO) << "Indexing time: " << diff.count();
    return true;
  }

  template<typename T, typename TagT>
  bool build_disk_index(const char *data1Path, const char *data2Path, const char *indexFilePath,
                        const char *indexBuildParameters, pipeann::Metric _compareMetric, bool single_file_index,
                        const char *tag_file) {
    std::stringstream parser;
    parser << std::string(indexBuildParameters);
    std::string cur_param;
    std::vector<std::string> param_list;
    while (parser >> cur_param)
      param_list.push_back(cur_param);

    if (param_list.size() < 5) {
      LOG(INFO) << "Correct usage of parameters is: R (max degree)"
                   " L (indexing list size, should be >= R) "
                   " B (RAM limit of final index in GB) "
                   " M (memory limit while indexing in GB)"
                   " T (number of threads for indexing) "
                   " [C (compression ratio for PQ. Overrides parameter value B)] "
                   " [--merge_method 2|3]";
      return false;
    }

    // Parse optional --merge_method parameter
    int merge_method = 0;
    for (size_t pi = 5; pi + 1 < param_list.size(); pi++) {
      if (param_list[pi] == "--merge_method") {
        merge_method = std::atoi(param_list[pi + 1].c_str());
        break;
      }
    }

    std::string data1FilePath(data1Path);
    std::string data2FilePath(data2Path);
    std::string index_prefix_path(indexFilePath);
    std::string pq_pivots_path1 = index_prefix_path + "emb_pq_pivots.bin";
    std::string pq_compressed_vectors_path1 = index_prefix_path + "emb_pq_compressed.bin";
    std::string mem_index_path1 = index_prefix_path + "1_mem.index";
    std::string disk_index_path1 = index_prefix_path + "1_disk.index";
    std::string medoids_path1 = disk_index_path1 + "1_medoids.bin";
    std::string centroids_path1 = disk_index_path1 + "1_centroids.bin";
    std::string pq_pivots_path2 = index_prefix_path + "loc_pq_pivots.bin";
    std::string pq_compressed_vectors_path2 = index_prefix_path + "loc_pq_compressed.bin";
    std::string mem_index_path2 = index_prefix_path + "2_mem.index";
    std::string disk_index_path2 = index_prefix_path + "2_disk.index";
    std::string medoids_path2 = disk_index_path2 + "2_medoids.bin";
    std::string centroids_path2 = disk_index_path2 + "2_centroids.bin";
    std::string mem_index_path = index_prefix_path + "mem.index";
    std::string medoids_path = index_prefix_path + "medoids.bin";
    std::string disk_meta_path = index_prefix_path + "_disk.index";
    std::string disk_graph_path = index_prefix_path + "disk_index_graph";
    std::string disk_data_path = index_prefix_path + "disk_index_data";
    std::string disk_single_path = index_prefix_path + "single_index";

    unsigned R = (unsigned) atoi(param_list[0].c_str());
    unsigned L = (unsigned) atoi(param_list[1].c_str());

    double final_index_ram_limit = get_memory_budget(param_list[2]);
    if (final_index_ram_limit <= 0) {
      LOG(ERROR) << "Insufficient memory budget (or string was not in right "
                    "format). Should be > 0.";
      return false;
    }
    double indexing_ram_budget = (float) atof(param_list[3].c_str());
    if (indexing_ram_budget <= 0) {
      LOG(ERROR) << "Not building index. Please provide more RAM budget";
      return false;
    }
    _u32 num_threads = (_u32) atoi(param_list[4].c_str());

    if (num_threads != 0) {
      omp_set_num_threads(num_threads);
    }

    LOG(INFO) << "Starting index build: R=" << R << " L=" << L << " Query RAM budget: " << final_index_ram_limit
              << " Indexing RAM budget: " << indexing_ram_budget << " T: " << num_threads << " Final index will be in "
              << (single_file_index ? "single file" : "multiple files");

    std::string normalized_file_path1 = data1FilePath;
    if (_compareMetric == pipeann::Metric::COSINE) {
      if (std::is_floating_point<T>::value) {
        LOG(INFO) << "Cosine metric chosen. Normalizing vectors and "
                     "changing distance to L2 to boost accuracy.";

        normalized_file_path1 = std::string(indexFilePath) + "_data.normalized.bin";
        normalize_data_file(data1FilePath, normalized_file_path1);
        _compareMetric = pipeann::Metric::L2;
      } else {
        LOG(ERROR) << "WARNING: Cannot normalize integral data types."
                   << " Using cosine distance with integer data types may "
                      "result in poor recall."
                   << " Consider using L2 distance with integral data types.";
      }
    }

    std::string normalized_file_path2 = data2FilePath;
    if (_compareMetric == pipeann::Metric::COSINE) {
      if (std::is_floating_point<T>::value) {
        LOG(INFO) << "Cosine metric chosen. Normalizing vectors and "
                     "changing distance to L2 to boost accuracy.";

        normalized_file_path2 = std::string(indexFilePath) + "_data.normalized.bin";
        normalize_data_file(data2FilePath, normalized_file_path2);
        _compareMetric = pipeann::Metric::L2;
      } else {
        LOG(ERROR) << "WARNING: Cannot normalize integral data types."
                   << " Using cosine distance with integer data types may "
                      "result in poor recall."
                   << " Consider using L2 distance with integral data types.";
      }
    }

    auto s = std::chrono::high_resolution_clock::now();

    size_t points_num1, dim1;
    size_t points_num2, dim2;

    pipeann::get_bin_metadata(normalized_file_path1, points_num1, dim1);
    auto training_set_size1 = PQ_TRAINING_SET_FRACTION * points_num1 > MAX_PQ_TRAINING_SET_SIZE
                                  ? MAX_PQ_TRAINING_SET_SIZE
                                  : (_u32) std::round(PQ_TRAINING_SET_FRACTION * points_num1);
    training_set_size1 = (training_set_size1 == 0) ? 1 : training_set_size1;
    LOG(INFO) << "(Normalized, if required) file : " << normalized_file_path1 << " has: " << points_num1
              << " points. Changing training set size to " << training_set_size1 << " points";

    pipeann::get_bin_metadata(normalized_file_path2, points_num2, dim2);
    auto training_set_size2 = PQ_TRAINING_SET_FRACTION * points_num2 > MAX_PQ_TRAINING_SET_SIZE
                                  ? MAX_PQ_TRAINING_SET_SIZE
                                  : (_u32) std::round(PQ_TRAINING_SET_FRACTION * points_num2);
    training_set_size2 = (training_set_size2 == 0) ? 1 : training_set_size2;
    LOG(INFO) << "(Normalized, if required) file : " << normalized_file_path2 << " has: " << points_num2
              << " points. Changing training set size to " << training_set_size2 << " points";

    size_t num_pq_chunks = 64;  // calculate_num_pq_chunks(final_index_ram_limit, points_num, dim);

    size_t train_size1, train_dim1;
    size_t train_size2, train_dim2;
    float *train_data1 = nullptr;  // maximum: 256000 * dim * data_size, 1GB for 1024-dim float vector.
    float *train_data2 = nullptr;  // maximum: 256000 * dim * data_size, 1GB for 1024-dim float vector.

    auto start = std::chrono::high_resolution_clock::now();
    double p_val = ((double) training_set_size1 / (double) points_num1);
    // generates random sample and sets it to train_data and updates train_size
#ifdef ENABLE_PQ_GEN
    gen_random_slice<T>(normalized_file_path1, p_val, train_data1, train_size1, train_dim1);
    gen_random_slice<T>(normalized_file_path2, p_val, train_data2, train_size2, train_dim2);
    LOG(INFO) << "Generating PQ pivots with training data of size: " << train_size1
              << " num PQ chunks: " << num_pq_chunks;
    LOG(INFO) << "Generating PQ pivots with training data of size: " << train_size2
              << " num PQ chunks: " << num_pq_chunks;
    generate_pq_pivots(train_data1, train_size1, (uint32_t) dim1, 256, (uint32_t) num_pq_chunks, NUM_KMEANS,
                       pq_pivots_path1);
    generate_pq_pivots(train_data2, train_size2, (uint32_t) dim2, 256, (uint32_t) num_pq_chunks, NUM_KMEANS,
                       pq_pivots_path2);
#else
    LOG(INFO) << "PQ Generation is skipped (compiled with -DENABLE_PQ_GEN=OFF)";
#endif
    auto end = std::chrono::high_resolution_clock::now();

#ifdef ENABLE_PQ_GEN
    LOG(INFO) << "Pivots generated in " << std::chrono::duration<double>(end - start).count() << "s.";
    start = std::chrono::high_resolution_clock::now();
    generate_pq_data_from_pivots<T>(normalized_file_path1, 256, (uint32_t) num_pq_chunks, pq_pivots_path1,
                                    pq_compressed_vectors_path1);  // 64MB.
    generate_pq_data_from_pivots<T>(normalized_file_path2, 256, (uint32_t) num_pq_chunks, pq_pivots_path2,
                                    pq_compressed_vectors_path2);  // 64MB.
#else
    LOG(INFO) << "PQ Generation is skipped (compiled with -DENABLE_PQ_GEN=OFF)";
#endif

    // if (1) {  // start to refine pq pivots
    //   std::cout << "start to refine pq pivots" << std::endl;
    //   std::string pq_pivots_refined_path = index_prefix_path + "_pq_pivots_refined.bin";
    //   std::string pq_compressed_vectors_refined_path = index_prefix_path + "_pq_compressed_refined.bin";
    //   std::string pq_pivots_1_path = index_prefix_path + "_pq_pivots_1.bin";
    //   std::string pq_compressed_vectors_1_path = index_prefix_path + "_pq_compressed_1.bin";
    //   std::string pq_pivots_2_path = index_prefix_path + "_pq_pivots_2.bin";
    //   std::string pq_compressed_vectors_2_path = index_prefix_path + "_pq_compressed_2.bin";
    //   // 512
    //   int type = 1;
    //   if (type == 1) {
    //     refine_pq_pivots_from_errors_toppercent_in_one_file<T>(
    //         normalized_file_path1, pq_pivots_path1, pq_pivots_refined_path, index_prefix_path + "_map.bin",
    //         pq_compressed_vectors_refined_path, 256, (uint32_t) num_pq_chunks, NUM_KMEANS, 0.20);
    //     // refine_pq_pivots_from_errors_toppercent<T>(normalized_file_path, pq_pivots_path, pq_pivots_refined_path,
    //     // index_prefix_path + "_map.bin", pq_compressed_vectors_refined_path, 256, (uint32_t) num_pq_chunks,
    //     // NUM_KMEANS, 0.20);
    //   } else if (type == 2) {
    //     refine_pq_pivots_from_errors<T>(normalized_file_path1, pq_pivots_path1, pq_pivots_refined_path,
    //                                     pq_compressed_vectors_refined_path, 256, (uint32_t) num_pq_chunks, NUM_KMEANS,
    //                                     0.20);
    //   } else if (type == 3) {
    //     build_dual_pq_512_from_scratch<T>(normalized_file_path1, pq_pivots_path1, pq_pivots_1_path, pq_pivots_2_path,
    //                                       index_prefix_path + "_map.bin", pq_compressed_vectors_1_path,
    //                                       pq_compressed_vectors_2_path, 256, (uint32_t) num_pq_chunks, NUM_KMEANS);
    //   }
    //   // 双PQ策略
    //   // 质心合并策略
    //   std::cout << "refine pq pivots done" << std::endl;
    // }
    // exit(0);
    // std::string pq_pivots_path_2 = index_prefix_path + "_pq_pivots_2.bin";
    // std::string pq_compressed_vectors_path_2 = index_prefix_path + "_pq_compressed_2.bin";
    // std::string pq_pivots_path_tmp = index_prefix_path + "_pq_pivots_tmp.bin";

    // std::string shuffled_normalized_file_path = normalized_file_path;
    // int use_double_pq = 0;
    // if(use_double_pq){
    //   LOG(INFO) << "double pq";
    //   gen_random_slice<T>(shuffled_normalized_file_path, p_val, train_data, train_size, train_dim);
    //   // train_size = points_num * 0.25;
    //   // train_data = new float[train_size * dim];
    //   // LOG(INFO) << "gen_random_slice done";
    //   // generate_pq_pivots(train_data, train_size, (uint32_t)dim, 256, (uint32_t)num_pq_chunks, NUM_KMEANS,
    //   pq_pivots_path_tmp);
    //   // LOG(INFO) << "generate_pq_pivots done";
    //   // generate_train_data<T>(normalized_file_path, train_data, train_size, 256, (uint32_t)num_pq_chunks,
    //   pq_pivots_path_tmp); LOG(INFO) << "generate_train_data done"; generate_pq_pivots(train_data, train_size,
    //   (uint32_t)dim, 256, (uint32_t)num_pq_chunks, NUM_KMEANS, pq_pivots_path_2); LOG(INFO) << "generate_pq_pivots
    //   done"; generate_pq_data_from_pivots<T>(shuffled_normalized_file_path, 256,
    //                                   (uint32_t)num_pq_chunks, pq_pivots_path_2,
    //                                   pq_compressed_vectors_path_2);
    //   LOG(INFO) << "generate_pq_data_from_pivots done";
    // }

    // delete[] train_data;
    // exit(0);

    train_data1 = nullptr;
    end = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "Compressed data generated and written in: " << std::chrono::duration<double>(end - start).count()
              << "s.";
    start = std::chrono::high_resolution_clock::now();
    pipeann::build_merged_vamana_index<T>(normalized_file_path1, normalized_file_path2, _compareMetric,
                                          single_file_index, L, R, p_val, indexing_ram_budget, mem_index_path1,
                                          medoids_path1, centroids_path1, mem_index_path2, medoids_path2,
                                          centroids_path2, mem_index_path, medoids_path, tag_file);
    end = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "Vamana index built in: " << std::chrono::duration<double>(end - start).count() << "s.";

    if (merge_method > 0) {
      LOG(INFO) << "Alpha range merge method: " << merge_method
                << " (1=no merge, 2=greedy, 3=max-coverage), compressing to max "
                << kMergedMaxAlphaRangeLen << " pairs per neighbor";
    }
    // if (tag_file == nullptr) {
      if (single_file_index) {
        pipeann::create_disk_layout_single_file_aligned<T, TagT>(
            mem_index_path, normalized_file_path1, normalized_file_path2,
            medoids_path, disk_single_path, merge_method);
      } else {
        pipeann::create_disk_layout<T, TagT>(
            mem_index_path, normalized_file_path1, normalized_file_path2,
            medoids_path, false, disk_meta_path, disk_graph_path,
            disk_data_path, merge_method);
      }
    // } else {
    //   std::string tag_filename = std::string(tag_file);
    //   pipeann::create_disk_layout<T, TagT>(mem_index_path, normalized_file_path, tag_filename, pq_pivots_path,
    //                                        pq_compressed_vectors_path, single_file_index, disk_index_path);
    // }

    // LOG(INFO) << "Deleting memory index file: " << mem_index_path;
    // std::remove(mem_index_path.c_str());
    // TODO: This is poor design. The decision to add the ".data" prefix
    // is taken by build_vamana_index. So, we shouldn't repeate it here.
    // Checking to see if we can merge the data and index into one file.
    // std::remove((mem_index_path + ".data").c_str());
    // if (normalized_file_path != dataFilePath) {
    //   // then we created a normalized vector file. Delete it.
    //   LOG(INFO) << "Deleting normalized vector file: " << normalized_file_path;
    //   std::remove(normalized_file_path.c_str());
    // }

    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    LOG(INFO) << "Indexing time: " << diff.count();
    return true;
  }

  template<typename T, typename TagT>
  bool build_disk_index_py(const char *dataPath, const char *indexFilePath, uint32_t R, uint32_t L, uint32_t M,
                           uint32_t num_threads, uint32_t PQ_bytes, pipeann::Metric _compareMetric,
                           bool single_file_index, const char *tag_file) {
    std::string dataFilePath(dataPath);
    std::string index_prefix_path(indexFilePath);
    std::string pq_pivots_path = index_prefix_path + "_pq_pivots.bin";
    std::string pq_compressed_vectors_path = index_prefix_path + "_pq_compressed.bin";
    std::string mem_index_path = index_prefix_path + "_mem.index";
    std::string disk_index_path = index_prefix_path + "_disk.index";
    std::string medoids_path = disk_index_path + "_medoids.bin";
    std::string centroids_path = disk_index_path + "_centroids.bin";
    std::string sample_base_prefix = index_prefix_path + "_sample";

    double final_index_ram_limit = get_memory_budget(M);
    if (final_index_ram_limit <= 0) {
      LOG(ERROR) << "Insufficient memory budget (or string was not in right "
                    "format). Should be > 0.";
      return false;
    }

    if (num_threads != 0) {
      omp_set_num_threads(num_threads);
    }

    LOG(INFO) << "Starting index build: R=" << R << " L=" << L << " Query RAM budget: " << final_index_ram_limit
              << " T: " << num_threads << " Final index will be in "
              << (single_file_index ? "single file" : "multiple files");

    std::string normalized_file_path = dataFilePath;
    if (_compareMetric == pipeann::Metric::COSINE) {
      if (std::is_floating_point<T>::value) {
        LOG(INFO) << "Cosine metric chosen. Normalizing vectors and "
                     "changing distance to L2 to boost accuracy.";

        normalized_file_path = std::string(indexFilePath) + "_data.normalized.bin";
        normalize_data_file(dataFilePath, normalized_file_path);
        _compareMetric = pipeann::Metric::L2;
      } else {
        LOG(ERROR) << "WARNING: Cannot normalize integral data types."
                   << " Using cosine distance with integer data types may "
                      "result in poor recall."
                   << " Consider using L2 distance with integral data types.";
      }
    }

    auto s = std::chrono::high_resolution_clock::now();

    size_t points_num, dim;

    pipeann::get_bin_metadata(normalized_file_path, points_num, dim);
    auto training_set_size = PQ_TRAINING_SET_FRACTION * points_num > MAX_PQ_TRAINING_SET_SIZE
                                 ? MAX_PQ_TRAINING_SET_SIZE
                                 : (_u32) std::round(PQ_TRAINING_SET_FRACTION * points_num);
    training_set_size = (training_set_size == 0) ? 1 : training_set_size;
    LOG(INFO) << "(Normalized, if required) file : " << normalized_file_path << " has: " << points_num
              << " points. Changing training set size to " << training_set_size << " points";

    size_t num_pq_chunks = PQ_bytes;

    size_t train_size, train_dim;
    float *train_data;

    auto start = std::chrono::high_resolution_clock::now();
    double p_val = ((double) training_set_size / (double) points_num);
    // generates random sample and sets it to train_data and updates train_size
    gen_random_slice<T>(normalized_file_path, p_val, train_data, train_size, train_dim);

    LOG(INFO) << "Generating PQ pivots with training data of size: " << train_size
              << " num PQ chunks: " << num_pq_chunks;
    generate_pq_pivots(train_data, train_size, (uint32_t) dim, 256, (uint32_t) num_pq_chunks, NUM_KMEANS,
                       pq_pivots_path);
    auto end = std::chrono::high_resolution_clock::now();

    LOG(INFO) << "Pivots generated in " << std::chrono::duration<double>(end - start).count() << "s.";
    start = std::chrono::high_resolution_clock::now();
    generate_pq_data_from_pivots<T>(normalized_file_path, 256, (uint32_t) num_pq_chunks, pq_pivots_path,
                                    pq_compressed_vectors_path);
    delete[] train_data;
    train_data = nullptr;
    end = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "Compressed data generated and written in: " << std::chrono::duration<double>(end - start).count()
              << "s.";
    start = std::chrono::high_resolution_clock::now();
    pipeann::build_merged_vamana_index<T>(normalized_file_path, _compareMetric, single_file_index, L, R, p_val, M,
                                          mem_index_path, medoids_path, centroids_path, tag_file);
    end = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "Vamana index built in: " << std::chrono::duration<double>(end - start).count() << "s.";

    if (tag_file == nullptr) {
      pipeann::create_disk_layout<T, TagT>(mem_index_path, normalized_file_path, "", pq_pivots_path,
                                           pq_compressed_vectors_path, single_file_index, disk_index_path);
    } else {
      std::string tag_filename = std::string(tag_file);
      pipeann::create_disk_layout<T, TagT>(mem_index_path, normalized_file_path, tag_filename, pq_pivots_path,
                                           pq_compressed_vectors_path, single_file_index, disk_index_path);
    }

    double ten_percent_points = std::ceil(points_num * 0.1);
    double num_sample_points =
        ten_percent_points > MAX_SAMPLE_POINTS_FOR_WARMUP ? MAX_SAMPLE_POINTS_FOR_WARMUP : ten_percent_points;
    double sample_sampling_rate = num_sample_points / points_num;
    LOG(INFO) << "Generating warmup file with " << num_sample_points
              << " points using a sampling rate of: " << sample_sampling_rate;
    gen_random_slice<T>(normalized_file_path, sample_base_prefix, sample_sampling_rate);

    LOG(INFO) << "Deleting memory index file: " << mem_index_path;
    std::remove(mem_index_path.c_str());
    // TODO: This is poor design. The decision to add the ".data" prefix
    // is taken by build_vamana_index. So, we shouldn't repeate it here.
    // Checking to see if we can merge the data and index into one file.
    std::remove((mem_index_path + ".data").c_str());
    if (normalized_file_path != dataFilePath) {
      // then we created a normalized vector file. Delete it.
      LOG(INFO) << "Deleting normalized vector file: " << normalized_file_path;
      std::remove(normalized_file_path.c_str());
    }

    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    LOG(INFO) << "Indexing time: " << diff.count();
    return true;
  }

  template void create_disk_layout<int8_t, uint32_t>(const std::string &mem_index_file, const std::string &base_file,
                                                     const std::string &tag_file, const std::string &pq_pivots_file,
                                                     const std::string &pq_compressed_vectors_file,
                                                     bool single_file_index, const std::string &output_file);
  template void create_disk_layout<uint8_t, uint32_t>(const std::string &mem_index_file, const std::string &base_file,
                                                      const std::string &tag_file, const std::string &pq_pivots_file,
                                                      const std::string &pq_compressed_vectors_file,
                                                      bool single_file_index, const std::string &output_file);
  template void create_disk_layout<float, uint32_t>(const std::string &mem_index_file, const std::string &base_file,
                                                    const std::string &tag_file, const std::string &pq_pivots_file,
                                                    const std::string &pq_compressed_vectors_file,
                                                    bool single_file_index, const std::string &output_file);
  // template void create_disk_layout<int8_t, uint64_t>(
  //     const std::string &mem_index_file, const std::string &base_file, const std::string &tag_file,
  //     const std::string &pq_pivots_file, const std::string &pq_compressed_vectors_file, bool single_file_index,
  //     const std::string &output_file);
  // template void create_disk_layout<uint8_t, uint64_t>(
  //     const std::string &mem_index_file, const std::string &base_file, const std::string &tag_file,
  //     const std::string &pq_pivots_file, const std::string &pq_compressed_vectors_file, bool single_file_index,
  //     const std::string &output_file);
  // template void create_disk_layout<float, uint64_t>(
  //     const std::string &mem_index_file, const std::string &base_file, const std::string &tag_file,
  //     const std::string &pq_pivots_file, const std::string &pq_compressed_vectors_file, bool single_file_index,
  //     const std::string &output_file);

  template bool build_disk_index<int8_t, uint32_t>(const char *dataFilePath, const char *indexFilePath,
                                                   const char *indexBuildParameters, pipeann::Metric _compareMetric,
                                                   bool singleFileIndex, const char *tag_file);
  template bool build_disk_index<uint8_t, uint32_t>(const char *dataFilePath, const char *indexFilePath,
                                                    const char *indexBuildParameters, pipeann::Metric _compareMetric,
                                                    bool singleFileIndex, const char *tag_file);
  template bool build_disk_index<float, uint32_t>(const char *dataFilePath, const char *indexFilePath,
                                                  const char *indexBuildParameters, pipeann::Metric _compareMetric,
                                                  bool singleFileIndex, const char *tag_file);
  template bool build_disk_index<float, uint32_t>(const char *data1Path, const char *data2Path,
                                                  const char *indexFilePath, const char *indexBuildParameters,
                                                  pipeann::Metric _compareMetric, bool single_file_index, const char *tag_file);
  template bool build_disk_index<int8_t, uint32_t>(const char *data1Path, const char *data2Path,
                                                  const char *indexFilePath, const char *indexBuildParameters,
                                                  pipeann::Metric _compareMetric, bool single_file_index, const char *tag_file);
  template bool build_disk_index<uint8_t, uint32_t>(const char *data1Path, const char *data2Path,
                                                  const char *indexFilePath, const char *indexBuildParameters,
                                                  pipeann::Metric _compareMetric, bool single_file_index, const char *tag_file);
  // template bool build_disk_index<int8_t, uint64_t>(const char *dataFilePath,
  //                                                                    const char *indexFilePath,
  //                                                                    const char *indexBuildParameters,
  //                                                                    pipeann::Metric _compareMetric,
  //                                                                    bool singleFileIndex, const char *tag_file);
  // template bool build_disk_index<uint8_t, uint64_t>(const char *dataFilePath,
  //                                                                     const char *indexFilePath,
  //                                                                     const char *indexBuildParameters,
  //                                                                     pipeann::Metric _compareMetric,
  //                                                                     bool singleFileIndex, const char *tag_file);
  // template bool build_disk_index<float, uint64_t>(const char *dataFilePath, const char
  // *indexFilePath,
  //                                                                   const char *indexBuildParameters,
  //                                                                   pipeann::Metric _compareMetric,
  //                                                                   bool singleFileIndex, const char *tag_file);

  template int build_merged_vamana_index<int8_t>(std::string base_file, pipeann::Metric _compareMetric,
                                                 bool single_file_index, unsigned L, unsigned R, double sampling_rate,
                                                 double ram_budget, std::string mem_index_path,
                                                 std::string medoids_path, std::string centroids_file,
                                                 const char *tag_file);
  template int build_merged_vamana_index<float>(std::string base_file, pipeann::Metric _compareMetric,
                                                bool single_file_index, unsigned L, unsigned R, double sampling_rate,
                                                double ram_budget, std::string mem_index_path, std::string medoids_path,
                                                std::string centroids_file, const char *tag_file);
  template int build_merged_vamana_index<uint8_t>(std::string base_file, pipeann::Metric _compareMetric,
                                                  bool single_file_index, unsigned L, unsigned R, double sampling_rate,
                                                  double ram_budget, std::string mem_index_path,
                                                  std::string medoids_path, std::string centroids_file,
                                                  const char *tag_file);
  template int build_merged_vamana_index<int8_t>(std::string base1_file, std::string base2_file, pipeann::Metric _compareMetric,
                                bool single_file_index, unsigned L, unsigned R, double sampling_rate, double ram_budget,
                                std::string mem_index_path1, std::string medoids_file1, std::string centroids_file1,
                                std::string mem_index_path2, std::string medoids_file2, std::string centroids_file2,
                                std::string mem_index_path, std::string medoids_file,
                                const char *tag_file) ;
  template int build_merged_vamana_index<uint8_t>(std::string base1_file, std::string base2_file, pipeann::Metric _compareMetric,
                                bool single_file_index, unsigned L, unsigned R, double sampling_rate, double ram_budget,
                                std::string mem_index_path1, std::string medoids_file1, std::string centroids_file1,
                                std::string mem_index_path2, std::string medoids_file2, std::string centroids_file2,
                                std::string mem_index_path, std::string medoids_file,
                                const char *tag_file) ;
  template int build_merged_vamana_index<float>(std::string base1_file, std::string base2_file, pipeann::Metric _compareMetric,
                                bool single_file_index, unsigned L, unsigned R, double sampling_rate, double ram_budget,
                                std::string mem_index_path1, std::string medoids_file1, std::string centroids_file1,
                                std::string mem_index_path2, std::string medoids_file2, std::string centroids_file2,
                                std::string mem_index_path, std::string medoids_file,
                                const char *tag_file);

  template<typename T>
  void UpdateEnterpointSet(std::vector<NeighborH> &enterpoints_skyeline, std::vector<uint32_t> &enterpoints,
                           T* emb_center, T* loc_center, uint32_t id, Metric m, const Parameters &pars) {
    unsigned _aligned_dim1 = pars.Get<unsigned>("dim1");
    unsigned _aligned_dim2 = pars.Get<unsigned>("dim2");
    T *_data1 = pars.Get<T *>("data1");
    T *_data2 = pars.Get<T *>("data2");
    Distance<T> *_distance = get_distance_function<T>(m);
    float e_d = _distance->compare(emb_center, _data1 + _aligned_dim1 * (size_t) id, (unsigned) _aligned_dim1);
    float s_d = _distance->compare(loc_center, _data2 + _aligned_dim2 * (size_t) id, (unsigned) _aligned_dim2);

    enterpoints_skyeline.emplace_back(NeighborH(id, e_d, s_d, true, 0));
    sort(enterpoints_skyeline.begin(), enterpoints_skyeline.end());
    float max_emb_dis = 0;
    float min_emb_dis = 1e9;
    std::vector<NeighborH> skyline;
    for (auto it = enterpoints_skyeline.rbegin(); it != enterpoints_skyeline.rend(); ++it) {
      if (it->distance1 > max_emb_dis) {
        skyline.push_back(*it);
        max_emb_dis = it->distance1;
      }
    }
    enterpoints_skyeline.swap(skyline);
    enterpoints.clear();
    for (size_t i = 0; i < enterpoints_skyeline.size(); i++) {
      enterpoints.push_back(enterpoints_skyeline[i].id);
    }
  }
  // template void UpdateEnterpointSet<float>(std::vector<NeighborH> &enterpoints_skyeline, std::vector<uint32_t> &enterpoints,
  //                          float* emb_center, float* loc_center, uint32_t id, Metric m, const Parameters &pars);
  // template void UpdateEnterpointSet<int8_t>(std::vector<NeighborH> &enterpoints_skyeline, std::vector<uint32_t> &enterpoints,
  //                          T* emb_center, int8_t* loc_center, uint32_t id, Metric m, const Parameters &pars);
  // template void UpdateEnterpointSet<uint8_t>(std::vector<NeighborH> &enterpoints_skyeline, std::vector<uint32_t> &enterpoints,
  //                          T* emb_center, i* loc_center, uint32_t id, Metric m, const Parameters &pars);

  template<typename T>
  int merge_two_graphs(const std::string &graph1_file, const std::string &graph2_file,
                       const std::string &medoids1_file, const std::string &medoids2_file,
                       unsigned max_degree, unsigned L, const std::string &output_graph_file,
                       const std::string &output_medoids_file, pipeann::Metric m, const pipeann::Parameters &paras) {

    cached_ifstream graph1_reader;
    cached_ifstream graph2_reader;
    LOG(INFO) << "Loading graph1 from " << graph1_file;
    graph1_reader.open(graph1_file, 1024*1048576);

    LOG(INFO) << "Loading graph2 from " << graph2_file;
    graph2_reader.open(graph2_file, 1024*1048576);

    _u64 num_pts1 = 0, num_pts2 = 0;
    graph1_reader.read((char *) &num_pts1, sizeof(_u64));
    graph2_reader.read((char *) &num_pts2, sizeof(_u64));
    if (num_pts1 != num_pts2) {
      LOG(ERROR) << "Graphs have different number of points: " << num_pts1 << " vs " << num_pts2;
      return -1;
    }
    cached_ofstream diskann_writer(output_graph_file, 1024 * 1048576);

    uint64_t node_num = 0;
    diskann_writer.write((char *) &node_num, sizeof(uint64_t));
    uint32_t max_nbr_num = max_degree;
    diskann_writer.write((char *) &max_nbr_num, sizeof(uint32_t));
    uint32_t max_alpha_range_num = 0;
    diskann_writer.write((char *) &max_alpha_range_num, sizeof(uint32_t));

    std::string data1_file = paras.Get<std::string>("data1_file");
    std::string data2_file = paras.Get<std::string>("data2_file");
    unsigned _aligned_dim1 = paras.Get<unsigned>("dim1");
    unsigned _aligned_dim2 = paras.Get<unsigned>("dim2");
    T *_data1 = paras.Get<T *>("data1");
    T *_data2 = paras.Get<T *>("data2");

    auto emb_center = std::make_unique<T[]>(_aligned_dim1);
    auto loc_center = std::make_unique<T[]>(_aligned_dim2);
    _u32 _nd = num_pts1;

    for (unsigned j = 0; j < _aligned_dim1; j++)
      emb_center[j] = 0;

    for (unsigned i = 0; i < _nd; i++) {
      for (unsigned j = 0; j < _aligned_dim1; j++) {
        emb_center[j] += *(_data1 + i * _aligned_dim1 + j);
      }
    }

    for (unsigned j = 0; j < _aligned_dim1; j++) {
      emb_center[j] /= _nd;
    }

    for (unsigned j = 0; j < _aligned_dim2; j++)
      loc_center[j] = 0;

    for (unsigned i = 0; i < _nd; i++) {
      for (unsigned j = 0; j < _aligned_dim2; j++) {
        loc_center[j] += *(_data2 + i * _aligned_dim2 + j);
      }
    }

    for (unsigned j = 0; j < _aligned_dim2; j++) {
      loc_center[j] /= _nd;
    }

    std::vector<NeighborH> enterpoints_skyline;
    std::vector<uint32_t> enterpoints;

    uint32_t nnodes = num_pts1;
    unsigned nnbrs = 0, shard_nnbrs = 0;
    std::vector<bool> nhood_set(nnodes, 0);
    std::vector<NeighborH> final_nhood;
    std::random_device rng;
    std::mt19937 urng(rng());
    enterpoints.emplace_back(0);

    size_t max = 0, min = 1 << 30, total = 0, cnt = 0;
    for (size_t i = 0; i < nnodes; i ++) {
      if (i != 0)
        UpdateEnterpointSet(enterpoints_skyline, enterpoints, emb_center.get(), loc_center.get(), i, m, paras);
      graph1_reader.read((char *) &shard_nnbrs, sizeof(unsigned));
      std::vector<NeighborH> shard_nhood1;
      shard_nhood1.reserve(shard_nnbrs);
      for (unsigned k = 0; k < shard_nnbrs; k++) {
        unsigned neighbor_id;
        graph1_reader.read((char *)&neighbor_id, sizeof(unsigned));
        // unsigned range_size;
        float dist1 = 0, dist2 = 0;
        graph1_reader.read((char *)&dist1, sizeof(float));
        graph1_reader.read((char *)&dist2, sizeof(float));
        shard_nhood1.emplace_back(NeighborH(neighbor_id, dist1, dist2, false, 0));
      }
      for (_u64 j = 0; j < shard_nnbrs; j++) {
        if (nhood_set[shard_nhood1[j].id] == 0) {
          nhood_set[shard_nhood1[j].id] = 1;
          final_nhood.emplace_back(shard_nhood1[j].id, shard_nhood1[j].distance1, shard_nhood1[j].distance2, false, 0);
        }
      }

      graph2_reader.read((char *) &shard_nnbrs, sizeof(unsigned));
      std::vector<NeighborH> shard_nhood2;
      shard_nhood2.reserve(shard_nnbrs);
      for (unsigned k = 0; k < shard_nnbrs; k++)
      {
        unsigned neighbor_id;
        graph2_reader.read((char *)&neighbor_id, sizeof(unsigned));
        // unsigned range_size;
        float dist1 = 0, dist2 = 0;
        graph2_reader.read((char *)&dist1, sizeof(float));
        graph2_reader.read((char *)&dist2, sizeof(float));
        shard_nhood2.emplace_back(NeighborH(neighbor_id, dist1, dist2, false, 0));
      }
      for (_u64 j = 0; j < shard_nnbrs; j++) {
        if (nhood_set[shard_nhood2[j].id] == 0) {
          nhood_set[shard_nhood2[j].id] = 1;
          final_nhood.emplace_back(shard_nhood2[j].id, shard_nhood2[j].distance1, shard_nhood2[j].distance2, false, 0);
        }
      }

      nnbrs = (unsigned) (std::min)(final_nhood.size(), (uint64_t) max_degree);
      std::vector<DEGNeighbor> prune_list;
      // prune_neighbors<T>(final_nhood, max_degree, m, prune_list, paras);
      prune_neighbors_no_opt<T>(final_nhood, max_degree, m, prune_list, paras);
      // write into merged ofstream
      max = std::max(max, prune_list.size());
      min = (std::min)(min, prune_list.size());
      total += prune_list.size();
      if (prune_list.size() < 2)
        cnt++;
      nnbrs = prune_list.size();
      // LOG(INFO) << i << " " << nnbrs << " " << final_nhood.size() << " " << max_degree ;
      diskann_writer.write((char *) &nnbrs, sizeof(unsigned));
      for (unsigned k = 0; k < nnbrs; k++) {
        DEGNeighbor &neighbor = prune_list[k];
        unsigned neighbor_id = neighbor.id_;
        diskann_writer.write((char *)&neighbor_id, sizeof(unsigned));
        std::vector<std::pair<float, float>> &use_range = neighbor.available_range;
        unsigned range_size = use_range.size();
        max_alpha_range_num = std::max(range_size, max_alpha_range_num);
        diskann_writer.write((char *)&range_size, sizeof(unsigned));
        for (unsigned t = 0; t < range_size; t++) {
          int8_t x = static_cast<int8_t>(use_range[t].first * 100);
          int8_t y = static_cast<int8_t>(use_range[t].second * 100);
          diskann_writer.write((char *) &x, sizeof(int8_t));
          diskann_writer.write((char *) &y, sizeof(int8_t));
        }
      }
      for (auto &p : final_nhood)
        nhood_set[p.id] = 0;
      final_nhood.clear();
    }

    std::ofstream medoid_writer(output_medoids_file.c_str(), std::ios::binary);
    _u32 one_val = 1;
    uint32_t nmeds = enterpoints.size();
    medoid_writer.write((char *) &nmeds, sizeof(uint32_t));
    medoid_writer.write((char *) &one_val, sizeof(uint32_t));
    for (auto m : enterpoints) {
      medoid_writer.write((char *) &m, sizeof(uint32_t));
    }

    diskann_writer.reset();
    node_num = nnodes;
    diskann_writer.write((char *) &node_num, sizeof(uint64_t));
    diskann_writer.write((char *) &max_nbr_num, sizeof(uint32_t));
    diskann_writer.write((char *) &max_alpha_range_num, sizeof(uint32_t));
    diskann_writer.close();

    LOG(INFO) << "Index merge with degree: max:" << max << "  avg:" << (float) total / (float) (nnodes)
              << "  min:" << min << "  count(deg<2):" << cnt;
    LOG(INFO) << nmeds;
    LOG(INFO) << "Merge completed successfully";
    return 0;
  }

  template int merge_two_graphs<float>(const std::string &graph1_file, const std::string &graph2_file,
                      const std::string &medoids1_file, const std::string &medoids2_file,
                       unsigned max_degree, unsigned L, const std::string &output_graph_file,
                       const std::string &output_medoids_file, pipeann::Metric m, const pipeann::Parameters &paras);
  template int merge_two_graphs<int8_t>(const std::string &graph1_file, const std::string &graph2_file,
                      const std::string &medoids1_file, const std::string &medoids2_file,
                       unsigned max_degree, unsigned L, const std::string &output_graph_file,
                       const std::string &output_medoids_file, pipeann::Metric m, const pipeann::Parameters &paras);
  template int merge_two_graphs<uint8_t>(const std::string &graph1_file, const std::string &graph2_file,
                      const std::string &medoids1_file, const std::string &medoids2_file,
                       unsigned max_degree, unsigned L, const std::string &output_graph_file,
                       const std::string &output_medoids_file, pipeann::Metric m, const pipeann::Parameters &paras);

  void findSkyline(std::vector<NeighborH> &points, std::vector<NeighborH> &skyline,
                        std::vector<NeighborH> &remain_points);

  template<typename T>
  void prune_neighbors_no_opt(std::vector<NeighborH> pool,
                      unsigned R, pipeann::Metric m, std::vector<DEGNeighbor> &pruned_list,
                      const pipeann::Parameters &parameter) {
    Distance<T> *_distance = get_distance_function<T>(m);
    std::vector<DEGNeighbor> picked;
    // pool 按照layer排序 在同层内按照geo_distance排序
    sort(pool.begin(), pool.end());
    std::vector<NeighborH> skyline_result;
    std::vector<NeighborH> remain_points;
    std::vector<NeighborH> result;
    int l = 0;
    while (!pool.empty()) {
        findSkyline(pool, skyline_result, remain_points);
        pool.swap(remain_points);
        for (auto &point : skyline_result)
        {
          result.emplace_back(point.id, point.distance1, point.distance2, true, l);
        }
        skyline_result.clear();
        remain_points.clear();
        l++;
    }
    pool.swap(result);
    std::string data1_file = parameter.Get<std::string>("data1_file");
    std::string data2_file = parameter.Get<std::string>("data2_file");
    unsigned _aligned_dim1 = parameter.Get<unsigned>("dim1");
    unsigned _aligned_dim2 = parameter.Get<unsigned>("dim2");
    T *_data1 = parameter.Get<T *>("data1");
    T *_data2 = parameter.Get<T *>("data2");
    size_t iter = 0;
    int visited_layer = 0;
    const unsigned range = parameter.Get<unsigned>("R");
    auto judge_dominate = [](const std::pair<float, float> &a, const std::pair<float, float> &b) {
    float de_a = a.first;
    float ds_a = a.second;
    float de_b = b.first;
    float ds_b = b.second;

    // 条件1：a 在两个维度都不劣于 b
    bool not_worse = (de_a <= de_b) && (ds_a <= ds_b);

    // 条件2：a 至少在一个维度严格更优
    bool strictly_better = (de_a < de_b) || (ds_a < ds_b);

    return not_worse && strictly_better;
  };
      while (picked.size() < range && iter < pool.size())
      {
          std::vector<NeighborH> candidate;
          while (iter < pool.size())
          {
              if (pool[iter].layer == visited_layer)
              {
                  candidate.emplace_back(pool[iter]);
              }
              else
              {
                  break;
              }
              iter++;
          }
          std::vector<DEGNeighbor> tempres_picked;
          for (size_t i = 0; i < candidate.size(); i++)
          {
              // 这里先初始化useful range 根据斜率算出来
              std::vector<std::pair<float, float>> prune_range;
              float cur_geo_dist = candidate[i].distance2; // s_pq
              float cur_emb_dist = candidate[i].distance1; // e_pq
              for (size_t j = 0; j < picked.size(); j++)
              {
                  const std::vector<std::pair<float, float>> &picked_use_range = picked[j].available_range;
                  // we want to find out if this edge can prune the candidate within its picked_avaiable_range
                auto id = picked[j].id_;
                float xq_e_dist = _distance->compare(_data1 + _aligned_dim1 * (size_t) id,
                                                      _data1 + _aligned_dim1 * (size_t) (size_t) candidate[i].id,
                                                      (unsigned) _aligned_dim1);
                // E(x,q)

                float xq_s_dist = _distance->compare(_data2 + _aligned_dim2 * (size_t) id,
                                                      _data2 + _aligned_dim2 * (size_t) (size_t) candidate[i].id,
                                                      (unsigned) _aligned_dim2);
                  // S(x,q)

                  float exist_e_dist = picked[j].emb_distance_; // e_xp

                  float exist_s_dist = picked[j].geo_distance_; // s_xp

                  // alpha * (E(p,x) - S(p,x) - E(p,q) + S(p,q)) <= S(p,q) - S(p,x)
                  // alpha * (E(q,x) - S(q,x) - E(p,q) + S(p,q)) <= S(p,q) - S(q,x)
                  // if alpha holds on for the two equation at the same time, the edge will be pruned
                  // now for equation 1
                  float diff1 = exist_e_dist - cur_emb_dist + cur_geo_dist - exist_s_dist;
                  float diff2 = cur_geo_dist - exist_s_dist;
                  /*
                  diff1 > 0 && diff2 > 0
                  equation 1 holds on when alpha < = diff2 / diff1
                  diff1 < 0 && diff2 < 0
                  equation 1 holds on when alpha > = diff2 / diff1
                  diff1 < 0 && diff2 > 0
                  the equation hold forever
                  diff1 > 0 && diff2 < 0
                  equation never hold which means this edge will not be pruned by this strategy
                  */
                  // float eq1_prune_upper_alpha = 1;
                  // float eq1_prune_lower_alpha = 0;
                  std::pair<float, float> tmp_prune_range_1;
                  if (diff1 > 0 && diff2 > 0)
                  {
                      // equation 1 holds on when alpha < = diff2 / diff1
                      tmp_prune_range_1 = std::make_pair(0.0f, std::min(diff2 / diff1, 1.0f));
                      // eq1_prune_lower_alpha = diff2 / diff1 ;
                  }
                  else if (diff1 < 0 && diff2 < 0)
                  {
                      // equation 1 holds on when alpha > = diff2 / diff1
                      // eq1_prune_upper_alpha = diff2 / diff1 ;
                      tmp_prune_range_1 = std::make_pair(std::min(diff2 / diff1, 1.0f), 1.0f);
                  }
                  else if (diff1 < 0 && diff2 > 0)
                  {
                      tmp_prune_range_1 = {0.0f, 1.0f};
                      // the equation hold forever
                  }
                  else if (diff1 > 0 && diff2 < 0)
                  {
                      // equation never hold
                      // break;
                      tmp_prune_range_1 = {0.0f, 0.0f};
                  }
                  // now for equation 2
                  float diff3 = xq_e_dist - cur_emb_dist + cur_geo_dist - xq_s_dist;
                  float diff4 = cur_geo_dist - xq_s_dist;
                  /*
                  similar to previous
                  */
                  // when alpha >= eq1_prune_upper_alpha and alpha <= eq1_prune_lower_alpha, the equation holds on
                  // float eq2_prune_upper_alpha = 1;
                  // float eq2_prune_lower_alpha = 0;
                  std::pair<float, float> tmp_prune_range_2;
                  if (diff3 > 0 && diff4 > 0)
                  {
                      // equation 2 holds on when alpha < = diff4 / diff3
                      // eq2_prune_upper_alpha = diff4 / diff3;
                      // tmp_prune_range.second = std::min(tmp_prune_range.second, diff4 / diff3);
                      tmp_prune_range_2 = std::make_pair(0.0f, std::min(1.0f, diff4 / diff3));
                  }
                  else if (diff3 < 0 && diff4 < 0)
                  {
                      // equation 2 holds on when alpha > = diff4 / diff3
                      // eq2_prune_lower_alpha = diff4 / diff3;
                      // tmp_prune_range.first = std::max(tmp_prune_range.first, diff4 / diff3);
                      tmp_prune_range_2 = std::make_pair(std::min(diff4 / diff3, 1.0f), 1.0f);
                  }
                  else if (diff3 < 0 && diff4 > 0)
                  {
                      // the equation hold forever
                      // then we do not change the previous range
                      tmp_prune_range_2 = {0.0f, 1.0f};
                  }
                  else if (diff3 > 0 && diff4 < 0)
                  {
                      // equation never hold
                      // break;
                      tmp_prune_range_2 = {0.0f, 0.0f};
                  }

                  std::pair<float, float> tmp_prune_range;
                  tmp_prune_range.first = std::max(tmp_prune_range_1.first, tmp_prune_range_2.first);
                  tmp_prune_range.second = std::min(tmp_prune_range_1.second, tmp_prune_range_2.second);

                  if (tmp_prune_range.second > tmp_prune_range.first)
                  {
                      // now we consider whether this range is useful range, that is (second > first)
                      // now we check its intersection range with shared_use_range
                      intersection(picked_use_range, tmp_prune_range, prune_range);
                  }
                  else
                  {
                      continue;
                      // this range is not useful, so this edge will not be pruned by this selected edge
                  }
              }
              prune_range = mergeIntervals(prune_range);
              std::vector<std::pair<float, float>> after_pruned_use_range;
              get_use_range(prune_range, after_pruned_use_range);
              float threshold = 0.1;
              float use_size = 0;
              for (int j = 0; j < after_pruned_use_range.size(); j++)
              {
                  use_size = use_size + after_pruned_use_range[j].second - after_pruned_use_range[j].first;
              }
              if (use_size >= threshold)
              {
                  picked.push_back(DEGNeighbor(candidate[i].id, candidate[i].distance1,
                                                          candidate[i].distance2, after_pruned_use_range, visited_layer));
              } else
              {
              }
          }
          visited_layer++;
          if (picked.size() >= range)
              break;
      }
      pruned_list.swap(picked);
  }
  template<typename T>
  void prune_neighbors(std::vector<NeighborH> pool,
                      unsigned R, pipeann::Metric m, std::vector<DEGNeighbor> &pruned_list,
                      const pipeann::Parameters &pars) {
    Distance<T> *_distance = get_distance_function<T>(m);
    std::vector<DEGNeighbor> picked;
      // pool 按照layer排序 在同层内按照geo_distance排序
      sort(pool.begin(), pool.end());
      std::vector<NeighborH> skyline_result;
      std::vector<NeighborH> remain_points;
      std::vector<NeighborH> result;
      int l = 0;
      while (!pool.empty()) {
          findSkyline(pool, skyline_result, remain_points);
          pool.swap(remain_points);
          for (auto &point : skyline_result)
          {
            result.emplace_back(point.id, point.distance1, point.distance2, true, l);
          }
          skyline_result.clear();
          remain_points.clear();
          l++;
      }
      pool.swap(result);
      size_t iter = 0;
      int visited_layer = 0;
      const unsigned range = R;
    
    std::string data1_file = pars.Get<std::string>("data1_file");
    std::string data2_file = pars.Get<std::string>("data2_file");
    unsigned _aligned_dim1 = pars.Get<unsigned>("dim1");
    unsigned _aligned_dim2 = pars.Get<unsigned>("dim2");
    T *_data1 = pars.Get<T *>("data1");
    T *_data2 = pars.Get<T *>("data2");


    auto judge_dominate = [](const std::pair<float, float> &a, const std::pair<float, float> &b) {
      float de_a = a.first;
      float ds_a = a.second;
      float de_b = b.first;
      float ds_b = b.second;

      // 条件1：a 在两个维度都不劣于 b
      bool not_worse = (de_a <= de_b) && (ds_a <= ds_b);

      // 条件2：a 至少在一个维度严格更优
      bool strictly_better = (de_a < de_b) || (ds_a < ds_b);

      return not_worse && strictly_better;
    };
    // pool 按照layer排序 在同层内按照geo_distance排序
    while (picked.size() < range && iter < pool.size()) {
      std::vector<NeighborH> candidate;
      while (iter < pool.size()) {
        if (pool[iter].layer == visited_layer) {
          candidate.emplace_back(pool[iter]);
        } else {
          break;
        }
        iter++;
      }
      std::vector<DEGNeighbor> tempres_picked;

      if (visited_layer == 0) {
        // candidate = spread_from_center(candidate);
        for (size_t i = 0; i < candidate.size(); i++) {
          // 这里先初始化useful range 根据斜率算出来
          std::vector<std::pair<float, float>> prune_range;
          float cur_geo_dist = candidate[i].distance2;  // s_pq
          float cur_emb_dist = candidate[i].distance1;  // e_pq
          for (size_t j = 0; j < tempres_picked.size(); j++) {
            const std::vector<std::pair<float, float>> &picked_use_range = tempres_picked[j].available_range;
            auto id = tempres_picked[j].id_;
            // we want to find out if this edge can prune the candidate within its picked_avaiable_range
            float xq_e_dist = _distance->compare(_data1 + _aligned_dim1 * (size_t) id,
                                                 _data1 + _aligned_dim1 * (size_t) candidate[i].id,
                                                 (unsigned) _aligned_dim1);
            // E(x,q)

            float xq_s_dist = _distance->compare(_data2 + _aligned_dim2 * (size_t) id,
                                                 _data2 + _aligned_dim2 * (size_t) candidate[i].id,
                                                 (unsigned) _aligned_dim2);
            // S(x,q)

            float exist_e_dist = tempres_picked[j].emb_distance_;  // e_xp

            float exist_s_dist = tempres_picked[j].geo_distance_;  // s_xp

            // alpha * (E(p,x) - S(p,x) - E(p,q) + S(p,q)) <= S(p,q) - S(p,x)
            // alpha * (E(q,x) - S(q,x) - E(p,q) + S(p,q)) <= S(p,q) - S(q,x)
            // if alpha holds on for the two equation at the same time, the edge will be pruned
            // now for equation 1
            float diff1 = exist_e_dist - cur_emb_dist + cur_geo_dist - exist_s_dist;
            float diff2 = cur_geo_dist - exist_s_dist;
            /*
            diff1 > 0 && diff2 > 0
            equation 1 holds on when alpha < = diff2 / diff1
            diff1 < 0 && diff2 < 0
            equation 1 holds on when alpha > = diff2 / diff1
            diff1 < 0 && diff2 > 0
            the equation hold forever
            diff1 > 0 && diff2 < 0
            equation never hold which means this edge will not be pruned by this strategy
            */
            // float eq1_prune_upper_alpha = 1;
            // float eq1_prune_lower_alpha = 0;
            std::pair<float, float> tmp_prune_range_1;
            if (diff1 > 0 && diff2 > 0) {
              // equation 1 holds on when alpha < = diff2 / diff1
              tmp_prune_range_1 = std::make_pair(0.0f, std::min(diff2 / diff1, 1.0f));
              // eq1_prune_lower_alpha = diff2 / diff1 ;
            } else if (diff1 < 0 && diff2 < 0) {
              // equation 1 holds on when alpha > = diff2 / diff1
              // eq1_prune_upper_alpha = diff2 / diff1 ;
              tmp_prune_range_1 = std::make_pair(std::min(diff2 / diff1, 1.0f), 1.0f);
            } else if (diff1 < 0 && diff2 > 0) {
              tmp_prune_range_1 = {0.0f, 1.0f};
              // the equation hold forever
            } else if (diff1 > 0 && diff2 < 0) {
              // equation never hold
              // break;
              tmp_prune_range_1 = {0.0f, 0.0f};
            }
            // now for equation 2
            float diff3 = xq_e_dist - cur_emb_dist + cur_geo_dist - xq_s_dist;
            float diff4 = cur_geo_dist - xq_s_dist;
            /*
            similar to previous
            */
            // when alpha >= eq1_prune_upper_alpha and alpha <= eq1_prune_lower_alpha, the equation holds on
            // float eq2_prune_upper_alpha = 1;
            // float eq2_prune_lower_alpha = 0;
            std::pair<float, float> tmp_prune_range_2;
            if (diff3 > 0 && diff4 > 0) {
              // equation 2 holds on when alpha < = diff4 / diff3
              // eq2_prune_upper_alpha = diff4 / diff3;
              // tmp_prune_range.second = std::min(tmp_prune_range.second, diff4 / diff3);
              tmp_prune_range_2 = std::make_pair(0.0f, std::min(1.0f, diff4 / diff3));
            } else if (diff3 < 0 && diff4 < 0) {
              // equation 2 holds on when alpha > = diff4 / diff3
              // eq2_prune_lower_alpha = diff4 / diff3;
              // tmp_prune_range.first = std::max(tmp_prune_range.first, diff4 / diff3);
              tmp_prune_range_2 = std::make_pair(std::min(diff4 / diff3, 1.0f), 1.0f);
            } else if (diff3 < 0 && diff4 > 0) {
              // the equation hold forever
              // then we do not change the previous range
              tmp_prune_range_2 = {0.0f, 1.0f};
            } else if (diff3 > 0 && diff4 < 0) {
              // equation never hold
              // break;
              tmp_prune_range_2 = {0.0f, 0.0f};
            }

            std::pair<float, float> tmp_prune_range;
            tmp_prune_range.first = std::max(tmp_prune_range_1.first, tmp_prune_range_2.first);
            tmp_prune_range.second = std::min(tmp_prune_range_1.second, tmp_prune_range_2.second);

            if (tmp_prune_range.second > tmp_prune_range.first) {
              // now we consider whether this range is useful range, that is (second > first)
              // now we check its intersection range with shared_use_range
              intersection(picked_use_range, tmp_prune_range, prune_range);
            } else {
              continue;
              // this range is not useful, so this edge will not be pruned by this selected edge
            }
          }
          prune_range = mergeIntervals(prune_range);
          std::vector<std::pair<float, float>> after_pruned_use_range;
          get_use_range(prune_range, after_pruned_use_range);
          float threshold = 0.1;
          float use_size = 0;
          for (int j = 0; j < after_pruned_use_range.size(); j++) {
            use_size = use_size + after_pruned_use_range[j].second - after_pruned_use_range[j].first;
          }
          if (use_size >= threshold) {
            tempres_picked.push_back(DEGNeighbor(candidate[i].id, candidate[i].distance1, candidate[i].distance2,
                                                 after_pruned_use_range, visited_layer));
          }
        }
        if (tempres_picked.empty()) {
          continue;
        }
        DEGNeighbor lastnode = tempres_picked[tempres_picked.size() - 1];
        tempres_picked.clear();
        tempres_picked.push_back(lastnode);
        int last_i = 0;
        for (int i = candidate.size() - 1; i >= 0; i--) {
          if (lastnode.id_ == candidate[i].id) {
            last_i = i;
            break;
          }
        }
        for (int i = last_i - 1; i >= 0; i--) {
          // 这里先初始化useful range 根据斜率算出来
          std::vector<std::pair<float, float>> prune_range;
          float cur_geo_dist = candidate[i].distance2;  // s_pq
          float cur_emb_dist = candidate[i].distance1;  // e_pq
          for (size_t j = 0; j < tempres_picked.size(); j++) {
            const std::vector<std::pair<float, float>> &picked_use_range = tempres_picked[j].available_range;
            // we want to find out if this edge can prune the candidate within its picked_avaiable_range
            float xq_e_dist =
                _distance->compare(_data1 + _aligned_dim1 * (size_t) tempres_picked[j].id_,
                                   _data1 + _aligned_dim1 * (size_t) candidate[i].id, (unsigned) _aligned_dim1);
            // E(x,q)

            float xq_s_dist =
                _distance->compare(_data2 + _aligned_dim2 * (size_t) tempres_picked[j].id_,
                                   _data2 + _aligned_dim2 * (size_t) candidate[i].id, (unsigned) _aligned_dim2);
            // S(x,q)
            // float xq_e_dist = index->get_E_Dist()->compare(
            //     index->getBaseEmbData() + (size_t)tempres_picked[j].id_ * index->getBaseEmbDim(),
            //     index->getBaseEmbData() + (size_t)candidate[i].id_ * index->getBaseEmbDim(),
            //     index->getBaseEmbDim());

            // float xq_s_dist = index->get_S_Dist()->compare(
            //     index->getBaseLocData() + (size_t)tempres_picked[j].id_ * index->getBaseLocDim(),
            //     index->getBaseLocData() + (size_t)candidate[i].id_ * index->getBaseLocDim(),
            //     index->getBaseLocDim());

            float exist_e_dist = tempres_picked[j].emb_distance_;  // e_xp

            float exist_s_dist = tempres_picked[j].geo_distance_;  // s_xp

            // alpha * (E(p,x) - S(p,x) - E(p,q) + S(p,q)) <= S(p,q) - S(p,x)
            // alpha * (E(q,x) - S(q,x) - E(p,q) + S(p,q)) <= S(p,q) - S(q,x)
            // if alpha holds on for the two equation at the same time, the edge will be pruned
            // now for equation 1
            float diff1 = exist_e_dist - cur_emb_dist + cur_geo_dist - exist_s_dist;
            float diff2 = cur_geo_dist - exist_s_dist;
            /*
            diff1 > 0 && diff2 > 0
            equation 1 holds on when alpha < = diff2 / diff1
            diff1 < 0 && diff2 < 0
            equation 1 holds on when alpha > = diff2 / diff1
            diff1 < 0 && diff2 > 0
            the equation hold forever
            diff1 > 0 && diff2 < 0
            equation never hold which means this edge will not be pruned by this strategy
            */
            // float eq1_prune_upper_alpha = 1;
            // float eq1_prune_lower_alpha = 0;
            std::pair<float, float> tmp_prune_range_1;
            if (diff1 > 0 && diff2 > 0) {
              // equation 1 holds on when alpha < = diff2 / diff1
              tmp_prune_range_1 = std::make_pair(0.0f, std::min(diff2 / diff1, 1.0f));
              // eq1_prune_lower_alpha = diff2 / diff1 ;
            } else if (diff1 < 0 && diff2 < 0) {
              // equation 1 holds on when alpha > = diff2 / diff1
              // eq1_prune_upper_alpha = diff2 / diff1 ;
              tmp_prune_range_1 = std::make_pair(std::min(diff2 / diff1, 1.0f), 1.0f);
            } else if (diff1 < 0 && diff2 > 0) {
              tmp_prune_range_1 = {0.0f, 1.0f};
              // the equation hold forever
            } else if (diff1 > 0 && diff2 < 0) {
              // equation never hold
              // break;
              tmp_prune_range_1 = {0.0f, 0.0f};
            }
            // now for equation 2
            float diff3 = xq_e_dist - cur_emb_dist + cur_geo_dist - xq_s_dist;
            float diff4 = cur_geo_dist - xq_s_dist;
            /*
            similar to previous
            */
            // when alpha >= eq1_prune_upper_alpha and alpha <= eq1_prune_lower_alpha, the equation holds on
            // float eq2_prune_upper_alpha = 1;
            // float eq2_prune_lower_alpha = 0;
            std::pair<float, float> tmp_prune_range_2;
            if (diff3 > 0 && diff4 > 0) {
              // equation 2 holds on when alpha < = diff4 / diff3
              // eq2_prune_upper_alpha = diff4 / diff3;
              // tmp_prune_range.second = std::min(tmp_prune_range.second, diff4 / diff3);
              tmp_prune_range_2 = std::make_pair(0.0f, std::min(1.0f, diff4 / diff3));
            } else if (diff3 < 0 && diff4 < 0) {
              // equation 2 holds on when alpha > = diff4 / diff3
              // eq2_prune_lower_alpha = diff4 / diff3;
              // tmp_prune_range.first = std::max(tmp_prune_range.first, diff4 / diff3);
              tmp_prune_range_2 = std::make_pair(std::min(diff4 / diff3, 1.0f), 1.0f);
            } else if (diff3 < 0 && diff4 > 0) {
              // the equation hold forever
              // then we do not change the previous range
              tmp_prune_range_2 = {0.0f, 1.0f};
            } else if (diff3 > 0 && diff4 < 0) {
              // equation never hold
              // break;
              tmp_prune_range_2 = {0.0f, 0.0f};
            }

            std::pair<float, float> tmp_prune_range;
            tmp_prune_range.first = std::max(tmp_prune_range_1.first, tmp_prune_range_2.first);
            tmp_prune_range.second = std::min(tmp_prune_range_1.second, tmp_prune_range_2.second);

            if (tmp_prune_range.second > tmp_prune_range.first) {
              // now we consider whether this range is useful range, that is (second > first)
              // now we check its intersection range with shared_use_range
              intersection(picked_use_range, tmp_prune_range, prune_range);
            } else {
              continue;
              // this range is not useful, so this edge will not be pruned by this selected edge
            }
          }
          prune_range = mergeIntervals(prune_range);
          std::vector<std::pair<float, float>> after_pruned_use_range;
          get_use_range(prune_range, after_pruned_use_range);
          float threshold = 0.1;
          float use_size = 0;
          for (int j = 0; j < after_pruned_use_range.size(); j++) {
            use_size = use_size + after_pruned_use_range[j].second - after_pruned_use_range[j].first;
          }
          if (use_size >= threshold) {
            tempres_picked.push_back(DEGNeighbor(candidate[i].id, candidate[i].distance1, candidate[i].distance2,
                                                 after_pruned_use_range, visited_layer));
          }
        }
        picked.insert(picked.end(), tempres_picked.begin(), tempres_picked.end());
      } else {
        for (int i = 0; i < candidate.size(); i++) {
          // 这里先初始化useful range 根据斜率算出来
          std::vector<std::pair<float, float>> prune_range;
          float cur_geo_dist = candidate[i].distance2;  // s_pq
          float cur_emb_dist = candidate[i].distance1;  // e_pq

          for (size_t j = 0; j < picked.size(); j++) {
            float exist_e_dist = picked[j].emb_distance_;  // e_xp

            float exist_s_dist = picked[j].geo_distance_;  // s_xp
            if (!judge_dominate(std::make_pair(exist_s_dist, exist_e_dist),
                                std::make_pair(cur_geo_dist, cur_emb_dist))) {
              continue;
            }
            const std::vector<std::pair<float, float>> &picked_use_range = picked[j].available_range;
            // we want to find out if this edge can prune the candidate within its picked_avaiable_range
            float xq_e_dist =
                _distance->compare(_data1 + _aligned_dim1 * (size_t) picked[j].id_,
                                   _data1 + _aligned_dim1 * (size_t) candidate[i].id, (unsigned) _aligned_dim1);
            // E(x,q)

            float xq_s_dist =
                _distance->compare(_data2 + _aligned_dim2 * (size_t) picked[j].id_,
                                   _data2 + _aligned_dim2 * (size_t) candidate[i].id, (unsigned) _aligned_dim2);
            // S(x,q)
            // float xq_e_dist = index->get_E_Dist()->compare(
            //     index->getBaseEmbData() + (size_t)picked[j].id_ * index->getBaseEmbDim(),
            //     index->getBaseEmbData() + (size_t)candidate[i].id_ * index->getBaseEmbDim(),
            //     index->getBaseEmbDim());

            // float xq_s_dist = index->get_S_Dist()->compare(
            //     index->getBaseLocData() + (size_t)picked[j].id_ * index->getBaseLocDim(),
            //     index->getBaseLocData() + (size_t)candidate[i].id_ * index->getBaseLocDim(),
            //     index->getBaseLocDim());

            // alpha * (E(p,x) - S(p,x) - E(p,q) + S(p,q)) <= S(p,q) - S(p,x)
            // alpha * (E(q,x) - S(q,x) - E(p,q) + S(p,q)) <= S(p,q) - S(q,x)
            // if alpha holds on for the two equation at the same time, the edge will be pruned
            // now for equation 1
            float diff1 = exist_e_dist - cur_emb_dist + cur_geo_dist - exist_s_dist;
            float diff2 = cur_geo_dist - exist_s_dist;
            /*
            diff1 > 0 && diff2 > 0
            equation 1 holds on when alpha < = diff2 / diff1
            diff1 < 0 && diff2 < 0
            equation 1 holds on when alpha > = diff2 / diff1
            diff1 < 0 && diff2 > 0
            the equation hold forever
            diff1 > 0 && diff2 < 0
            equation never hold which means this edge will not be pruned by this strategy
            */
            // float eq1_prune_upper_alpha = 1;
            // float eq1_prune_lower_alpha = 0;
            std::pair<float, float> tmp_prune_range_1;
            if (diff1 > 0 && diff2 > 0) {
              // equation 1 holds on when alpha < = diff2 / diff1
              tmp_prune_range_1 = std::make_pair(0.0f, std::min(diff2 / diff1, 1.0f));
              // eq1_prune_lower_alpha = diff2 / diff1 ;
            } else if (diff1 < 0 && diff2 < 0) {
              // equation 1 holds on when alpha > = diff2 / diff1
              // eq1_prune_upper_alpha = diff2 / diff1 ;
              tmp_prune_range_1 = std::make_pair(std::min(diff2 / diff1, 1.0f), 1.0f);
            } else if (diff1 < 0 && diff2 > 0) {
              tmp_prune_range_1 = {0.0f, 1.0f};
              // the equation hold forever
            } else if (diff1 > 0 && diff2 < 0) {
              // equation never hold
              // break;
              tmp_prune_range_1 = {0.0f, 0.0f};
            }
            // now for equation 2
            float diff3 = xq_e_dist - cur_emb_dist + cur_geo_dist - xq_s_dist;
            float diff4 = cur_geo_dist - xq_s_dist;
            /*
            similar to previous
            */
            // when alpha >= eq1_prune_upper_alpha and alpha <= eq1_prune_lower_alpha, the equation holds on
            // float eq2_prune_upper_alpha = 1;
            // float eq2_prune_lower_alpha = 0;
            std::pair<float, float> tmp_prune_range_2;
            if (diff3 > 0 && diff4 > 0) {
              // equation 2 holds on when alpha < = diff4 / diff3
              // eq2_prune_upper_alpha = diff4 / diff3;
              // tmp_prune_range.second = std::min(tmp_prune_range.second, diff4 / diff3);
              tmp_prune_range_2 = std::make_pair(0.0f, std::min(1.0f, diff4 / diff3));
            } else if (diff3 < 0 && diff4 < 0) {
              // equation 2 holds on when alpha > = diff4 / diff3
              // eq2_prune_lower_alpha = diff4 / diff3;
              // tmp_prune_range.first = std::max(tmp_prune_range.first, diff4 / diff3);
              tmp_prune_range_2 = std::make_pair(std::min(diff4 / diff3, 1.0f), 1.0f);
            } else if (diff3 < 0 && diff4 > 0) {
              // the equation hold forever
              // then we do not change the previous range
              tmp_prune_range_2 = {0.0f, 1.0f};
            } else if (diff3 > 0 && diff4 < 0) {
              // equation never hold
              // break;
              tmp_prune_range_2 = {0.0f, 0.0f};
            }

            std::pair<float, float> tmp_prune_range;
            tmp_prune_range.first = std::max(tmp_prune_range_1.first, tmp_prune_range_2.first);
            tmp_prune_range.second = std::min(tmp_prune_range_1.second, tmp_prune_range_2.second);

            if (tmp_prune_range.second > tmp_prune_range.first) {
              // now we consider whether this range is useful range, that is (second > first)
              // now we check its intersection range with shared_use_range
              intersection(picked_use_range, tmp_prune_range, prune_range);
            } else {
              continue;
              // this range is not useful, so this edge will not be pruned by this selected edge
            }
          }

          prune_range = mergeIntervals(prune_range);
          std::vector<std::pair<float, float>> after_pruned_use_range;
          get_use_range(prune_range, after_pruned_use_range);
          float threshold = 0.1;
          float use_size = 0;
          for (int j = 0; j < after_pruned_use_range.size(); j++) {
            use_size = use_size + after_pruned_use_range[j].second - after_pruned_use_range[j].first;
          }
          if (use_size >= threshold) {
            picked.push_back(DEGNeighbor(candidate[i].id, candidate[i].distance1, candidate[i].distance2,
                                         after_pruned_use_range, visited_layer));
          }
        }
      }
      visited_layer++;
      if (picked.size() >= range)
        break;
    }
    if (picked.size() > range)
      picked.resize(range);
    pruned_list.swap(picked);
  }
  void intersection(const std::vector<std::pair<float, float>> &picked_available_range,
                                          const std::pair<float, float> &use_range,
                                          std::vector<std::pair<float, float>> &shared_use_range) {
    // 遍历picked_available_range中的每个范围
    // std::sort(picked_available_range.begin(), picked_available_range.end());
    for (const auto &range : picked_available_range) {
      // 计算交集的下限和上限
      if (range.second <= use_range.first) {
        continue;
      } else if (range.first >= use_range.second) {
        break;
      } else {
        float lower_bound = std::max(range.first, use_range.first);
        float upper_bound = std::min(range.second, use_range.second);
        // 检查交集是否有效（下限小于等于上限）
        if (lower_bound < upper_bound) {
          // 如果交集有效，添加到shared_use_range中
          shared_use_range.push_back({lower_bound, upper_bound});
        }
      }
    }
  }

  void get_use_range(std::vector<std::pair<float, float>> &prune_range,
                                           std::vector<std::pair<float, float>> &after_pruned_use_range) {
    if (prune_range.empty()) {
      after_pruned_use_range.push_back(std::make_pair(0.0f, 1.0f));
      return;
    }
    if (prune_range[0].first > 0) {
      float gap = prune_range[0].first - 0;
      if (gap >= 0.01) {
        after_pruned_use_range.push_back(std::make_pair(0, prune_range[0].first));
      }
    }
    int iter;
    for (iter = 0; iter < prune_range.size() - 1; iter++) {
      float gap = prune_range[iter + 1].first - prune_range[iter].second;
      if (gap >= 0.01) {
        after_pruned_use_range.push_back(std::make_pair(prune_range[iter].second, prune_range[iter + 1].first));
      }
    }
    if (prune_range[iter].second < 1) {
      float gap = 1 - prune_range[iter].second;
      if (gap >= 0.01) {
        after_pruned_use_range.push_back(std::make_pair(prune_range[iter].second, 1));
      }
    }
    return;
  };

  std::vector<std::pair<float, float>> get_use_range(
      std::vector<std::pair<float, float>> &intervals) {
    std::vector<std::pair<float, float>> use_range;
    if (intervals.empty()) {
      use_range.push_back(std::make_pair(0, 1));
      return use_range;
    }
    if (intervals[0].first > 0 && intervals[0].first >= 0.01) {
      use_range.push_back(std::make_pair(0, intervals[0].first));
    }
    for (size_t i = 0; i < intervals.size() - 1; i++) {
      float gap = intervals[i + 1].first - intervals[i].second;
      if (gap >= 0.01) {
        use_range.push_back(std::make_pair(intervals[i].second, intervals[i + 1].first));
      }
    }
    if (intervals[intervals.size() - 1].second < 1 && (1 - intervals[intervals.size() - 1].second) >= 0.01) {
      use_range.push_back(std::make_pair(intervals[intervals.size() - 1].second, 1));
    }
    return use_range;
  };

  std::vector<std::pair<float, float>> mergeIntervals(
      std::vector<std::pair<float, float>> &intervals) {
    if (intervals.empty())
      return {};

    // 先对区间按照起始值进行排序
    std::sort(intervals.begin(), intervals.end());

    std::vector<std::pair<float, float>> merged;

    merged.push_back(intervals[0]);

    for (size_t i = 1; i < intervals.size() - 1; i++) {
      if (intervals[i].first <= merged.back().second) {
        // 如果当前区间的起始值小于等于前一个区间的终止值，则合并这两个区间
        merged.back().second = std::max(merged.back().second, intervals[i].second);
      } else {
        // 否则，当前区间与前一个区间不相交，将其添加到结果中
        merged.push_back(intervals[i]);
      }
    }
    return merged;
  }

};  // namespace pipeann
