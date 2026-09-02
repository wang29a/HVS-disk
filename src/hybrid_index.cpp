
#include <algorithm>
#include <boost/dynamic_bitset.hpp>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <thread>
#include <omp.h>
#include <random>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <unordered_set>
#include "log.h"
#include "neighbor.h"
#include "tsl/robin_set.h"
#include <unordered_map>

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>

#include "index.h"
#include "parameters.h"
#include "timer.h"
#include "utils.h"
#include "query_buf.h"
#include "v2/lock_table.h"

// only L2 implemented. Need to implement inner product search

namespace pipeann {

  // Initialize an index with metric m, load the data of type T with filename
  // (bin), and initialize max_points
  template<typename T, typename TagT>

  HybridIndex<T, TagT>::HybridIndex(Metric m, const size_t dim1, const size_t dim2, const size_t max_points,
                                    const bool dynamic_index, const bool save_index_in_one_file, const bool enable_tags)
      : _dist_metric(m), _dim1(dim1), _dim2(dim2), _max_points(max_points), _save_as_one_file(save_index_in_one_file),
        _dynamic_index(dynamic_index), _enable_tags(enable_tags) {
    if (dynamic_index && !enable_tags) {
      LOG(ERROR) << "WARNING: Dynamic Indices must have tags enabled. Auto-enabling.";
      _enable_tags = true;
    }
    // data is stored to _nd * aligned_dim matrix with necessary
    // zero-padding
    _aligned_dim1 = ROUND_UP(_dim1, 8);
    _aligned_dim2 = ROUND_UP(_dim2, 8);

    if (dynamic_index)
      _num_frozen_pts = 1;

    if (_max_points == 0) {
      _max_points = 1;
    }

    alloc_aligned(((void **) &_data1), (_max_points + 1) * _aligned_dim1 * sizeof(T), 8 * sizeof(T));
    alloc_aligned(((void **) &_data2), (_max_points + 1) * _aligned_dim2 * sizeof(T), 8 * sizeof(T));
    // std::memset(_data, 0, (_max_points + 1) * _aligned_dim * sizeof(T));

    _ep = (unsigned) _max_points;

    _final_graph.reserve(_max_points + _num_frozen_pts);
    _final_graph.resize(_max_points + _num_frozen_pts);

    for (size_t i = 0; i < _max_points + _num_frozen_pts; i++)
      _final_graph[i].clear();

    constexpr uint64_t kLockTableEntries = 131072;  // ~1MB lock table.
    this->_locks = new v2::LockTable(kLockTableEntries);
    LOG(INFO) << "Getting distance function for metric: " << (m == pipeann::Metric::COSINE ? "cosine" : "l2");
    this->_distance = get_distance_function<T>(m);
    _width = 0;
  }

  template<typename T, typename TagT>
  HybridIndex<T, TagT>::~HybridIndex() {
    delete this->_distance;
    delete this->_locks;
    aligned_free(_data1);
    aligned_free(_data2);
  }

  // template<typename T, typename TagT>
  // _u64 HybridIndex<T, TagT>::save_data(std::string data_file, size_t offset) {
  //   return save_data_in_base_dimensions(data_file, _data, _nd + _num_frozen_pts, _dim, _aligned_dim, offset);
  // }

  // save the graph index on a file as an adjacency list. For each point,
  // first store the number of neighbors, and then the neighbor list (each as
  // 4 byte unsigned)
  template<typename T, typename TagT>
  _u64 HybridIndex<T, TagT>::save_graph(std::string graph_file, size_t offset) {
    std::ofstream out;
    open_file_to_write(out, graph_file);

    out.seekp(offset, out.beg);
    _u64 index_size = 24;
    _u32 max_degree = 0;
    out.write((char *) &index_size, sizeof(uint64_t));
    out.write((char *) &_width, sizeof(unsigned));
    unsigned ep_u32 = _ep;
    out.write((char *) &ep_u32, sizeof(unsigned));
    out.write((char *) &_num_frozen_pts, sizeof(_u64));
    for (unsigned i = 0; i < _nd + _num_frozen_pts; i++) {
      unsigned GK = (unsigned) _final_graph[i].size();
      out.write((char *) &GK, sizeof(unsigned));
      for (unsigned k = 0; k < GK; k++) {
        DEGNeighbor &neighbor = _final_graph[i][k];
        unsigned neighbor_id = neighbor.id_;
        float dist1 = neighbor.emb_distance_;
        float dist2 = neighbor.geo_distance_;
        out.write((char *) &neighbor_id, sizeof(unsigned));
        out.write((char *) &dist1, sizeof(float));
        out.write((char *) &dist2, sizeof(float));
        // std::vector<std::pair<float, float>> &use_range = neighbor.available_range;
        // unsigned range_size = use_range.size();
        // out.write((char *) &range_size, sizeof(unsigned));
        // for (unsigned t = 0; t < range_size; t++) {
        //   int8_t x = static_cast<int8_t>(use_range[t].first * 100);
        //   int8_t y = static_cast<int8_t>(use_range[t].second * 100);
        //   out.write((char *) &x, sizeof(int8_t));
        //   out.write((char *) &y, sizeof(int8_t));
        // }
      }
      max_degree = _final_graph[i].size() > max_degree ? (_u32) _final_graph[i].size() : max_degree;
      index_size += (_u64) (sizeof(unsigned) * (GK + 1));
    }
    out.seekp(offset, out.beg);
    out.write((char *) &index_size, sizeof(uint64_t));
    out.write((char *) &max_degree, sizeof(_u32));
    out.close();
    return index_size;  // number of bytes written
  }

  template<typename T, typename TagT>
  _u64 HybridIndex<T, TagT>::save_graph_single(std::string graph_file, size_t offset) {
    std::ofstream out;
    open_file_to_write(out, graph_file);

    out.seekp(offset, out.beg);
    _u64 index_size = 24;
    _u32 max_nbr = 0;
    _u32 max_alpha_range = 0;
    out.write((char *) &index_size, sizeof(uint64_t));
    out.write((char *) &max_nbr, sizeof(unsigned));
    out.write((char *) &max_alpha_range, sizeof(unsigned));
    for (unsigned i = 0; i < _nd + _num_frozen_pts; i++) {
      unsigned GK = (unsigned) _final_graph[i].size();
      max_nbr = std::max(GK, max_nbr);
      out.write((char *) &GK, sizeof(unsigned));
      for (unsigned k = 0; k < GK; k++) {
        DEGNeighbor &neighbor = _final_graph[i][k];
        unsigned neighbor_id = neighbor.id_;
        out.write((char *) &neighbor_id, sizeof(unsigned));
        std::vector<std::pair<float, float>> use_range = neighbor.available_range;
        unsigned range_size = use_range.size();
        out.write((char *) &range_size, sizeof(unsigned));
        max_alpha_range = std::max(range_size, max_alpha_range);
        for (unsigned t = 0; t < range_size; t++) {
          int8_t x = static_cast<int8_t>(use_range[t].first * 100);
          int8_t y = static_cast<int8_t>(use_range[t].second * 100);
          out.write((char *) &x, sizeof(int8_t));
          out.write((char *) &y, sizeof(int8_t));
        }
      }
    }
    out.seekp(offset, out.beg);
    index_size = _u64(_nd);
    out.write((char *) &index_size, sizeof(uint64_t));
    out.write((char *) &max_nbr, sizeof(unsigned));
    out.write((char *) &max_alpha_range, sizeof(unsigned));
    out.close();
    return index_size;  // number of bytes written
  }

  template<typename T, typename TagT>
  void HybridIndex<T, TagT>::save(const char *filename) {
    // first check if no thread is inserting
    auto start = std::chrono::high_resolution_clock::now();
    std::unique_lock<std::shared_timed_mutex> lock(_update_lock);
    _change_lock.lock();

    // compact_data();
    // compact_frozen_point();
    if (!_save_as_one_file) {
      std::string graph_file = std::string(filename);
      std::string tags_file = std::string(filename) + ".tags";
      std::string data_file = std::string(filename) + ".data";
      std::string delete_list_file = std::string(filename) + ".del";

      // Because the save_* functions use append mode, ensure that
      // the files are deleted before save. Ideally, we should check
      // the error code for delete_file, but will ignore now because
      // delete should succeed if save will succeed.
      delete_file(graph_file);
      save_graph(graph_file);
      // delete_file(data_file);
      // save_data(data_file);
      // delete_file(tags_file);
      // save_tags(tags_file);
      // delete_file(delete_list_file);
      // save_delete_list(delete_list_file);
    } else {
      delete_file(filename);
      std::vector<size_t> cumul_bytes(5, 0);
      cumul_bytes[0] = METADATA_SIZE;
      cumul_bytes[1] = cumul_bytes[0] + save_graph(std::string(filename), cumul_bytes[0]);
      // cumul_bytes[2] = cumul_bytes[1] + save_data(std::string(filename), cumul_bytes[1]);
      // cumul_bytes[3] = cumul_bytes[2] + save_tags(std::string(filename), cumul_bytes[2]);
      // cumul_bytes[4] = cumul_bytes[3] + save_delete_list(filename, cumul_bytes[3]);
      pipeann::save_bin<_u64>(filename, cumul_bytes.data(), cumul_bytes.size(), 1, 0);

      LOG(INFO) << "Saved index as one file to " << filename << " of size " << cumul_bytes[cumul_bytes.size() - 1]
                << "B.";
    }

    // reposition_frozen_point_to_end();

    _change_lock.unlock();
    auto stop = std::chrono::high_resolution_clock::now();
    auto timespan = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
    LOG(INFO) << "Time taken for save: " << timespan.count() << "s.";
  }

  template<typename T, typename TagT>
  void HybridIndex<T, TagT>::save_single(const char *filename) {
    // first check if no thread is inserting
    auto start = std::chrono::high_resolution_clock::now();
    std::unique_lock<std::shared_timed_mutex> lock(_update_lock);
    _change_lock.lock();

    // compact_data();
    // compact_frozen_point();
    if (!_save_as_one_file) {
      std::string graph_file = std::string(filename);
      std::string tags_file = std::string(filename) + ".tags";
      std::string data_file = std::string(filename) + ".data";
      std::string delete_list_file = std::string(filename) + ".del";

      // Because the save_* functions use append mode, ensure that
      // the files are deleted before save. Ideally, we should check
      // the error code for delete_file, but will ignore now because
      // delete should succeed if save will succeed.
      delete_file(graph_file);
      save_graph_single(graph_file);
      // delete_file(data_file);
      // save_data(data_file);
      // delete_file(tags_file);
      // save_tags(tags_file);
      // delete_file(delete_list_file);
      // save_delete_list(delete_list_file);
    } else {
      delete_file(filename);
      std::vector<size_t> cumul_bytes(5, 0);
      cumul_bytes[0] = METADATA_SIZE;
      cumul_bytes[1] = cumul_bytes[0] + save_graph(std::string(filename), cumul_bytes[0]);
      // cumul_bytes[2] = cumul_bytes[1] + save_data(std::string(filename), cumul_bytes[1]);
      // cumul_bytes[3] = cumul_bytes[2] + save_tags(std::string(filename), cumul_bytes[2]);
      // cumul_bytes[4] = cumul_bytes[3] + save_delete_list(filename, cumul_bytes[3]);
      pipeann::save_bin<_u64>(filename, cumul_bytes.data(), cumul_bytes.size(), 1, 0);

      LOG(INFO) << "Saved index as one file to " << filename << " of size " << cumul_bytes[cumul_bytes.size() - 1]
                << "B.";
    }
    _change_lock.unlock();
    auto stop = std::chrono::high_resolution_clock::now();
    auto timespan = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
    LOG(INFO) << "Time taken for save: " << timespan.count() << "s.";
  }

  void findSkyline(std::vector<NeighborH> &points, std::vector<NeighborH> &skyline,
                        std::vector<NeighborH> &remain_points) {
    float min_emb_dis = std::numeric_limits<float>::max();
    for (const auto &point : points) {
      if (point.distance1 < min_emb_dis) {
        skyline.push_back(point);
        min_emb_dis = point.distance1;
      } else {
        remain_points.emplace_back(point);
      }
    }
  };

  template<typename T, typename TagT>
  std::pair<uint32_t, uint32_t> HybridIndex<T, TagT>::iterate_to_fixed_point(
      const T *node_coords1, const T *node_coords2, const unsigned Lsize, const std::vector<unsigned> &init_ids,
      std::vector<NeighborH> &pool, tsl::robin_set<unsigned> &expanded_nodes_ids, std::vector<NeighborH> &best_L_nodes, std::shared_mutex &ids_rw_mutex,
      bool ret_frozen) {
    auto findSkyline = [](std::vector<NeighborH> &points, std::vector<NeighborH> &skyline,
                          std::vector<NeighborH> &remain_points) {
      float min_emb_dis = std::numeric_limits<float>::max();
      for (const auto &point : points) {
        if (point.distance1 < min_emb_dis) {
          skyline.push_back(point);
          min_emb_dis = point.distance1;
        } else {
          remain_points.emplace_back(point);
        }
      }
    };
    auto updateNeighbor = [Lsize, findSkyline](std::vector<NeighborH> &pool, int &nk) {
      std::vector<NeighborH> skyline_result;
      std::vector<NeighborH> remain_points;
      std::vector<NeighborH> candidate;
      candidate.swap(pool);
      int l = 0;
      int k = 0;
      sort(candidate.begin(), candidate.end());
      bool updated = true;
      while (pool.size() < Lsize && candidate.size() > 0) {
        findSkyline(candidate, skyline_result, remain_points);
        candidate.swap(remain_points);
        for (auto &point : skyline_result) {
          pool.emplace_back(point.id, point.distance1, point.distance2, point.flag, l);
          if (updated) {
            if (point.flag == true) {
              nk = k;
              updated = false;
            } else {
              nk++;
            }
          }
          k++;
        }
        skyline_result.clear();
        remain_points.clear();
        l++;
      }
    };
    auto init_queue = [findSkyline](std::vector<NeighborH> &insert_points)
    {
      std::vector<NeighborH> skyline_result;
      std::vector<NeighborH> remain_points;
      std::vector<NeighborH> pool;
      int l = 0;
      while (!insert_points.empty())
      {
        findSkyline(insert_points, skyline_result, remain_points);
        // findConvexHull(insert_points, skyline_result, remain_points);
        insert_points.swap(remain_points);
        for (auto &point : skyline_result)
        {
          pool.emplace_back(point.id, point.distance1, point.distance2, true, l);
        }
        skyline_result.clear();
        remain_points.clear();
        l++;
      }
      pool.swap(insert_points);
    };

    pool.reserve(Lsize);
    expanded_nodes_ids.reserve(Lsize);
    NeighborH nn;
    {
      std::shared_lock lock(ids_rw_mutex);
      for (auto id : init_ids) {
        assert(id < _nd + _num_frozen_pts);
        nn = NeighborH(
            id, _distance->compare(_data1 + _aligned_dim1 * (size_t) id, node_coords1, (unsigned) _aligned_dim1),
            _distance->compare(_data2 + _aligned_dim2 * (size_t) id, node_coords2, (unsigned) _aligned_dim2), true, 0);
        expanded_nodes_ids.insert(id);
        pool.emplace_back(nn);
      }
    }
    sort(pool.begin(), pool.end());
    init_queue(pool);

    size_t k = 0;
    int l = 0;
    while (k < pool.size()) {
      while (pool[k].layer == l) {
        if (pool[k].flag) {
          pool[k].flag = false;
          unsigned n = pool[k].id;
          std::vector<unsigned> des;
          {
            // v2::SparseReadLockGuard<uint64_t> guard(&_locks, n);
            v2::LockGuard guard(_locks->rdlock(n));
            for (unsigned m = 0; m < _final_graph[n].size(); m++) {
              if (_final_graph[n][m].id_ >= _max_points + _num_frozen_pts) {
                LOG(ERROR) << "Wrong id found: " << _final_graph[n][m].id_;
                crash();
              }
              des.emplace_back(_final_graph[n][m].id_);
            }
          }
          for (unsigned m = 0; m < des.size(); ++m) {
            unsigned id = des[m];
            if (expanded_nodes_ids.find(id) == expanded_nodes_ids.end()) {
              expanded_nodes_ids.insert(id);
              float dist1 =
                  _distance->compare(node_coords1, _data1 + _aligned_dim1 * (size_t) id, (unsigned) _aligned_dim1);
              float dist2 =
                  _distance->compare(node_coords2, _data2 + _aligned_dim2 * (size_t) id, (unsigned) _aligned_dim2);

              NeighborH nn(id, dist1, dist2, true, 1);
              pool.emplace_back(nn);
            }
          }
        }
        k++;
        if (k >= pool.size()) {
          break;
        }
      }
      int nk = 0;
      updateNeighbor(pool, nk);
      k = nk;
      if (k < pool.size()) {
        l = pool[k].layer;
      }
    }
    return std::make_pair(0, 0);
  }

  template<typename T, typename TagT>
  void HybridIndex<T, TagT>::get_expanded_nodes(const size_t node_id, const unsigned Lindex,
                                                std::vector<unsigned> init_ids,
                                                std::vector<NeighborH> &expanded_nodes_info,
                                                tsl::robin_set<unsigned> &expanded_nodes_ids,
                                                std::shared_mutex &ids_rw_mutex) {
    const T *node_coords1 = _data1 + _aligned_dim1 * node_id;
    const T *node_coords2 = _data2 + _aligned_dim2 * node_id;
    std::vector<NeighborH> best_L_nodes;

    if (init_ids.size() == 0)
      init_ids.emplace_back(_ep);

    iterate_to_fixed_point(node_coords1, node_coords2, Lindex, init_ids, expanded_nodes_info, expanded_nodes_ids,
                           best_L_nodes, ids_rw_mutex);
  }
  /* inter_insert():
   * This function tries to add reverse links from all the visited nodes to
   * the current node n.
   */
  template<typename T, typename TagT>
  void HybridIndex<T, TagT>::inter_insert(unsigned n, std::vector<DEGNeighbor> &pruned_list,
                                          const Parameters &parameter) {
    const auto range = parameter.Get<unsigned>("R");
    assert(n >= 0 && n < _nd + _num_frozen_pts);

    const auto &src_pool = pruned_list;

    assert(!src_pool.empty());

    for (auto &des : src_pool) {
      /* des.id is the id of the neighbors of n */
      assert(des >= 0 && des < _max_points + _num_frozen_pts);
      /* des_pool contains the neighbors of the neighbors of n */

      auto &des_pool = _final_graph[des.id_];
      std::vector<NeighborH> copy_of_neighbors;
      bool prune_needed = false;
      {
        // v2::SparseWriteLockGuard<uint64_t> guard(&_locks, des);
        v2::LockGuard guard(_locks->wrlock(des.id_));
        if (std::find(des_pool.begin(), des_pool.end(), n) == des_pool.end()) {
          if (des_pool.size() < (_u64) (range)) {
            des_pool.emplace_back(n, des.emb_distance_, des.geo_distance_, des.available_range, des.layer_);
            prune_needed = false;
          } else {
            for (auto &n : des_pool) {
              copy_of_neighbors.emplace_back(n.id_, n.emb_distance_, n.geo_distance_, false, n.layer_);
            }
            copy_of_neighbors.emplace_back(n, des.emb_distance_, des.geo_distance_, false, des.layer_);
            prune_needed = true;
          }
        }
      }  // des lock is released by this point

      if (prune_needed) {
        // copy_of_neighbors.push_back(n);
        tsl::robin_set<unsigned> dummy_visited(0);
        std::vector<NeighborH> dummy_pool(0);

        size_t reserveSize = (size_t) (range);
        dummy_visited.reserve(reserveSize);
        dummy_pool.reserve(reserveSize);

        for (auto cur_nbr : copy_of_neighbors) {
          if (dummy_visited.find(cur_nbr.id) == dummy_visited.end() && cur_nbr.id != des.id_) {
            // float dist1 = _distance->compare(_data1 + _aligned_dim1 * (size_t) des.id,
            //                                 _data1 + _aligned_dim1 * (size_t) cur_nbr.id, (unsigned) _aligned_dim1);
            // float dist2 = _distance->compare(_data2 + _aligned_dim2 * (size_t) des.id,
            //                                 _data2 + _aligned_dim2 * (size_t) cur_nbr.id, (unsigned) _aligned_dim2);
            dummy_pool.emplace_back(NeighborH(cur_nbr.id, cur_nbr.distance1, cur_nbr.distance2, true, 0));
            dummy_visited.insert(cur_nbr.id);
          }
        }
        std::vector<DEGNeighbor> new_out_neighbors;
        // prune_neighbors(dummy_pool, parameter, new_out_neighbors);
        prune_neighbors_no_opt(dummy_pool, parameter, new_out_neighbors);
        {
          // v2::SparseWriteLockGuard<uint64_t> guard(&_locks, des);
          v2::LockGuard guard(_locks->wrlock(des.id_));
          _final_graph[des.id_].assign(new_out_neighbors.begin(), new_out_neighbors.end());
        }
      }
    }
  }

  template<typename T, typename TagT>
  void HybridIndex<T, TagT>::UpdateEnterpointSet(std::vector<NeighborH> &enterpoints_skyeline, std::vector<uint32_t> &enterpoints,
                           T* emb_center, T* loc_center, uint32_t id, std::shared_mutex &ids_rw_mutex) {
    float e_d = _distance->compare(emb_center, _data1 + _aligned_dim1 * (size_t) id, (unsigned) _aligned_dim1);
    float s_d = _distance->compare(loc_center, _data2 + _aligned_dim2 * (size_t) id, (unsigned) _aligned_dim2);

    {
      std::unique_lock lock(ids_rw_mutex);
      enterpoints_skyeline.push_back(NeighborH(id, e_d, s_d, true, 0));
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
  }
  // one-pass graph building.
  template<typename T, typename TagT>
  void HybridIndex<T, TagT>::link(Parameters &parameters) {
    unsigned num_threads = parameters.Get<unsigned>("num_threads");
    _saturate_graph = parameters.Get<bool>("saturate_graph");
    unsigned L = parameters.Get<unsigned>("L");  // Search list size
    const unsigned range = parameters.Get<unsigned>("R");

    LOG(INFO) << "Parameters: " << "L: " << L << ", R: " << range
              << ", saturate_graph: " << (_saturate_graph ? "true" : "false") << ", num_threads: " << num_threads
              << ", alpha: " << parameters.Get<float>("alpha");
    if (num_threads != 0)
      omp_set_num_threads(num_threads);

#pragma omp parallel
#pragma omp single
    LOG(INFO) << "num_threads: " << omp_get_num_threads();
    int64_t n_vecs_to_visit = _nd + _num_frozen_pts;
    _ep = 0;

    // std::vector<unsigned> init_ids;
    std::shared_mutex ids_rw_mutex;
    _init_ids.emplace_back(_ep);

    pipeann::Timer link_timer;
    std::cerr << "ep:" << _ep << std::endl;
    std::string index_prefix_path;
    try {
      index_prefix_path = parameters.Get<std::string>("index_prefix_path");
    } catch (const std::exception &e) {
    }
    uint64_t num_points = n_vecs_to_visit;
    std::cerr << "num_points: " << num_points << std::endl;
    std::cerr << "aligned_dim: " << _aligned_dim1 << std::endl;
    std::cerr << "index_prefix_path: " << index_prefix_path << std::endl;


    static constexpr uint32_t kInvalidID = std::numeric_limits<uint32_t>::max();
    v2::SparseLockTable<uint64_t> idx_lock_table;
    uint32_t K = 3;
    auto emb_center = std::make_unique<T[]>(_aligned_dim1);
    auto loc_center = std::make_unique<T[]>(_aligned_dim2);
    _u32 _nd = n_vecs_to_visit;

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
    // std::vector<uint32_t> enterpoints;
    // enterpoints.emplace_back(0);
#pragma omp parallel for schedule(dynamic)
    for (int64_t node = 1; node < n_vecs_to_visit; node++) {
      // search.
      std::vector<NeighborH> pool;
      tsl::robin_set<unsigned> visited;
      pool.reserve(2 * L);
      visited.reserve(2 * L);
      // 在调用 get_expanded_nodes 之前，加锁拷贝
      std::vector<unsigned> local_init_ids;
      {
          std::shared_lock lock(ids_rw_mutex);
          local_init_ids = _init_ids;
      }
      get_expanded_nodes(node, L, local_init_ids, pool, visited, ids_rw_mutex);
      // remove the node itself from pool.
      for (auto it = pool.begin(); it != pool.end();) {
        if (it->id == node) {
          it = pool.erase(it);
        } else {
          ++it;
        }
      }
      // prune neighbors.
      std::vector<DEGNeighbor> pruned_list;
      // prune_neighbors(pool, parameters, pruned_list);
      prune_neighbors_no_opt(pool, parameters, pruned_list);

      {
        // v2::SparseWriteLockGuard<uint64_t> guard(&_locks, node);
        v2::LockGuard guard(_locks->wrlock(node));
        _final_graph[node].assign(pruned_list.begin(), pruned_list.end());
      }

      inter_insert(node, pruned_list, parameters);

      if (node % 1000 == 0) {
        std::cerr << "\r" << (100.0 * node) / (n_vecs_to_visit) << "% of index build completed. " << _init_ids.size() << "  ";
      }
      UpdateEnterpointSet(enterpoints_skyline, _init_ids, emb_center.get(), loc_center.get(), node, ids_rw_mutex);
      // exit(0);
    }

    size_t size = n_vecs_to_visit;
    std::cerr << "K:" << K << std::endl;
    // exit(0);
    if (_nd > 0) {
      LOG(INFO) << "Starting final cleanup..";
    }
#pragma omp parallel for schedule(dynamic, 65536)
    for (_s64 node_ctr = 0; node_ctr < n_vecs_to_visit; node_ctr++) {
      auto node = node_ctr;
      if (_final_graph[node].size() > range) {
        tsl::robin_set<unsigned> dummy_visited(0);
        std::vector<NeighborH> dummy_pool(0);
        std::vector<DEGNeighbor> new_out_neighbors;

        for (auto &cur_nbr : _final_graph[node]) {
          auto cur_id = cur_nbr.id_;
          if (dummy_visited.find(cur_id) == dummy_visited.end() && cur_id != node) {
            // float dist1 = _distance->compare(_data1 + _aligned_dim1 * (size_t) node,
            //                                  _data1 + _aligned_dim1 * (size_t) cur_id, (unsigned) _aligned_dim1);
            // float dist2 = _distance->compare(_data2 + _aligned_dim2 * (size_t) node,
            //                                  _data2 + _aligned_dim2 * (size_t) cur_id, (unsigned) _aligned_dim2);
            dummy_pool.emplace_back(NeighborH(cur_id, cur_nbr.emb_distance_, cur_nbr.geo_distance_, true, 0));
            dummy_visited.insert(cur_id);
          }
        }
        // prune_neighbors(dummy_pool, parameters, new_out_neighbors);
        prune_neighbors_no_opt(dummy_pool, parameters, new_out_neighbors);

        _final_graph[node].clear();
        for (auto &nonbr : new_out_neighbors)
          _final_graph[node].emplace_back(nonbr);
      }
    }
    LOG(INFO) << "init ids size: " << _init_ids.size();
    // this->_init_ids;
    if (_nd > 0) {
      LOG(INFO) << "done. Link time: " << ((double) link_timer.elapsed() / (double) 1000000) << "s";
    }
  }

  template<typename T, typename TagT>
  void HybridIndex<T, TagT>::build(const char *filename1, const char *filename2, const size_t num_points_to_load,
                                   Parameters &parameters, const std::vector<TagT> &tags) {
    if (!file_exists(filename1)) {
      LOG(ERROR) << "Data file provided " << filename1 << " does not exist.";
      crash();
    }
    if (!file_exists(filename2)) {
      LOG(ERROR) << "Data file provided " << filename2 << " does not exist.";
      crash();
    }

    size_t file_num_points1, file_dim1;
    size_t file_num_points2, file_dim2;
    if (filename1 == nullptr || filename2 == nullptr) {
      LOG(INFO) << "Starting with an empty index.";
      _nd = 0;
    } else {
      pipeann::get_bin_metadata(filename1, file_num_points1, file_dim1);
      if (file_num_points1 > _max_points || num_points_to_load > file_num_points1) {
        LOG(ERROR) << "ERROR: Driver requests loading " << num_points_to_load << " points and file has "
                   << file_num_points1 << " points, but " << "index can support only " << _max_points
                   << " points as specified in constructor.";
        crash();
      }
      if (file_dim1 != _dim1) {
        LOG(ERROR) << "ERROR: Driver requests loading " << _dim1 << " dimension," << "but file has " << file_dim1
                   << " dimension.";
        crash();
      }
      pipeann::get_bin_metadata(filename2, file_num_points2, file_dim2);
      if (file_num_points2 > _max_points || num_points_to_load > file_num_points2) {
        LOG(ERROR) << "ERROR: Driver requests loading " << num_points_to_load << " points and file has "
                   << file_num_points2 << " points, but " << "index can support only " << _max_points
                   << " points as specified in constructor.";
        crash();
      }
      if (file_dim2 != _dim2) {
        LOG(ERROR) << "ERROR: Driver requests loading " << _dim2 << " dimension," << "but file has " << file_dim2
                   << " dimension.";
        crash();
      }

      copy_aligned_data_from_file<T>(std::string(filename1), _data1, file_num_points1, file_dim1, _aligned_dim1);
      copy_aligned_data_from_file<T>(std::string(filename2), _data2, file_num_points2, file_dim2, _aligned_dim2);

      LOG(INFO) << "Loading only first " << num_points_to_load << " from file.. ";
      _nd = num_points_to_load;
      if (_enable_tags && tags.size() != num_points_to_load) {
        LOG(ERROR) << "ERROR: Driver requests loading " << num_points_to_load << " points from file,"
                   << "but tags vector is of size " << tags.size() << ".";
        crash();
      }
      if (_enable_tags) {
        for (size_t i = 0; i < tags.size(); ++i) {
          _tag_to_location[tags[i]] = (unsigned) i;
          _location_to_tag[(unsigned) i] = tags[i];
        }
      }
    }

    // generate_frozen_point();
    link(parameters);  // Primary func for creating nsg graph

    size_t max = 0, min = 1 << 30, total = 0, cnt = 0;
    for (size_t i = 0; i < _nd; i++) {
      auto &pool = _final_graph[i];
      max = (std::max)(max, pool.size());
      min = (std::min)(min, pool.size());
      total += pool.size();
      if (pool.size() < 2)
        cnt++;
    }
    if (min > max)
      min = max;
    if (_nd > 0) {
      LOG(INFO) << "Index built with degree: max:" << max << "  avg:" << (float) total / (float) (_nd + _num_frozen_pts)
                << "  min:" << min << "  count(deg<2):" << cnt;
    }
    _width = (std::max)((unsigned) max, _width);
    _has_built = true;
  }

  template<typename T, typename TagT>
  void HybridIndex<T, TagT>::prune_neighbors_no_opt(std::vector<NeighborH> &pool,
                                            const Parameters &parameter, std::vector<DEGNeighbor> &pruned_list) {
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
              for (size_t j = 0; j < after_pruned_use_range.size(); j++)
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

  template<typename T, typename TagT>
  void HybridIndex<T, TagT>::prune_neighbors(std::vector<NeighborH> &pool,
                                             const Parameters &parameter, std::vector<DEGNeighbor> &pruned_list) {
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
                                                 _data1 + _aligned_dim1 * (size_t) (size_t) candidate[i].id,
                                                 (unsigned) _aligned_dim1);
            // E(x,q)

            float xq_s_dist = _distance->compare(_data2 + _aligned_dim2 * (size_t) id,
                                                 _data2 + _aligned_dim2 * (size_t) (size_t) candidate[i].id,
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
          for (size_t j = 0; j < after_pruned_use_range.size(); j++) {
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
        for (size_t i = 0; i < candidate.size(); i++) {
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
  template<typename T, typename TagT>
  void HybridIndex<T, TagT>::intersection(const std::vector<std::pair<float, float>> &picked_available_range,
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

  template<typename T, typename TagT>
  void HybridIndex<T, TagT>::get_use_range(std::vector<std::pair<float, float>> &prune_range,
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

  template<typename T, typename TagT>
  std::vector<std::pair<float, float>> HybridIndex<T, TagT>::get_use_range(
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

  template<typename T, typename TagT>
  std::vector<std::pair<float, float>> HybridIndex<T, TagT>::mergeIntervals(
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
  /*  Internals of the library */
  // EXPORTS
  template class HybridIndex<float, uint32_t>;
  template class HybridIndex<int8_t, uint32_t>;
  template class HybridIndex<uint8_t, uint32_t>;
}  // namespace pipeann
