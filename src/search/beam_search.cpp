#include "aligned_file_reader.h"
#include "libcuckoo/cuckoohash_map.hh"
#include "log.h"
#include "ssd_index.h"
#include <cstdlib>
#include <malloc.h>
#include <algorithm>
#include <filesystem>

#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include "timer.h"
#include "tsl/robin_map.h"
#include "utils.h"
#include "v2/page_cache.h"

#include <unistd.h>
#include <sys/syscall.h>
#include "linux_aligned_file_reader.h"

extern int global_sw;

namespace pipeann {
  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::do_beam_search(const T *query1, uint32_t mem_L, uint32_t l_search, const uint32_t beam_width,
                                         std::vector<Neighbor> &expanded_nodes_info,
                                         tsl::robin_map<uint32_t, T *> *coord_map, QueryStats *stats,
                                         tsl::robin_set<uint32_t> *exclude_nodes /* tags */, bool dyn_search_l,
                                         std::vector<uint64_t> *passthrough_page_ref,
                                         QueryBuffer<T> * passthrough_data
                                        ) {
    uint32_t original_l_search = l_search;
    auto diskSearchBegin = std::chrono::high_resolution_clock::now();

    QueryBuffer<T> *query_buf;
    if(passthrough_data != nullptr){
      query_buf = passthrough_data;
    } else {
      query_buf = pop_query_buf(query1);
    }
    void *ctx = reader->get_ctx();

    const T *query = query_buf->aligned_query_T;

    // reset query
    query_buf->reset();

    // pointers to buffers for data
    T *data_buf = query_buf->coord_scratch;
    _u64 &data_buf_idx = query_buf->coord_idx;
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);

    // sector scratch
    char *sector_scratch = query_buf->sector_scratch;
    _u64 &sector_scratch_idx = query_buf->sector_idx;

    // query <-> PQ chunk centers distances
    float *pq_dists = query_buf->aligned_pqtable_dist_scratch;
    pq_table.populate_chunk_distances(query, pq_dists);

    // query <-> neighbor list
    float *dist_scratch = query_buf->aligned_dist_scratch;
    _u8 *pq_coord_scratch = query_buf->aligned_pq_coord_scratch;

    // lambda to batch compute query<-> node distances in PQ space
    auto compute_dists = [this, pq_coord_scratch, pq_dists](const unsigned *ids, const _u64 n_ids, float *dists_out) {
      ::aggregate_coords(ids, n_ids, this->data.data(), this->n_chunks, pq_coord_scratch);
      ::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists, dists_out);
    };

    Timer query_timer, io_timer, cpu_timer;
    std::vector<Neighbor> retset;
    retset.resize(mem_L + 10 * l_search);
    tsl::robin_set<_u64> visited(4096);

    // re-naming `expanded_nodes_info` to not change rest of the code
    std::vector<Neighbor> &full_retset = expanded_nodes_info;
    full_retset.reserve(10 * l_search);
    _u32 best_medoid = medoids[0];

    unsigned cur_list_size = 0;
    auto compute_and_add_to_retset = [&](const unsigned *node_ids, const _u64 n_ids) {
      compute_dists(node_ids, n_ids, dist_scratch);
      for (_u64 i = 0; i < n_ids; ++i) {
        retset[cur_list_size].id = node_ids[i];
        retset[cur_list_size].distance = dist_scratch[i];
        retset[cur_list_size++].flag = true;
        visited.insert(node_ids[i]);
      }
    };

    if (mem_L) {
      std::vector<unsigned> mem_tags(mem_L);
      std::vector<float> mem_dists(mem_L);
      mem_index_->search_with_tags(query, mem_L, mem_L, mem_tags.data(), mem_dists.data());
      compute_and_add_to_retset(mem_tags.data(), std::min((unsigned) mem_L, (unsigned) l_search));
    } else {
      // Do not use optimized start point.
      compute_and_add_to_retset(&best_medoid, 1);
    }

    std::sort(retset.begin(), retset.begin() + cur_list_size);

    unsigned cmps = 0;
    unsigned hops = 0;
    unsigned num_ios = 0;
    unsigned k = 0;

    // cleared every iteration
    std::vector<unsigned> frontier;
    using fnhood_t = std::tuple<unsigned, unsigned, char *>;
    std::vector<fnhood_t> frontier_nhoods;
    std::vector<IORequest> frontier_read_reqs;
    std::vector<uint32_t> vec_rdlocks;

    std::vector<uint64_t> new_page_ref{};
    std::vector<uint64_t> &page_ref = passthrough_page_ref ? *passthrough_page_ref : new_page_ref;

    while (k < cur_list_size) {
      auto nk = cur_list_size;
      // clear iteration state
      frontier.clear();
      frontier_nhoods.clear();
      frontier_read_reqs.clear();
      vec_rdlocks.clear();
      sector_scratch_idx = 0;
      // find new beam
      // WAS: _u64 marker = k - 1;
      _u32 marker = k;
      _u32 num_seen = 0;
      while (marker < cur_list_size && frontier.size() < beam_width && num_seen < beam_width) {
        if (retset[marker].flag) {
          num_seen++;
          frontier.push_back(retset[marker].id);
          retset[marker].flag = false;
        }
        marker++;
      }

      // read nhoods of frontier ids
      std::vector<uint32_t> locked;
      if (!frontier.empty()) {
        if (stats != nullptr)
          stats->n_hops++;
        locked = this->lock_idx(idx_lock_table, kInvalidID, frontier, true);
        for (_u64 i = 0; i < frontier.size(); i++) {
          uint32_t id = frontier[i];
          uint32_t loc = this->id2loc(id);
          uint64_t offset = loc_sector_no(loc) * SECTOR_LEN;
          auto sector_buf = sector_scratch + sector_scratch_idx * size_per_io;
          fnhood_t fnhood = std::make_tuple(id, loc, sector_buf);
          sector_scratch_idx++;
          frontier_nhoods.push_back(fnhood);
          frontier_read_reqs.emplace_back(IORequest(offset, size_per_io, sector_buf, u_loc_offset(loc), max_node_len));
          if (stats != nullptr) {
            stats->n_4k++;
            stats->n_ios++;
          }
          num_ios++;
        }
        io_timer.reset();
#ifdef DIRECT_READ_CC
        reader->read(frontier_read_reqs, ctx);
#else
        reader->read_alloc(frontier_read_reqs, ctx, &page_ref);
#endif

        if (stats != nullptr) {
          stats->io_us += (double) io_timer.elapsed();
        }
        this->unlock_idx(idx_lock_table, locked);
      }

      for (auto &frontier_nhood : frontier_nhoods) {
        auto [id, loc, sector_buf] = frontier_nhood;
        char *node_disk_buf = offset_to_loc(sector_buf, loc);
        unsigned *node_buf = offset_to_node_nhood(node_disk_buf);
        _u64 nnbrs = (_u64) (*node_buf);
        T *node_fp_coords = offset_to_node_coords(node_disk_buf);
        assert(data_buf_idx < MAX_N_CMPS);

        T *node_fp_coords_copy = data_buf + (data_buf_idx * aligned_dim);
        data_buf_idx++;
        memcpy(node_fp_coords_copy, node_fp_coords, data_dim * sizeof(T));
        float cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy, (unsigned) aligned_dim);

        if (coord_map != nullptr) {
          coord_map->insert(std::make_pair(id, node_fp_coords_copy));
        }
        full_retset.push_back(Neighbor(id, cur_expanded_dist, true));

        unsigned *node_nbrs = (node_buf + 1);

        // compute node_nbrs <-> query dist in PQ space
        cpu_timer.reset();
        compute_dists(node_nbrs, nnbrs, dist_scratch);
        if (stats != nullptr) {
          stats->n_cmps += (double) nnbrs;
          stats->cpu_us += (double) cpu_timer.elapsed();
        }

        cpu_timer.reset();
        // process prefetch-ed nhood
        for (_u64 m = 0; m < nnbrs; ++m) {
          unsigned id = node_nbrs[m];
          if (unlikely(id > this->cur_id)) {
            LOG(ERROR) << "ID is larger than current ID, " << id << " vs " << this->cur_id;
            crash();
          }
          if (visited.find(id) != visited.end()) {
            continue;
          } else {
            visited.insert(id);
            cmps++;
            float dist = dist_scratch[m];
            if (stats != nullptr) {
              stats->n_cmps++;
            }
            if (dist >= retset[cur_list_size - 1].distance && (cur_list_size == l_search))
              continue;
            Neighbor nn(id, dist, true);
            // variable search_L for deleted nodes.
            // Return position in sorted list where nn inserted.

            auto r = InsertIntoPool(retset.data(), cur_list_size, nn);

            if (cur_list_size < l_search) {
              ++cur_list_size;
              if (unlikely(cur_list_size >= retset.size())) {
                retset.resize(2 * cur_list_size);
              }
            }

            if (r < nk)
              nk = r;  // nk logs the best position in the retset that was
                       // updated due to neighbors of n.
          }
        }

        if (dyn_search_l) {
          // TODO(gh): contention still exists in id2tag(x)
          // O(n), but it is not slow as L is typically smaller than 300.
          // l_search monotonically increases to handle deleted nodes.
          _u32 tot = 0, cur = 0;
          for (cur = 0; cur < cur_list_size; ++cur) {
            uint32_t tag = id2tag(retset[cur].id);
            if (exclude_nodes->find(tag) == exclude_nodes->end()) {
              ++tot;
              if (tot == original_l_search) {
                break;
              }
            }
          }
          // cur is the stopped index (cur + 1 is the length it should be)
          l_search = std::max(original_l_search, cur + 1);
        }

        if (stats != nullptr) {
          stats->cpu_us += (double) cpu_timer.elapsed();
        }
      }

      // update best inserted position
      //

      if (nk <= k)
        k = nk;  // k is the best position in retset updated in this round.
      else
        ++k;

      hops++;
      if (stats != nullptr && stats->n_current_used != 0) {
        auto diskSearchEnd = std::chrono::high_resolution_clock::now();
        double elapsedSeconds =
            std::chrono::duration_cast<std::chrono::milliseconds>(diskSearchEnd - diskSearchBegin).count();
        if (elapsedSeconds >= stats->n_current_used)
          break;
      }
    }
    // re-sort by distance
    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) { return left < right; });

    if (passthrough_page_ref == nullptr) {
      reader->deref(&page_ref, ctx);
    }

    
    if(passthrough_data == nullptr){
      push_query_buf(query_buf);
    }

    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::do_beam_search(const T *query1, const T *query2, const float alpha, uint32_t mem_L, uint32_t l_search, const uint32_t beam_width,
                                         std::vector<Neighbor> &expanded_nodes_info,
                                         tsl::robin_map<uint32_t, T *> *coord_map, QueryStats *stats,
                                         tsl::robin_set<uint32_t> *exclude_nodes /* tags */, bool dyn_search_l,
                                         std::vector<uint64_t> *passthrough_page_ref,
                                         QueryBuffer<T> * passthrough_data
                                        ) {
    uint32_t original_l_search = l_search;
    auto diskSearchBegin = std::chrono::high_resolution_clock::now();

    QueryBuffer<T> *query_buf;
    if(passthrough_data != nullptr){
      query_buf = passthrough_data;
    } else {
      query_buf = pop_query_buf(query1, query2);
    }
    void *ctx = reader->get_ctx();

    const T *query = query_buf->aligned_query_T;
    const T *query_2 = query_buf->aligned_queryl_T;

    // reset query
    query_buf->reset();

    // pointers to buffers for data
    T *data_buf = query_buf->coord_scratch;
    T *datal_buf = query_buf->coordl_scratch;
    _u64 &data_buf_idx = query_buf->coord_idx;
    _u64 &datal_buf_idx = query_buf->coordl_idx;
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);
    _mm_prefetch((char *) datal_buf, _MM_HINT_T1);

    // sector scratch
    char *sector_scratch = query_buf->sector_scratch;
    _u64 &sector_scratch_idx = query_buf->sector_idx;

    // query <-> PQ chunk centers distances
    float *pq_dists = query_buf->aligned_pqtable_dist_scratch;
    pq_table.populate_chunk_distances(query, pq_dists);
    float *pq_distsl = query_buf->aligned_pqtable_distl_scratch;
    pq_tablel.populate_chunk_distances(query_2, pq_distsl);

    // query <-> neighbor list
    float *dist_scratch = query_buf->aligned_dist_scratch;
    float *distl_scratch = query_buf->aligned_distl_scratch;
    _u8 *pq_coord_scratch = query_buf->aligned_pq_coord_scratch;
    _u8 *pq_coordl_scratch = query_buf->aligned_pq_coordl_scratch;

    // lambda to batch compute query<-> node distances in PQ space
    auto compute_dists = [this, pq_coord_scratch, pq_dists](const unsigned *ids, const _u64 n_ids, float *dists_out) {
      ::aggregate_coords(ids, n_ids, this->data.data(), this->n_chunks, pq_coord_scratch);
      ::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists, dists_out);
    };
    auto compute_distsl = [this, pq_coordl_scratch, pq_distsl](const unsigned *ids, const _u64 n_ids, float *dists_out) {
      ::aggregate_coords(ids, n_ids, this->datal.data(), this->n_chunks, pq_coordl_scratch);
      ::pq_dist_lookup(pq_coordl_scratch, n_ids, this->n_chunks, pq_distsl, dists_out);
    };

    Timer query_timer, io_timer, cpu_timer;
    std::vector<Neighbor> retset;
    retset.resize(mem_L + 10 * l_search);
    tsl::robin_set<_u64> visited(4096);

    // re-naming `expanded_nodes_info` to not change rest of the code
    std::vector<Neighbor> &full_retset = expanded_nodes_info;
    full_retset.reserve(10 * l_search);
    _u32 best_medoid = medoids[0];

    unsigned cur_list_size = 0;
    auto compute_and_add_to_retset = [&](const unsigned *node_ids, const _u64 n_ids) {
      compute_dists(node_ids, n_ids, dist_scratch);
      compute_distsl(node_ids, n_ids, distl_scratch);
      for (_u64 i = 0; i < n_ids; ++i) {
        retset[cur_list_size].id = node_ids[i];
        retset[cur_list_size].distance = alpha*dist_scratch[i]+ (1-alpha)*distl_scratch[i];
        // LOG(INFO) << node_ids[i] << " " << dist_scratch[i] << " " << distl_scratch[i];
        retset[cur_list_size].rev_id = node_ids[i];
        retset[cur_list_size++].flag = true;
        visited.insert(node_ids[i]);
      }
    };

    if (mem_L) {
      std::vector<unsigned> mem_tags(mem_L);
      std::vector<float> mem_dists(mem_L);
      mem_index_->search_with_tags(query, mem_L, mem_L, mem_tags.data(), mem_dists.data());
      compute_and_add_to_retset(mem_tags.data(), std::min((unsigned) mem_L, (unsigned) l_search));
    } else {
      // Do not use optimized start point.
      compute_and_add_to_retset(medoids, num_medoids);
    }

    std::sort(retset.begin(), retset.begin() + cur_list_size);

    unsigned cmps = 0;
    unsigned hops = 0;
    unsigned num_ios = 0;
    unsigned k = 0;

    // cleared every iteration
    std::vector<unsigned> frontier;
    using fnhood_t = std::tuple<unsigned, unsigned, char *>;
    std::vector<fnhood_t> frontier_nhoods;
    std::vector<IORequest> frontier_read_reqs;
    std::vector<uint32_t> vec_rdlocks;

    std::vector<uint64_t> new_page_ref{};
    std::vector<uint64_t> &page_ref = passthrough_page_ref ? *passthrough_page_ref : new_page_ref;

    while (k < cur_list_size) {
      auto nk = cur_list_size;
      // clear iteration state
      frontier.clear();
      frontier_nhoods.clear();
      frontier_read_reqs.clear();
      vec_rdlocks.clear();
      sector_scratch_idx = 0;
      // find new beam
      // WAS: _u64 marker = k - 1;
      _u32 marker = k;
      _u32 num_seen = 0;
      while (marker < cur_list_size && frontier.size() < beam_width && num_seen < beam_width) {
        if (retset[marker].flag) {
          num_seen++;
          frontier.push_back(retset[marker].id);
          retset[marker].flag = false;
          if (this->count_visited_nodes) {
            #pragma omp critical
            {
              auto &cnt = this->node_visit_counter[retset[marker].id].second;
              ++cnt;
              if (this->count_visited_nbrs) {
                unsigned r_id = retset[marker].rev_id;
                unsigned id = retset[marker].id;
                if (r_id != id) {
                  ++(this->nbrs_freq_counter_[r_id][id]);
                }
              }
            }
          }
        }
        marker++;
      }

      // read nhoods of frontier ids
      std::vector<uint32_t> locked;
      if (!frontier.empty()) {
        if (stats != nullptr)
          stats->n_hops++;
        locked = this->lock_idx(idx_lock_table, kInvalidID, frontier, true);
        sector_scratch_idx = 0;
        for (_u64 i = 0; i < frontier.size(); i++) {
          uint32_t id = frontier[i];
          uint32_t loc = this->id2loc(id);
          uint64_t offset = loc_sector_no(loc) * SECTOR_LEN;
          auto sector_buf = sector_scratch + sector_scratch_idx * size_per_io;
          fnhood_t fnhood = std::make_tuple(id, loc, sector_buf);
          sector_scratch_idx++;
          frontier_nhoods.push_back(fnhood);
          frontier_read_reqs.emplace_back(IORequest(offset, size_per_io, sector_buf, u_loc_offset(loc), max_node_len));
          if (stats != nullptr) {
            stats->n_4k++;
            stats->n_ios++;
          }
          num_ios++;
        }
        io_timer.reset();
#ifdef DIRECT_READ_CC
        reader->read(frontier_read_reqs, ctx);
#else
        reader->read_alloc(frontier_read_reqs, ctx, &page_ref);
#endif

        if (stats != nullptr) {
          stats->io_us += (double) io_timer.elapsed();
        }
        this->unlock_idx(idx_lock_table, locked);
      }

      data_buf_idx = 0;
      datal_buf_idx = 0;
      for (auto &frontier_nhood : frontier_nhoods) {
        auto [id, loc, sector_buf] = frontier_nhood;
        char *node_disk_buf = offset_to_loc(sector_buf, loc);
        unsigned *node_buf = offset_to_node_nhood(node_disk_buf);
        _u64 nnbrs = (_u64) (*node_buf);
        T *node_fp_coords = offset_to_node_coords(node_disk_buf);
        T *node_fp_coords2 = offset_to_node_coords2(node_disk_buf);
        assert(data_buf_idx < MAX_N_CMPS);

        T *node_fp_coords_copy1 = data_buf + (data_buf_idx * data_dim);
        data_buf_idx++;
        T *node_fp_coords_copy2 = datal_buf + (datal_buf_idx * datal_dim);
        datal_buf_idx++;
        memcpy(node_fp_coords_copy1, node_fp_coords, data_dim * sizeof(T));
        memcpy(node_fp_coords_copy2, node_fp_coords2, datal_dim * sizeof(T));
        float cur_expanded_dist1 = dist_cmp->compare(query, node_fp_coords_copy1, (unsigned) data_dim);
        float cur_expanded_dist2 = dist_cmp->compare(query_2, node_fp_coords_copy2, (unsigned) datal_dim);
        float dist = alpha*std::sqrt(cur_expanded_dist1) + (1-alpha)*std::sqrt(cur_expanded_dist2);

        if (coord_map != nullptr) {
          coord_map->insert(std::make_pair(id, node_fp_coords_copy1));
        }
        full_retset.push_back(Neighbor(id, dist, true));

        unsigned *node_nbrs = (node_buf + 1);
        unsigned nbors_cand_size = 0;

        uint32_t nbr_data_len = max_alpha_range_len*2+sizeof(uint32_t);
        for (_u64 m = 0; m < nnbrs; ++m) {
          uint32_t neighbor_id = *(uint32_t*)((char *)node_nbrs + m * nbr_data_len);
          if (unlikely(id > this->cur_id)) {
            LOG(ERROR) << "ID is larger than current ID, " << id << " vs " << this->cur_id;
            crash();
          }
          std::vector<std::pair<int8_t, int8_t>> use_range;
          const uint8_t* base_ptr = (const uint8_t*)node_nbrs + (m * nbr_data_len);
          const uint8_t* alpha_ptr = base_ptr + sizeof(uint32_t);
          for (size_t k = 0; k < max_alpha_range_len; k ++) {
            int8_t alpha1 = (int8_t)alpha_ptr[k * 2];
            int8_t alpha2 = (int8_t)alpha_ptr[k * 2 + 1];
            if(!(alpha1<=100) || !(alpha1>=0) || !(alpha2<=100) || !(alpha2>=0)){
              break;
            }
            if(alpha1 == 0 && alpha2 == 0) {
              break;
            }
            use_range.emplace_back(alpha1, alpha2);
          }
          bool search_flag = false;
          for (size_t i = 0; i < use_range.size(); i++) {
            if (alpha * 100 >= use_range[i].first && alpha * 100 <= use_range[i].second) {
              search_flag = true;
              break;
            }
            if (alpha * 100 < use_range[i].first) {
              break;
            }
            if (alpha * 100 > use_range[i].second) {
              continue;
            }
          }
          if (visited.find(neighbor_id) == visited.end() && search_flag) {
            node_nbrs[nbors_cand_size++] = neighbor_id;
            visited.insert(neighbor_id);
          }
        }
        // compute node_nbrs <-> query dist in PQ space
        if (nbors_cand_size) {
          cpu_timer.reset();
          compute_dists(node_nbrs, nbors_cand_size, dist_scratch);
          compute_distsl(node_nbrs, nbors_cand_size, distl_scratch);
          if (stats != nullptr) {
            stats->n_cmps += nbors_cand_size;
            stats->cpu_us += (double) cpu_timer.elapsed();
          }
          cpu_timer.reset();

          for (unsigned m = 0; m < nbors_cand_size; ++m) {
            const int nbor_id = node_nbrs[m];
            const float nbor_diste = dist_scratch[m];
            const float nbor_distl = distl_scratch[m];
            const float nbor_dist = alpha*nbor_diste + (1-alpha)*nbor_distl;
            if (stats != nullptr) {
              stats->n_cmps++;
            }
            if (nbor_dist >= retset[cur_list_size - 1].distance && (cur_list_size == l_search))
              continue;
            Neighbor nn(nbor_id, nbor_dist, true, id);
            // if(t)
            // std::cerr << nbor_id << "," << nbor_dist<< "  ";
            // Return position in sorted list where nn inserted
            auto r = InsertIntoPool(retset.data(), cur_list_size, nn);  // may be overflow in retset...
            if (cur_list_size < l_search) {
              ++cur_list_size;
              if (unlikely(cur_list_size >= retset.size())) {
                retset.resize(2 * cur_list_size);
              }
            }
            // nk logs the best position in the retset that was updated due to
            // neighbors of n.
            if (r < nk)
              nk = r;
          }
        }
        // process prefetch-ed nhood

        if (dyn_search_l) {
          // TODO(gh): contention still exists in id2tag(x)
          // O(n), but it is not slow as L is typically smaller than 300.
          // l_search monotonically increases to handle deleted nodes.
          _u32 tot = 0, cur = 0;
          for (cur = 0; cur < cur_list_size; ++cur) {
            uint32_t tag = id2tag(retset[cur].id);
            if (exclude_nodes->find(tag) == exclude_nodes->end()) {
              ++tot;
              if (tot == original_l_search) {
                break;
              }
            }
          }
          // cur is the stopped index (cur + 1 is the length it should be)
          l_search = std::max(original_l_search, cur + 1);
        }

        if (stats != nullptr) {
          stats->cpu_us += (double) cpu_timer.elapsed();
        }
      }

      // update best inserted position
      //

      if (nk <= k)
        k = nk;  // k is the best position in retset updated in this round.
      else
        ++k;

      hops++;
      if (stats != nullptr && stats->n_current_used != 0) {
        auto diskSearchEnd = std::chrono::high_resolution_clock::now();
        double elapsedSeconds =
            std::chrono::duration_cast<std::chrono::milliseconds>(diskSearchEnd - diskSearchBegin).count();
        if (elapsedSeconds >= stats->n_current_used)
          break;
      }
    }
    // re-sort by distance
    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) { return left < right; });

    if (passthrough_page_ref == nullptr) {
      reader->deref(&page_ref, ctx);
    }

    
    if(passthrough_data == nullptr){
      push_query_buf(query_buf);
    }

    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
  }

  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::beam_search(const T *query, const _u64 k_search, const _u32 mem_L, const _u64 l_search,
                                        TagT *res_tags, float *distances, const _u64 beam_width, QueryStats *stats,
                                        tsl::robin_set<uint32_t> *deleted_nodes, bool dyn_search_l) {
    // iterate to fixed point
    std::shared_lock lk(merge_lock);
    std::vector<Neighbor> expanded_nodes_info;
    this->do_beam_search(query, mem_L, (_u32) l_search, (_u32) beam_width, expanded_nodes_info, nullptr, stats,
                         deleted_nodes, dyn_search_l);
    _u64 res_count = 0;
    for (uint32_t i = 0; i < l_search && res_count < k_search && i < expanded_nodes_info.size(); i++) {
      res_tags[res_count] = id2tag(expanded_nodes_info[i].id);
      distances[res_count] = expanded_nodes_info[i].distance;
      res_count++;
    }
    return res_count;
  }

  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::beam_search(const T *query, const T *queryl, const float alpha, const _u64 k_search, const _u32 mem_L, const _u64 l_search,
                                        TagT *res_tags, float *distances, const _u64 beam_width, QueryStats *stats,
                                        tsl::robin_set<uint32_t> *deleted_nodes, bool dyn_search_l) {
    // iterate to fixed point
    std::shared_lock lk(merge_lock);
    std::vector<Neighbor> expanded_nodes_info;
    this->do_beam_search(query, queryl, alpha, mem_L, (_u32) l_search, (_u32) beam_width, expanded_nodes_info, nullptr, stats,
                         deleted_nodes, dyn_search_l);
    _u64 res_count = 0;
    for (uint32_t i = 0; i < l_search && res_count < k_search && i < expanded_nodes_info.size(); i++) {
      res_tags[res_count] = id2tag(expanded_nodes_info[i].id);
      distances[res_count] = expanded_nodes_info[i].distance;
      res_count++;
    }
    return res_count;
  }

  template class SSDIndex<float>;
  template class SSDIndex<_s8>;
  template class SSDIndex<_u8>;
}  // namespace pipeann
