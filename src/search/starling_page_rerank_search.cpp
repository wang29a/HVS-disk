#include "linux_aligned_file_reader.h"
#include "libcuckoo/cuckoohash_map.hh"
#include "log.h"
#include "neighbor.h"
#include "ssd_index.h"
#include <malloc.h>
#include <algorithm>

#include <memory>
#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include "timer.h"
#include "tsl/robin_set.h"
#include "utils.h"
#include <math.h>

#include <unistd.h>
#include <sys/syscall.h>

#ifndef USE_AIO
#include "liburing.h"
#endif
namespace pipeann {

  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::pq_page_rerank_search(const T *query_emb, const T *query_loc, const float alpha,
                                                   const _u64 k_search, const _u32 mem_L, const _u64 l_search,
                                                   TagT *res_tags, float *res_dists,
                                                   const _u64 beam_width, QueryStats *stats) {
    QueryBuffer<T> *query_buf = pop_query_buf(query_emb, query_loc);
    void *ctx = reader->get_ctx();

    if (beam_width > MAX_N_SECTOR_READS) {
      LOG(ERROR) << "Beamwidth can not be higher than MAX_N_SECTOR_READS";
      crash();
    }

    const T *query_e = query_buf->aligned_query_T;
    const T *query_l = query_buf->aligned_queryl_T;
    query_buf->reset();

    // pointers to buffers for data
    T *data_buf = query_buf->coord_scratch;
    _u64 &data_buf_idx = query_buf->coord_idx;
    T *datal_buf = query_buf->coordl_scratch;
    _u64 &datal_buf_idx = query_buf->coordl_idx;
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);

    // sector scratch
    char *sector_scratch = query_buf->sector_scratch;
    _u64 &sector_scratch_idx = query_buf->sector_idx;

    // dual PQ: emb and loc
    float *pq_dists = query_buf->aligned_pqtable_dist_scratch;
    float *pq_distsl = query_buf->aligned_pqtable_distl_scratch;
    pq_table.populate_chunk_distances(query_e, pq_dists);
    pq_tablel.populate_chunk_distances(query_l, pq_distsl);

    // scratch for PQ distance lookup
    float *dist_scratch = query_buf->aligned_dist_scratch;
    float *distl_scratch = query_buf->aligned_distl_scratch;
    _u8 *pq_coord_scratch = query_buf->aligned_pq_coord_scratch;
    _u8 *pq_coordl_scratch = query_buf->aligned_pq_coordl_scratch;

    Timer query_timer;

    // stats
    if (stats != nullptr) {
      stats->io_us = 0;
      stats->cpu_us = 0;
      stats->cpu_us1 = 0;
    }

    // ---- batch PQ distance computation for emb ----
    auto compute_pq_dists_e = [this, pq_coord_scratch, pq_dists](const unsigned *ids, const _u64 n_ids,
                                                                float *dists_out) {
      ::aggregate_coords(ids, n_ids, this->data.data(), this->n_chunks, pq_coord_scratch);
      ::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists, dists_out);
    };

    // ---- batch PQ distance computation for loc ----
    auto compute_pq_dists_l = [this, pq_coordl_scratch, pq_distsl](const unsigned *ids, const _u64 n_ids,
                                                                    float *dists_out) {
      ::aggregate_coords(ids, n_ids, this->datal.data(), this->n_chunks, pq_coordl_scratch);
      ::pq_dist_lookup(pq_coordl_scratch, n_ids, this->n_chunks, pq_distsl, dists_out);
    };

    // ---- init from medoid ----
    std::vector<Neighbor> retset(mem_L + l_search * 10);
    tsl::robin_set<_u64> &visited = *(query_buf->visited);
    tsl::robin_set<unsigned> &page_visited = *(query_buf->page_visited);
    unsigned cur_list_size = 0;

    // start from the first medoid with PQ distance
    compute_pq_dists_e(&medoids[0], num_medoids, dist_scratch);
    compute_pq_dists_l(&medoids[0], num_medoids, distl_scratch);
    for (size_t i = 0; i < num_medoids; i ++) {
      float d_e = dist_scratch[i], d_l = distl_scratch[i];
      retset[cur_list_size].id = medoids[i];
      retset[cur_list_size].distance = alpha * d_e + (1 - alpha) * d_l;
      retset[cur_list_size++].flag = true;
      visited.insert(medoids[i]);
    }

    std::sort(retset.begin(), retset.begin() + cur_list_size);

    // ---- coarse search state (page-level beam search with PQ) ----
    unsigned k = 0;

    std::vector<unsigned> frontier;
    frontier.reserve(2 * beam_width);

    using page_fnhood_t = std::tuple<unsigned, unsigned,PageArr, char *>;  // <node_id, page_id, page_buf>
    std::vector<page_fnhood_t> frontier_nhoods;
    frontier_nhoods.reserve(2 * beam_width);
    std::vector<IORequest> frontier_read_reqs;
    frontier_read_reqs.reserve(2 * beam_width);

    auto p_reader = (LinuxAlignedFileReader *)this->reader.get();
    auto & loc2phy_coord = p_reader->loc2phy_coord;
    FileHandle topo_fd = p_reader->topo_file_desc;
    std::unordered_map<unsigned, char *> page_buf_map;     // page-level缓存
    std::unordered_map<unsigned, int> page_rank_map;        // 页面排名
    auto & block_cache = p_reader->block_cache;            // 块缓存
    int use_topo_buffer = (p_reader->strategy >> 4) & 0x1;  // 缓存策略标志

    // neighbor layout: [4B nbr_count][max_nbr_len × (4B neighbor_id + 2B×max_alpha_range_len alpha_pairs)]
    uint32_t nbr_data_len = max_alpha_range_len * 2 + (uint32_t)sizeof(uint32_t);

    // ---- main search loop: page-level pipelined IO + PQ compute ----
    while (k < cur_list_size) {
      unsigned nk = cur_list_size;
      frontier.clear();
      frontier_nhoods.clear();
      frontier_read_reqs.clear();

      _u32 marker = k;
      _u32 num_seen = 0;

      // select frontier candidates (page-level dedup via page_visited)
      while (marker < cur_list_size && frontier.size() < beam_width && num_seen < beam_width) {
        const unsigned pid = id2page(retset[marker].id);
        if (retset[marker].flag) {
          ++num_seen;
          frontier.push_back(retset[marker].id);
          retset[marker].flag = false;
        }
        ++marker;
      }

      // ---- Issue async IO for frontier nodes (read full topo pages) ----
      std::vector<uint32_t> locked, page_locked;
      int n_ios = 0;
      std::vector<IORequest> disk_read_reqs;  // 只发送实际需要磁盘IO的请求
      if (!frontier.empty()) {
        if (stats != nullptr)
          ++stats->n_hops;
        // locked = this->lock_idx(idx_lock_table, kInvalidID, frontier, true);
        // page_locked = this->lock_page_idx(page_idx_lock_table, kInvalidID, frontier, true);

        disk_read_reqs.reserve(frontier.size());

        for (_u64 i = 0; i < frontier.size(); ++i) {
          const unsigned id = frontier[i];
          const uint64_t page_id = id2page(id) - 1;
          char *buf;
          bool from_cache = false;

          // 检查 page_buf_map
          if (page_buf_map.find(page_id) != page_buf_map.end()) {
            buf = page_buf_map[page_id];
            from_cache = true;
          } else {
            buf = sector_scratch + sector_scratch_idx * SECTOR_LEN;
            page_buf_map.insert({page_id, buf});
            sector_scratch_idx = (sector_scratch_idx + 1) % MAX_N_SECTOR_READS;
          }

          PageArr layout;
          if (unlikely(!page_layout.find(page_id, layout))) {
            LOG(ERROR) << "Page layout not found for page " << page_id;
            crash();
          }
          frontier_nhoods.emplace_back(id, page_id, layout, buf);

          IORequest req(page_id * SECTOR_LEN, SECTOR_LEN, buf, page_id * SECTOR_LEN, SECTOR_LEN);
          // frontier_read_reqs.push_back(req);

          if (from_cache) {
            // page_buf_map hit: 标记为已完成，不计入 n_ios
            // frontier_read_reqs.back().finished = true;
            if (stats != nullptr)
              ++stats->n_cache_hits;
          } else {
            disk_read_reqs.push_back(req);
            ++n_ios;
            if (stats != nullptr) {
              ++stats->n_4k;
              ++stats->n_ios;
            }
          }
        }
        // 发送实际需要磁盘IO的请求
        if (!disk_read_reqs.empty()) {
          reader->send_io(disk_read_reqs, ctx, false, topo_fd);
          // reader->read_fd(topo_fd, disk_read_reqs, ctx);
        }
      }

      // ---- Wait for this round's IO, then unlock ----
      auto io_time_st = std::chrono::high_resolution_clock::now();
      if (!frontier.empty()) {
        for (int i = 0; i < n_ios; ++i)
          reader->poll_wait(ctx);

        // this->unlock_page_idx(page_idx_lock_table, page_locked);
        // this->unlock_idx(idx_lock_table, locked);
      }
      auto io_time_ed = std::chrono::high_resolution_clock::now();
      if (stats != nullptr){
        stats->io_us += std::chrono::duration_cast<std::chrono::microseconds>(io_time_ed - io_time_st).count();
        stats->io_us1 += std::chrono::duration_cast<std::chrono::microseconds>(io_time_ed - io_time_st).count();
      }

      // ---- CPU: process this round's frontier nodes ----
      auto cpu_st = std::chrono::high_resolution_clock::now();
      for (auto &[id, pid, layout, buf] : frontier_nhoods) {
        // compute PQ distance for the entry node and expand neighbors
        for (unsigned j = 0; j < nnodes_per_sector; ++j) {
          const unsigned nid = layout[j];
          if (nid == id) {
            // compute PQ for entry node neighbors
            // neighbor layout: [4B nbr_count][max_nbr_len × (4B neighbor_id + 2B×max_alpha_range_len alpha_pairs)]
            unsigned *node_nbrs_raw = (unsigned *)(buf + j * max_node_len);
            unsigned nnbrs = *node_nbrs_raw;
            std::vector<unsigned> nbr_ids;
            nbr_ids.reserve(nnbrs);

            // filter by alpha range stored in each neighbor entry
            node_nbrs_raw ++;
            for (unsigned m = 0; m < nnbrs; ++m) {
              uint32_t neighbor_id = *(uint32_t *)((char *)node_nbrs_raw + m * nbr_data_len);
              const uint8_t *alpha_ptr = (const uint8_t *)node_nbrs_raw + m * nbr_data_len + sizeof(uint32_t);
              bool search_flag = false;
              for (uint32_t k = 0; k < max_alpha_range_len; ++k) {
                int8_t a1 = (int8_t)alpha_ptr[k * 2];
                int8_t a2 = (int8_t)alpha_ptr[k * 2 + 1];
                if (!(a1 >= 0 && a1 <= 100 && a2 >= 0 && a2 <= 100))
                  break;
                if (a1 == 0 && a2 == 0)
                  break;
                float a100 = alpha * 100.0f;
                if (a100 >= a1 && a100 <= a2) {
                  search_flag = true;
                  break;
                }
                if (a100 < a1)
                  break;
              }
              if (visited.find(neighbor_id) == visited.end() && search_flag) {
                nbr_ids.push_back(neighbor_id);
                visited.insert(neighbor_id);
              }
            }
            if (!nbr_ids.empty()) {
              compute_pq_dists_e(nbr_ids.data(), nbr_ids.size(), dist_scratch);
              compute_pq_dists_l(nbr_ids.data(), nbr_ids.size(), distl_scratch);
              for (size_t m = 0; m < nbr_ids.size(); ++m) {
                const unsigned nbor_id = nbr_ids[m];
                const float d_e = dist_scratch[m];
                const float d_l = distl_scratch[m];
                // LOG(INFO) << nbor_id << " " << d_e << " " << d_l;
                const float combined = alpha * d_e + (1 - alpha) * d_l;
                if (stats != nullptr)
                  stats->n_cmps++;
                if (cur_list_size < l_search || combined < retset[cur_list_size - 1].distance) {
                  Neighbor nn(nbor_id, combined, true);
                  auto r = InsertIntoPool(retset.data(), cur_list_size, nn);
                  if (cur_list_size < l_search)
                    ++cur_list_size;
                  if (r < nk)
                    nk = r;
                }
              }
            }
            break;
          }
        }
      }
      auto cpu_ed = std::chrono::high_resolution_clock::now();
      if (stats != nullptr)
        stats->cpu_us += std::chrono::duration_cast<std::chrono::microseconds>(cpu_ed - cpu_st).count();

      if (nk <= k)
        k = nk;
      else
        ++k;
    }

    // 清理 page_buf_map (coarse search 结束，rerank 开始前)
    page_buf_map.clear();
    page_rank_map.clear();

    // ---- RERANK: read exact coordinates and compute precise distances ----
    Timer rerank_timer_2;

    // compute rerank truncation length (adaptive based on l_search)
    int p = p_reader->trunc_len;
    int rerank_trunc_len;
    rerank_trunc_len = std::max(p, (int)std::ceil(p * (1 + log10((double)l_search / (double)p))));
    rerank_trunc_len = std::min(rerank_trunc_len, (int)l_search);
    rerank_trunc_len = std::min(rerank_trunc_len, (int)cur_list_size);

    // collect top candidates for rerank directly from retset (already PQ-sorted)
    std::vector<uint32_t> to_rerank_ids;
    to_rerank_ids.reserve((_u64)rerank_trunc_len);
    for (unsigned i = 0; i < (_u64)rerank_trunc_len; ++i) {
      to_rerank_ids.push_back(retset[i].id);
    }

    if (to_rerank_ids.empty()) {
      push_query_buf(query_buf);
      return 0;
    }

    // sort and deduplicate for IO
    std::sort(to_rerank_ids.begin(), to_rerank_ids.end());
    to_rerank_ids.erase(std::unique(to_rerank_ids.begin(), to_rerank_ids.end()), to_rerank_ids.end());

    // lock all nodes
    for (auto id : to_rerank_ids) {
      vec_lock_table.rdlock(id);
    }

    // prepare IO requests for exact coordinates (USE_TOPO_DISK mode)
    std::vector<IORequest> read_reqs;
    read_reqs.reserve(to_rerank_ids.size());
    std::vector<std::pair<uint32_t, char *>> idx_buf;  // <id, node_buf>
    idx_buf.reserve(to_rerank_ids.size());
    std::unordered_map<uint32_t, char *> pid_buf_map;

    FileHandle coord_fd = p_reader->coord_file_desc;

    for (auto id : to_rerank_ids) {
      unsigned loc = id2loc_coord(id);
      uint64_t pid = loc_sector_no_coord(loc);
      if (loc == kInvalidID) {
        LOG(ERROR) << "id " << id << " not found in id2loc_coord";
        exit(-1);
      }

      if (pid_buf_map.find(pid) != pid_buf_map.end()) {
        idx_buf.push_back({id, offset_to_loc_coord(pid_buf_map[pid], loc)});
        continue;
      }
      uint64_t &cur_buf_idx = query_buf->sector_idx;
      auto buf = sector_scratch + cur_buf_idx * size_per_io;
      pid_buf_map.insert({pid, buf});
      read_reqs.push_back(IORequest(static_cast<_u64>(pid) * SECTOR_LEN, size_per_io, buf,
                                      u_loc_offset_coord(loc), coord_len));
      cur_buf_idx = (cur_buf_idx + 1) % MAX_N_SECTOR_READS;
      idx_buf.push_back({id, offset_to_loc_coord(buf, loc)});
    }

    Timer io_timer;
    reader->read_fd(coord_fd, read_reqs, ctx);
    if (stats != nullptr) {
      stats->rerank_ios += read_reqs.size();
      stats->io_us += io_timer.elapsed();
      stats->io_us2 += io_timer.elapsed();
    }

    // unlock all nodes
    for (auto id : to_rerank_ids) {
      vec_lock_table.unlock(id);
    }

    // compute exact distances
    Timer timer;
    std::vector<std::pair<uint32_t, float>> exact_results;
    exact_results.reserve(idx_buf.size());
    for (auto &[id, node_buf] : idx_buf) {
      T *node_fp_coords_copy = data_buf + data_buf_idx * aligned_dim;
      T *node_fp_coordsl_copy = datal_buf + datal_buf_idx * aligned_dim;
      memcpy(node_fp_coords_copy, node_buf, data_dim * sizeof(T));
      memcpy(node_fp_coordsl_copy, node_buf + data_dim * sizeof(T), datal_dim * sizeof(T));
      // data_buf_idx++;
      float cur_expanded_diste = dist_cmp->compare(query_e, node_fp_coords_copy, (unsigned) data_dim);
      float cur_expanded_distl = dist_cmp->compare(query_l, node_fp_coordsl_copy, (unsigned) data_dim);
      float cur_expanded_dist = alpha*std::sqrt(cur_expanded_diste) + (1-alpha)*std::sqrt(cur_expanded_distl);
      exact_results.emplace_back(id, cur_expanded_dist);
    }
    if (stats != nullptr) {
      stats->accuracy_compute_us += timer.elapsed();
      stats->rerank_us_2 += rerank_timer_2.elapsed();
    }

    // sort by exact distance
    std::sort(exact_results.begin(), exact_results.end(),
              [](const auto &a, const auto &b) { return a.second < b.second; });

    // copy top k_search results
    _u64 t = 0;
    for (_u64 i = 0; i < exact_results.size() && t < k_search; ++i) {
      if (i > 0 && exact_results[i].first == exact_results[i - 1].first)
        continue;
      res_tags[t] = id2tag(exact_results[i].first);
      if (res_dists != nullptr)
        res_dists[t] = exact_results[i].second;
      ++t;
    }

    push_query_buf(query_buf);
    if (stats != nullptr)
      stats->total_us = (double) query_timer.elapsed();
    return t;
  }
  template class SSDIndex<float>;
  template class SSDIndex<_s8>;
  template class SSDIndex<_u8>;
}