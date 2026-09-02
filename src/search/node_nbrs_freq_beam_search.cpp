
#include "ssd_index.h"
#include <cstddef>

namespace pipeann {

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::init_node_visit_counter() {
    this->node_visit_counter.clear();
    this->node_visit_counter.resize(this->num_points);
    for (_u32 i = 0; i < node_visit_counter.size(); i++) {
      this->node_visit_counter[i].first = i;
      this->node_visit_counter[i].second = 0;
    }
  }
  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::generate_node_nbrs_freq(
    const std::string& freq_save_path, const size_t query_num,
    const T *query, const T *queryl, const float *alpha, const _u64 k_search, const _u32 mem_L, const _u64 l_search,
    TagT *res_tags, float *distances, const _u64 beam_width, QueryStats *stats) {

    this->count_visited_nodes = true;
    this->count_visited_nbrs = true;

    init_node_visit_counter();

    nbrs_freq_counter_.resize(this->num_points);
    for (auto& m : nbrs_freq_counter_) m.clear();
  #pragma omp parallel for schedule(dynamic, 1)
    for (_s64 i = 0; i < (int64_t) query_num; i++) {
      std::shared_lock lk(merge_lock);
      std::vector<Neighbor> expanded_nodes_info;
      do_beam_search(
        query + (i * data_dim), queryl + (i*datal_dim), *(alpha+i), mem_L, l_search,
        beam_width, expanded_nodes_info, nullptr, stats + i, nullptr, false);
      _u64 res_count = 0;
      for (uint32_t j = 0; j < l_search && res_count < k_search && j < expanded_nodes_info.size(); j ++) {
        res_tags[res_count+(i*k_search)] = id2tag(expanded_nodes_info[j].id);
        distances[res_count+(i*k_search)] = expanded_nodes_info[j].distance;
        // LOG(INFO) << res_count+(i*k_search) << " " << res_tags[res_count+(i*k_search)] << " ";
        res_count++;
      }
    }
    this->count_visited_nbrs = false;
    this->count_visited_nodes = false;

    // save freq file
    const std::string freq_file = freq_save_path + "_freq.bin";
    std::ofstream writer(freq_file, std::ios::binary | std::ios::out);
    LOG(INFO) << "Writing visited nodes and neighbors frequency: " << freq_file ;
    unsigned num = node_visit_counter.size(); // number of data points
    if (num != this->num_points) {
      LOG(EORR) << "Total number of elements mismatch";
      exit(1);
    }
    writer.write((char *)&num, sizeof(unsigned));

    for (size_t i = 0; i < num; ++i) {
      writer.write((char *)(&(node_visit_counter[i].second)), sizeof(unsigned));
    }
    for (size_t i = 0; i < num; ++i) {
      unsigned p_size = nbrs_freq_counter_[i].size();
      writer.write((char *)(&p_size), sizeof(unsigned));
      for (const auto [nbr_id, nbr_freq] : nbrs_freq_counter_[i]) {
        writer.write((char *)(&nbr_id), sizeof(unsigned));
        writer.write((char *)(&nbr_freq), sizeof(unsigned));
      }
    }
    LOG(INFO) << "Writing frequency file finished";
    return 0;
  }

  template class SSDIndex<float>;
  template class SSDIndex<_s8>;
  template class SSDIndex<_u8>;
}