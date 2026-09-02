#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <climits>

#include "log.h"

constexpr uint32_t kMergedMaxAlphaRangeLen = 2;

// =========================================================================
// Alpha Range Merge Utilities (same as in aux_utils.cpp)
// =========================================================================

// Method 2: Greedy Merge by Gap
static std::vector<std::pair<int8_t, int8_t>>
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
    result[best_idx].second = std::max(result[best_idx].second, result[best_idx + 1].second);
    result.erase(result.begin() + best_idx + 1);
  }
  return result;
}

// Method 3: Max-Coverage (optimal)
static std::vector<std::pair<int8_t, int8_t>>
merge_ranges_max_coverage(const std::vector<std::pair<int8_t, int8_t>>& ranges,
                          uint32_t target_max) {
  if (ranges.size() <= target_max || ranges.empty()) return ranges;
  if (target_max == 0) return {};

  bool covered[101] = {false};
  for (const auto& r : ranges) {
    int8_t lo = std::max<int8_t>(0, r.first);
    int8_t hi = std::min<int8_t>(100, r.second);
    for (int i = lo; i <= hi; i++) covered[i] = true;
  }

  int first = -1, last = -1;
  for (int i = 0; i <= 100; i++) {
    if (covered[i]) {
      if (first < 0) first = i;
      last = i;
    }
  }
  if (first < 0) return {};

  if (target_max == 1) {
    return {{(int8_t)first, (int8_t)last}};
  }

  // Find the largest contiguous uncovered gap, split there
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
  if (gap_start >= 0) {
    int gap_len = last + 1 - gap_start;
    if (gap_len > best_gap_len) {
      best_gap_len = gap_len;
      best_gap_start = gap_start;
      best_gap_end = last;
    }
  }

  if (best_gap_len <= 0) {
    return {{(int8_t)first, (int8_t)last}};
  }

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
static std::vector<std::pair<int8_t, int8_t>>
merge_alpha_ranges(const std::vector<std::pair<int8_t, int8_t>>& ranges,
                   int merge_method) {
  if (ranges.size() <= kMergedMaxAlphaRangeLen) return ranges;
  switch (merge_method) {
    case 2:
      return merge_ranges_greedy(ranges, kMergedMaxAlphaRangeLen);
    case 3:
      return merge_ranges_max_coverage(ranges, kMergedMaxAlphaRangeLen);
    default:
      return ranges;
  }
}

// =========================================================================
// Main Conversion Logic
// =========================================================================
void convert_topo_merge_alpha(const std::string &input_meta_file,
                               const std::string &input_topo_file,
                               const std::string &output_meta_file,
                               const std::string &output_topo_file,
                               int merge_method) {
  const size_t PAGE_SIZE = 8192;

  // ==========================================
  // 1. 读取原始 Meta 文件
  // ==========================================
  std::ifstream in_meta(input_meta_file, std::ios::binary);
  if (!in_meta.is_open()) {
    LOG(ERROR) << "Failed to open input meta file: " << input_meta_file;
    return;
  }

  std::vector<char> meta_header(PAGE_SIZE, 0);
  in_meta.read(meta_header.data(), PAGE_SIZE);
  in_meta.close();

  const char *m_ptr = meta_header.data();
  uint32_t node_num = *((uint32_t *)m_ptr);
  m_ptr += sizeof(uint32_t);
  uint32_t emb_dim = *((uint32_t *)m_ptr);
  m_ptr += sizeof(uint32_t);
  uint32_t loc_dim = *((uint32_t *)m_ptr);
  m_ptr += sizeof(uint32_t);
  uint32_t max_nbr_len = *((uint32_t *)m_ptr);
  m_ptr += sizeof(uint32_t);
  uint32_t old_max_alpha_range_len = *((uint32_t *)m_ptr);
  m_ptr += sizeof(uint32_t);
  uint64_t old_nnodes_per_sector = *((uint64_t *)m_ptr);
  m_ptr += sizeof(uint64_t);
  uint32_t enterpoint_set_size = *((uint32_t *)m_ptr);
  m_ptr += sizeof(uint32_t);
  std::vector<uint32_t> enterpoint_set(enterpoint_set_size);
  std::memcpy(enterpoint_set.data(), m_ptr, sizeof(uint32_t) * enterpoint_set_size);

  LOG(INFO) << "Input meta: node_num=" << node_num << ", emb_dim=" << emb_dim
            << ", loc_dim=" << loc_dim << ", max_nbr_len=" << max_nbr_len
            << ", max_alpha_range_len=" << old_max_alpha_range_len
            << ", enterpoint_set_size=" << enterpoint_set_size;
  LOG(INFO) << "Merge method: " << merge_method
            << " (2=greedy, 3=max-coverage), target max pairs: " << kMergedMaxAlphaRangeLen;

  // ==========================================
  // 2. 计算新的布局参数（合并后）
  // ==========================================
  uint32_t old_single_nbr_size = sizeof(uint32_t) + (2 * old_max_alpha_range_len * sizeof(int8_t));
  uint32_t old_fixed_topo_size = sizeof(uint32_t) + (max_nbr_len * old_single_nbr_size);

  uint32_t new_single_nbr_size = sizeof(uint32_t) + (2 * kMergedMaxAlphaRangeLen * sizeof(int8_t));
  uint32_t new_fixed_topo_size = sizeof(uint32_t) + (max_nbr_len * new_single_nbr_size);
  uint64_t new_nnodes_per_sector = PAGE_SIZE / new_fixed_topo_size;

  LOG(INFO) << "Old topo: single_nbr=" << old_single_nbr_size << "B, fixed_topo=" << old_fixed_topo_size
            << "B, nnodes/sector=" << old_nnodes_per_sector;
  LOG(INFO) << "New topo: single_nbr=" << new_single_nbr_size << "B, fixed_topo=" << new_fixed_topo_size
            << "B, nnodes/sector=" << new_nnodes_per_sector;

  // ==========================================
  // 3. 写入新的 Meta 文件（max_alpha_range_len = kMergedMaxAlphaRangeLen）
  // ==========================================
  std::remove(output_meta_file.c_str());
  std::ofstream out_meta(output_meta_file, std::ios::binary);
  size_t raw_meta_size =
      sizeof(node_num) + sizeof(emb_dim) + sizeof(loc_dim) +
      sizeof(max_nbr_len) + sizeof(uint32_t) +  // max_alpha_range_len = 2
      sizeof(new_nnodes_per_sector) +
      sizeof(enterpoint_set_size) + (sizeof(uint32_t) * enterpoint_set_size);

  size_t aligned_meta_size = (raw_meta_size + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
  std::vector<char> new_meta_buffer(aligned_meta_size, 0);

  char *new_m_ptr = new_meta_buffer.data();
  auto copy_meta = [&](const void *src, size_t sz) {
    std::memcpy(new_m_ptr, src, sz);
    new_m_ptr += sz;
  };

  uint32_t new_max_alpha_range_len = kMergedMaxAlphaRangeLen;
  copy_meta(&node_num, sizeof(uint32_t));
  copy_meta(&emb_dim, sizeof(uint32_t));
  copy_meta(&loc_dim, sizeof(uint32_t));
  copy_meta(&max_nbr_len, sizeof(uint32_t));
  copy_meta(&new_max_alpha_range_len, sizeof(uint32_t));
  copy_meta(&new_nnodes_per_sector, sizeof(uint64_t));
  copy_meta(&enterpoint_set_size, sizeof(uint32_t));
  copy_meta(enterpoint_set.data(), sizeof(uint32_t) * enterpoint_set_size);

  out_meta.write(new_meta_buffer.data(), aligned_meta_size);
  out_meta.close();

  LOG(INFO) << "New meta file written: " << output_meta_file;

  // ==========================================
  // 4. 读取原始 Topo 文件，合并 alpha range 后写入新 Topo 文件
  // ==========================================
  std::ifstream in_topo(input_topo_file, std::ios::binary);
  if (!in_topo.is_open()) {
    LOG(ERROR) << "Failed to open input topo file: " << input_topo_file;
    return;
  }

  std::remove(output_topo_file.c_str());
  std::ofstream out_topo(output_topo_file, std::ios::binary);
  std::vector<char> page_buf(PAGE_SIZE, 0);
  size_t page_off = 0;

  std::vector<char> sector_buf(PAGE_SIZE, 0);
  std::vector<char> new_node_buf(new_fixed_topo_size, 0);

  uint64_t nodes_read = 0;
  uint64_t nodes_per_sector_old = (uint64_t)old_nnodes_per_sector;

  // 统计合并效果
  uint64_t total_original_pairs = 0;
  uint64_t total_merged_pairs = 0;
  uint64_t neighbors_with_merge = 0;

  while (in_topo.read(sector_buf.data(), PAGE_SIZE) ||
         in_topo.gcount() > 0) {
    std::streamsize bytes_read = in_topo.gcount();
    if (bytes_read < (std::streamsize)(sizeof(uint32_t))) {
      break;
    }

    uint32_t nodes_in_this_sector = (uint32_t)(
        std::min((uint64_t)(bytes_read / old_fixed_topo_size), nodes_per_sector_old));
    nodes_in_this_sector = std::min(nodes_in_this_sector, (uint32_t)(node_num - nodes_read));

    for (uint32_t n = 0; n < nodes_in_this_sector; n++) {
      const char *src_ptr = sector_buf.data() + n * old_fixed_topo_size;
      uint32_t neighbor_size = *((uint32_t *)src_ptr);
      src_ptr += sizeof(uint32_t);

      // 构建新格式
      std::memset(new_node_buf.data(), 0, new_fixed_topo_size);
      char *dst_ptr = new_node_buf.data();
      uint32_t actual_nbr_to_write = std::min(neighbor_size, max_nbr_len);
      *((uint32_t *)dst_ptr) = actual_nbr_to_write;
      dst_ptr += sizeof(uint32_t);

      for (uint32_t k = 0; k < neighbor_size; k++) {
        // 磁盘格式: [ID(4B)] + [2×max_alpha_range_len × int8_t] (无 range_size 字段)
        uint32_t nbr_id = *((uint32_t *)src_ptr);
        src_ptr += sizeof(uint32_t);

        // 读取所有 range pairs (固定 old_max_alpha_range_len 对)
        // 磁盘格式无 range_size，未使用 slot 补 [0,0]，需 trim trailing [0,0]
        std::vector<std::pair<int8_t, int8_t>> temp_ranges;
        for (uint32_t t = 0; t < old_max_alpha_range_len; t++) {
          int8_t a = *((int8_t *)src_ptr);
          src_ptr += sizeof(int8_t);
          int8_t b = *((int8_t *)src_ptr);
          src_ptr += sizeof(int8_t);
          temp_ranges.push_back({a, b});
        }
        // Trim trailing [0,0] padding pairs (保留至少一个 [0,0] 如果全部是 padding)
        while (temp_ranges.size() > 1 &&
               temp_ranges.back().first == 0 && temp_ranges.back().second == 0) {
          temp_ranges.pop_back();
        }

        // 如果当前邻居超出 max_nbr_len，跳过写入
        if (k >= actual_nbr_to_write) continue;

        // 合并 alpha range
        size_t orig_count = temp_ranges.size();
        auto merged = merge_alpha_ranges(temp_ranges, merge_method);
        total_original_pairs += orig_count;
        total_merged_pairs += merged.size();
        if (merged.size() < orig_count) neighbors_with_merge++;

        // 写入新格式: [ID(4B)] + [merged ranges (max kMergedMaxAlphaRangeLen pairs)]
        *((uint32_t *)dst_ptr) = nbr_id;
        dst_ptr += sizeof(uint32_t);

        unsigned to_copy = std::min((unsigned)merged.size(), kMergedMaxAlphaRangeLen);
        for (unsigned r = 0; r < kMergedMaxAlphaRangeLen; r++) {
          if (r >= to_copy) {
          *((int8_t *)dst_ptr) = 0;
          dst_ptr += sizeof(int8_t);
          *((int8_t *)dst_ptr) = 0;
          dst_ptr += sizeof(int8_t);
            continue;
          }
          *((int8_t *)dst_ptr) = merged[r].first;
          dst_ptr += sizeof(int8_t);
          *((int8_t *)dst_ptr) = merged[r].second;
          dst_ptr += sizeof(int8_t);
        }
      }

      // 写入 page buffer
      if (page_off + new_fixed_topo_size > PAGE_SIZE) {
        std::memset(page_buf.data() + page_off, 0, PAGE_SIZE - page_off);
        out_topo.write(page_buf.data(), PAGE_SIZE);
        page_off = 0;
      }

      std::memcpy(page_buf.data() + page_off, new_node_buf.data(), new_fixed_topo_size);
      page_off += new_fixed_topo_size;

      nodes_read++;
      if (nodes_read % 100000 == 0) {
        LOG(INFO) << "Converted " << nodes_read << " nodes...";
      }

      if (nodes_read >= node_num) {
        break;
      }
    }
    if (nodes_read >= node_num) {
      break;
    }
  }

  // 刷新剩余不满一页的数据
  if (page_off > 0) {
    std::memset(page_buf.data() + page_off, 0, PAGE_SIZE - page_off);
    out_topo.write(page_buf.data(), PAGE_SIZE);
  }

  in_topo.close();
  out_topo.close();

  // 输出合并统计
  LOG(INFO) << "Conversion finished: " << nodes_read << " nodes converted";
  LOG(INFO) << "Alpha range merge stats:";
  LOG(INFO) << "  Original total pairs: " << total_original_pairs;
  LOG(INFO) << "  Merged total pairs:   " << total_merged_pairs;
  LOG(INFO) << "  Compression ratio:    " << (float)total_merged_pairs / std::max((uint64_t)1, total_original_pairs);
  LOG(INFO) << "  Neighbors with merge: " << neighbors_with_merge;
  LOG(INFO) << "Output topo file: " << output_topo_file;
}

int main(int argc, char **argv) {
  if (argc != 6) {
    std::cerr << "Usage: " << argv[0]
              << " <input_meta_file> <input_topo_file>"
              << " <output_meta_file> <output_topo_file>"
              << " <merge_method>" << std::endl;
    std::cerr << "  merge_method: 2=greedy gap merge, 3=max-coverage" << std::endl;
    return 1;
  }

  std::string input_meta_file = argv[1];
  std::string input_topo_file = argv[2];
  std::string output_meta_file = argv[3];
  std::string output_topo_file = argv[4];
  int merge_method = std::atoi(argv[5]);

  if (merge_method != 2 && merge_method != 3) {
    std::cerr << "Invalid merge_method: " << merge_method
              << ". Must be 2 (greedy) or 3 (max-coverage)." << std::endl;
    return 1;
  }

  convert_topo_merge_alpha(input_meta_file, input_topo_file,
                            output_meta_file, output_topo_file,
                            merge_method);

  LOG(INFO) << "Alpha range merge completed successfully!";
  return 0;
}
