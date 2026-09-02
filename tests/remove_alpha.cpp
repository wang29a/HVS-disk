#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <algorithm>

#include "log.h"

// 将带 alpha range 的 topo 索引转换为无 alpha 版本
void convert_topo_remove_alpha(const std::string &input_meta_file,
                                const std::string &input_topo_file,
                                const std::string &output_meta_file,
                                const std::string &output_topo_file,
                                const std::string &output_alpha_range_file = "") {
  const size_t PAGE_SIZE = 8192;

  // ==========================================
  // 1. 读取原始 Meta 文件
  // ==========================================
  std::ifstream in_meta(input_meta_file, std::ios::binary);
  if (!in_meta.is_open()) {
    LOG(ERROR) << "Failed to open input meta file: " << input_meta_file;
    return;
  }

  // 读取原始 meta header (定长 PAGE_SIZE)
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
  uint32_t max_alpha_range_len = *((uint32_t *)m_ptr);
  m_ptr += sizeof(uint32_t);
  uint64_t old_nnodes_per_sector = *((uint64_t *)m_ptr);
  m_ptr += sizeof(uint64_t);
  uint32_t enterpoint_set_size = *((uint32_t *)m_ptr);
  m_ptr += sizeof(uint32_t);
  std::vector<uint32_t> enterpoint_set(enterpoint_set_size);
  std::memcpy(enterpoint_set.data(), m_ptr, sizeof(uint32_t) * enterpoint_set_size);

  LOG(INFO) << "Input meta: node_num=" << node_num << ", emb_dim=" << emb_dim
            << ", loc_dim=" << loc_dim << ", max_nbr_len=" << max_nbr_len
            << ", max_alpha_range_len=" << max_alpha_range_len
            << ", enterpoint_set_size=" << enterpoint_set_size;

  // ==========================================
  // 2. 计算新的布局参数（无 alpha）
  // ==========================================
  // 原始单邻居大小: [ID(4B)] + [Alpha Range(2 * max_alpha_range_len * int8_t)]
  uint32_t old_single_nbr_size = sizeof(uint32_t) + (2 * max_alpha_range_len * sizeof(int8_t));
  uint32_t old_fixed_topo_size = sizeof(uint32_t) + (max_nbr_len * old_single_nbr_size);

  // 新单邻居大小: 只有 [ID(4B)]
  uint32_t new_single_nbr_size = sizeof(uint32_t);
  uint32_t new_fixed_topo_size = sizeof(uint32_t) + (max_nbr_len * new_single_nbr_size);
  uint64_t new_nnodes_per_sector = PAGE_SIZE / new_fixed_topo_size;

  LOG(INFO) << "Old topo: single_nbr=" << old_single_nbr_size << "B, fixed_topo=" << old_fixed_topo_size
            << "B, nnodes/sector=" << old_nnodes_per_sector;
  LOG(INFO) << "New topo: single_nbr=" << new_single_nbr_size << "B, fixed_topo=" << new_fixed_topo_size
            << "B, nnodes/sector=" << new_nnodes_per_sector;

  // ==========================================
  // 3. 写入新的 Meta 文件（max_alpha_range_len = 0）
  // ==========================================
  std::remove(output_meta_file.c_str());
  std::ofstream out_meta(output_meta_file, std::ios::binary);
  size_t raw_meta_size =
      sizeof(node_num) + sizeof(emb_dim) + sizeof(loc_dim) +
      sizeof(max_nbr_len) + sizeof(uint32_t) +  // new max_alpha_range_len = 0
      sizeof(new_nnodes_per_sector) +
      sizeof(enterpoint_set_size) + (sizeof(uint32_t) * enterpoint_set_size);

  size_t aligned_meta_size = (raw_meta_size + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
  std::vector<char> new_meta_buffer(aligned_meta_size, 0);

  char *new_m_ptr = new_meta_buffer.data();
  auto copy_meta = [&](const void *src, size_t sz) {
    std::memcpy(new_m_ptr, src, sz);
    new_m_ptr += sz;
  };

  uint32_t new_max_alpha_range_len = 0;  // 关键变化：设为 0
  copy_meta(&node_num, sizeof(uint32_t));
  copy_meta(&emb_dim, sizeof(uint32_t));
  copy_meta(&loc_dim, sizeof(uint32_t));
  copy_meta(&max_nbr_len, sizeof(uint32_t));
  copy_meta(&new_max_alpha_range_len, sizeof(uint32_t));  // max_alpha_range_len = 0
  copy_meta(&new_nnodes_per_sector, sizeof(uint64_t));
  copy_meta(&enterpoint_set_size, sizeof(uint32_t));
  copy_meta(enterpoint_set.data(), sizeof(uint32_t) * enterpoint_set_size);

  out_meta.write(new_meta_buffer.data(), aligned_meta_size);
  out_meta.close();

  LOG(INFO) << "New meta file written: " << output_meta_file;

  // ==========================================
  // 3.5. 打开 Alpha Range 输出文件（可选）
  // ==========================================
  std::ofstream out_alpha;
  if (!output_alpha_range_file.empty()) {
    out_alpha.open(output_alpha_range_file, std::ios::binary);
    // 写 header: orig_max_alpha_range_len
    out_alpha.write((const char *)&max_alpha_range_len, sizeof(uint32_t));
    LOG(INFO) << "Writing alpha range file: " << output_alpha_range_file
              << " with max_alpha_range_len=" << max_alpha_range_len;
  }

  // ==========================================
  // 4. 读取原始 Topo 文件，转换并写入新 Topo 文件
  //
  // 注意：原始文件每扇区有 padding（old_nnodes_per_sector * old_fixed_topo_size <= PAGE_SIZE，
  // 剩余的字节是 padding）。必须按扇区读取，不能连续按节点读取，否则会读到 padding 产生垃圾数据。
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

  // 扇区缓冲区：按 PAGE_SIZE 读取原始文件
  std::vector<char> sector_buf(PAGE_SIZE, 0);
  // 每个节点的新拓扑缓冲区
  std::vector<char> new_node_buf(new_fixed_topo_size, 0);

  uint64_t nodes_read = 0;
  uint64_t nodes_per_sector_old = (uint64_t)old_nnodes_per_sector;

  while (in_topo.read(sector_buf.data(), PAGE_SIZE) ||
         in_topo.gcount() > 0) {
    std::streamsize bytes_read = in_topo.gcount();
    if (bytes_read < (std::streamsize)(sizeof(uint32_t))) {
      break;  // 不足以构成一个节点头，结束
    }

    // 从当前扇区中逐个提取节点（最多 old_nnodes_per_sector 个）
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

      // 写入 alpha range 文件 (如果启用)
      if (out_alpha.is_open()) {
        out_alpha.write((const char *)&actual_nbr_to_write, sizeof(uint32_t));
      }

      // 逐个邻居处理：跳过 alpha range，只取 ID
      for (uint32_t k = 0; k < neighbor_size && k < max_nbr_len; k++) {
        // 旧格式: [ID(4B)] + [Alpha Range]
        uint32_t nbr_id = *((uint32_t *)src_ptr);
        src_ptr += sizeof(uint32_t);

        // 导出 alpha range 到 alpha 文件
        if (out_alpha.is_open()) {
          out_alpha.write(src_ptr, 2 * max_alpha_range_len * sizeof(int8_t));
        }
        src_ptr += 2 * max_alpha_range_len * sizeof(int8_t);  // 跳过 alpha range

        // 新格式: 只有 [ID(4B)]
        *((uint32_t *)dst_ptr) = nbr_id;
        dst_ptr += sizeof(uint32_t);
      }

      // 预检：当前页剩余空间不足时，先落盘再写入
      if (page_off + new_fixed_topo_size > PAGE_SIZE) {
        std::memset(page_buf.data() + page_off, 0, PAGE_SIZE - page_off);
        out_topo.write(page_buf.data(), PAGE_SIZE);
        page_off = 0;
      }

      // 将新节点写入 page buffer
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
  if (out_alpha.is_open()) {
    out_alpha.close();
    LOG(INFO) << "Alpha range file written: " << output_alpha_range_file;
  }

  LOG(INFO) << "Conversion finished: " << nodes_read << " nodes converted";
  LOG(INFO) << "Output topo file: " << output_topo_file;
}

int main(int argc, char **argv) {
  if (argc < 5 || argc > 6) {
    std::cerr << "Usage: " << argv[0]
              << " <input_meta_file> <input_topo_file>"
              << " <output_meta_file> <output_topo_file>"
              << " [output_alpha_range_file]" << std::endl;
    return 1;
  }

  std::string input_meta_file = argv[1];
  std::string input_topo_file = argv[2];
  std::string output_meta_file = argv[3];
  std::string output_topo_file = argv[4];
  std::string output_alpha_file = (argc >= 6) ? argv[5] : "";

  convert_topo_remove_alpha(input_meta_file, input_topo_file,
                             output_meta_file, output_topo_file,
                             output_alpha_file);

  LOG(INFO) << "Alpha removal completed successfully!";
  return 0;
}
