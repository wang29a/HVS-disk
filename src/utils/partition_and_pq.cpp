#include <math_utils.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <cstring>   // for std::memcpy
#include "cached_io.h"
#include "index.h"
#include "log.h"
#include "utils.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
#include <tsl/robin_map.h>

#include <cassert>
#include "partition_and_pq.h"

#define MAX_BLOCK_SIZE 16384  // 64MB for 1024-dim float vectors, 2MB for 128-dim uint8 vectors.

template<typename T>
void gen_random_slice(const std::string base_file, const std::string output_prefix, double sampling_rate,
                      size_t offset) {
  std::ifstream base_reader(base_file.c_str());
  base_reader.seekg(offset, std::ios::beg);

  std::ofstream sample_writer(std::string(output_prefix + "_data.bin").c_str(), std::ios::binary);
  std::ofstream sample_id_writer(std::string(output_prefix + "_ids.bin").c_str(), std::ios::binary);

  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  auto x = rd();
  std::mt19937 generator(x);  // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<float> distribution(0, 1);

  size_t npts, nd;
  uint32_t npts_u32, nd_u32;
  uint32_t num_sampled_pts_u32 = 0;
  uint32_t one_const = 1;

  base_reader.read((char *) &npts_u32, sizeof(uint32_t));
  base_reader.read((char *) &nd_u32, sizeof(uint32_t));
  LOG(INFO) << "Loading base " << base_file << ". #points: " << npts_u32 << ". #dim: " << nd_u32 << ".";
  sample_writer.write((char *) &num_sampled_pts_u32, sizeof(uint32_t));
  sample_writer.write((char *) &nd_u32, sizeof(uint32_t));
  sample_id_writer.write((char *) &num_sampled_pts_u32, sizeof(uint32_t));
  sample_id_writer.write((char *) &one_const, sizeof(uint32_t));

  npts = npts_u32;
  nd = nd_u32;
  std::unique_ptr<T[]> cur_row = std::make_unique<T[]>(nd);

  for (size_t i = 0; i < npts; i++) {
    float sample = distribution(generator);
    if (sample < (float) sampling_rate) {
      base_reader.read((char *) cur_row.get(), sizeof(T) * nd);
      sample_writer.write((char *) cur_row.get(), sizeof(T) * nd);
      uint32_t cur_i_u32 = (_u32) i;
      sample_id_writer.write((char *) &cur_i_u32, sizeof(uint32_t));
      num_sampled_pts_u32++;
    } else {
      base_reader.seekg(sizeof(T) * nd, base_reader.cur);  // skip this vector
    }
  }

  if (num_sampled_pts_u32 == 0) {
    // We have read something from file, so write it.
    sample_writer.write((char *) cur_row.get(), sizeof(T) * nd);
    num_sampled_pts_u32 = 1;
  }
  sample_writer.seekp(0, std::ios::beg);
  sample_writer.write((char *) &num_sampled_pts_u32, sizeof(uint32_t));
  sample_id_writer.seekp(0, std::ios::beg);
  sample_id_writer.write((char *) &num_sampled_pts_u32, sizeof(uint32_t));
  sample_writer.close();
  sample_id_writer.close();
  LOG(INFO) << "Wrote " << num_sampled_pts_u32 << " points to sample file: " << output_prefix + "_data.bin";
}

// streams data from the file, and samples each vector with probability p_val
// and returns a matrix of size slice_size* ndims as floating point type.
// the slice_size and ndims are set inside the function.

template<typename T>
void gen_random_slice(const std::string data_file, double p_val, std::unique_ptr<float[]> &sampled_data,
                      size_t &slice_size, size_t &ndims) {
  float *sampled_ptr = sampled_data.get();
  gen_random_slice<T>(data_file, p_val, sampled_ptr, slice_size, ndims);
  sampled_data.reset(sampled_ptr);
}
template<typename T>
void gen_random_slice(const std::string data_file, double p_val, float *&sampled_data, size_t &slice_size,
                      size_t &ndims) {
  size_t npts;
  uint32_t npts32, ndims32;
  std::vector<std::vector<float>> sampled_vectors;

  // amount to read in one shot
  _u64 read_blk_size = 64 * 1024 * 1024;
  std::ifstream base_reader(data_file.c_str());

  // metadata: npts, ndims
  base_reader.read((char *) &npts32, sizeof(unsigned));
  base_reader.read((char *) &ndims32, sizeof(unsigned));
  npts = npts32;
  ndims = ndims32;

  std::unique_ptr<T[]> cur_vector_T = std::make_unique<T[]>(ndims);
  p_val = p_val < 1 ? p_val : 1;

  std::random_device rd;  // Will be used to obtain a seed for the random number
  size_t x = rd();
  std::mt19937 generator((unsigned) x);
  std::uniform_real_distribution<float> distribution(0, 1);

  for (size_t i = 0; i < npts; i++) {
    float rnd_val = distribution(generator);
    if (rnd_val < (float) p_val) {
      base_reader.read((char *) cur_vector_T.get(), ndims * sizeof(T));
      std::vector<float> cur_vector_float;
      for (size_t d = 0; d < ndims; d++)
        cur_vector_float.push_back(cur_vector_T[d]);
      sampled_vectors.push_back(cur_vector_float);
    } else {
      base_reader.seekg(ndims * sizeof(T), base_reader.cur);  // skip this vector
    }
  }
  slice_size = sampled_vectors.size();
  if (slice_size == 0) {
    slice_size = 1;
    std::vector<float> cur_vector_float(cur_vector_T.get(), cur_vector_T.get() + ndims);
    sampled_vectors.push_back(cur_vector_float);
  }
  sampled_data = new float[slice_size * ndims];

  for (size_t i = 0; i < slice_size; i++) {
    for (size_t j = 0; j < ndims; j++) {
      sampled_data[i * ndims + j] = sampled_vectors[i][j];
    }
  }
}

// given training data in train_data of dimensions num_train * dim, generate PQ
// pivots using k-means algorithm to partition the co-ordinates into
// num_pq_chunks (if it divides dimension, else rounded) chunks, and runs
// k-means in each chunk to compute the PQ pivots and stores in bin format in
// file pq_pivots_path as a s num_centers*dim floating point binary file
template<typename T>
int generate_pq_pivots(const std::unique_ptr<T[]> &passed_train_data, size_t num_train, unsigned dim,
                       unsigned num_centers, unsigned num_pq_chunks, unsigned max_k_means_reps,
                       std::string pq_pivots_path) {
  std::unique_ptr<float[]> train_float = std::make_unique<float[]>(num_train * (size_t) (dim));
  float *flt_ptr = train_float.get();
  T *T_ptr = passed_train_data.get();

  for (_u64 i = 0; i < num_train; i++) {
    for (_u64 j = 0; j < (_u64) dim; j++) {
      flt_ptr[i * (_u64) dim + j] = (float) T_ptr[i * (_u64) dim + j];
    }
  }
  if (generate_pq_pivots(flt_ptr, num_train, dim, num_centers, num_pq_chunks, max_k_means_reps, pq_pivots_path) != 0)
    return -1;
  return 0;
}

int generate_pq_pivots(const float *passed_train_data, size_t num_train, unsigned dim, unsigned num_centers,
                       unsigned num_pq_chunks, unsigned max_k_means_reps, std::string pq_pivots_path) {
  if (num_pq_chunks > dim) {
    LOG(ERROR) << " Error: number of chunks more than dimension";
    return -1;
  }

  std::unique_ptr<float[]> train_data = std::make_unique<float[]>(num_train * dim);
  std::memcpy(train_data.get(), passed_train_data, num_train * dim * sizeof(float));

  for (uint64_t i = 0; i < num_train; i++) {
    for (uint64_t j = 0; j < dim; j++) {
      if (passed_train_data[i * dim + j] != train_data[i * dim + j])
        LOG(ERROR) << "error in copy";
    }
  }

  std::unique_ptr<float[]> full_pivot_data;

  // Calculate centroid and center the training data
  std::unique_ptr<float[]> centroid = std::make_unique<float[]>(dim);
  for (uint64_t d = 0; d < dim; d++) {
    centroid[d] = 0;
    for (uint64_t p = 0; p < num_train; p++) {
      centroid[d] += train_data[p * dim + d];
    }
    centroid[d] /= (float) num_train;
  }

  //  std::memset(centroid, 0 , dim*sizeof(float));

  for (uint64_t d = 0; d < dim; d++) {
    for (uint64_t p = 0; p < num_train; p++) {
      train_data[p * dim + d] -= centroid[d];
    }
  }

  std::vector<uint32_t> rearrangement;
  std::vector<uint32_t> chunk_offsets;

  size_t low_val = (size_t) std::floor((double) dim / (double) num_pq_chunks);
  size_t high_val = (size_t) std::ceil((double) dim / (double) num_pq_chunks);
  size_t max_num_high = dim - (low_val * num_pq_chunks);
  size_t cur_num_high = 0;
  size_t cur_bin_threshold = high_val;

  std::vector<std::vector<uint32_t>> bin_to_dims(num_pq_chunks);
  tsl::robin_map<uint32_t, uint32_t> dim_to_bin;
  std::vector<float> bin_loads(num_pq_chunks, 0);

  // Process dimensions not inserted by previous loop
  for (uint32_t d = 0; d < dim; d++) {
    if (dim_to_bin.find(d) != dim_to_bin.end())
      continue;
    auto cur_best = num_pq_chunks + 1;
    float cur_best_load = std::numeric_limits<float>::max();
    for (uint32_t b = 0; b < num_pq_chunks; b++) {
      if (bin_loads[b] < cur_best_load && bin_to_dims[b].size() < cur_bin_threshold) {
        cur_best = b;
        cur_best_load = bin_loads[b];
      }
    }
    bin_to_dims[cur_best].push_back(d);
    if (bin_to_dims[cur_best].size() == high_val) {
      cur_num_high++;
      if (cur_num_high == max_num_high)
        cur_bin_threshold = low_val;
    }
  }

  rearrangement.clear();
  chunk_offsets.clear();
  chunk_offsets.push_back(0);

  for (uint32_t b = 0; b < num_pq_chunks; b++) {
    for (auto p : bin_to_dims[b]) {
      rearrangement.push_back(p);
    }
    if (b > 0)
      chunk_offsets.push_back(chunk_offsets[b - 1] + (unsigned) bin_to_dims[b - 1].size());
  }
  chunk_offsets.push_back(dim);

  full_pivot_data.reset(new float[num_centers * dim]);

  // DEBUG ONLY
  double kmeans_time = 0.0, lloyds_time = 0.0, copy_time = 0.0;

  for (size_t i = 0; i < num_pq_chunks; i++) {
    size_t cur_chunk_size = chunk_offsets[i + 1] - chunk_offsets[i];

    if (cur_chunk_size == 0)
      continue;
    std::unique_ptr<float[]> cur_pivot_data = std::make_unique<float[]>(num_centers * cur_chunk_size);
    std::unique_ptr<float[]> cur_data = std::make_unique<float[]>(num_train * cur_chunk_size);
    std::unique_ptr<uint32_t[]> closest_center = std::make_unique<uint32_t[]>(num_train);

    memset((void *) cur_pivot_data.get(), 0, num_centers * cur_chunk_size * sizeof(float));

    auto start = std::chrono::high_resolution_clock::now();
    // LOG(INFO) << "thread: " << omp_get_max_threads();
#pragma omp parallel for schedule(static, 65536)
    for (int64_t j = 0; j < (_s64) num_train; j++) {
      std::memcpy(cur_data.get() + j * cur_chunk_size, train_data.get() + j * dim + chunk_offsets[i],
                  cur_chunk_size * sizeof(float));
    }
    auto end = std::chrono::high_resolution_clock::now();
    copy_time += std::chrono::duration<double>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    // kmeans::kmeanspp_selecting_pivots(cur_data.get(), num_train,
    // cur_chunk_size,
    //                                  cur_pivot_data.get(), num_centers);
    kmeans::selecting_pivots(cur_data.get(), num_train, cur_chunk_size, cur_pivot_data.get(), num_centers);

    unsigned k_means_reps = max_k_means_reps;

    kmeans::run_lloyds(cur_data.get(), num_train, cur_chunk_size, cur_pivot_data.get(), num_centers, k_means_reps,
                       nullptr, closest_center.get());
    end = std::chrono::high_resolution_clock::now();
    kmeans_time += std::chrono::duration<double>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    if (num_train > 2 * num_centers) {
      kmeans::run_lloyds(cur_data.get(), num_train, cur_chunk_size, cur_pivot_data.get(), num_centers, max_k_means_reps,
                         NULL, closest_center.get());
    }
    end = std::chrono::high_resolution_clock::now();
    lloyds_time += std::chrono::duration<double>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    for (uint64_t j = 0; j < num_centers; j++) {
      std::memcpy(full_pivot_data.get() + j * dim + chunk_offsets[i], cur_pivot_data.get() + j * cur_chunk_size,
                  cur_chunk_size * sizeof(float));
    }
    end = std::chrono::high_resolution_clock::now();
    copy_time += std::chrono::duration<double>(end - start).count();
  }
  LOG(INFO) << "Kmeans time: " << kmeans_time << " Lloyds time: " << lloyds_time << " Copy time: " << copy_time;

  std::vector<size_t> cumul_bytes(5, 0);
  cumul_bytes[0] = METADATA_SIZE;
  cumul_bytes[1] = cumul_bytes[0] + pipeann::save_bin<float>(pq_pivots_path.c_str(), full_pivot_data.get(),
                                                             (size_t) num_centers, dim, cumul_bytes[0]);
  cumul_bytes[2] = cumul_bytes[1] +
                   pipeann::save_bin<float>(pq_pivots_path.c_str(), centroid.get(), (size_t) dim, 1, cumul_bytes[1]);
  cumul_bytes[3] = cumul_bytes[2] + pipeann::save_bin<uint32_t>(pq_pivots_path.c_str(), rearrangement.data(),
                                                                rearrangement.size(), 1, cumul_bytes[2]);
  cumul_bytes[4] = cumul_bytes[3] + pipeann::save_bin<uint32_t>(pq_pivots_path.c_str(), chunk_offsets.data(),
                                                                chunk_offsets.size(), 1, cumul_bytes[3]);
  pipeann::save_bin<_u64>(pq_pivots_path.c_str(), cumul_bytes.data(), cumul_bytes.size(), 1, 0);

  LOG(INFO) << "Saved pq pivot data to " << pq_pivots_path << " of size " << cumul_bytes[cumul_bytes.size() - 1]
            << "B.";

  return 0;
}

template<typename T>
int generate_train_data(const std::string data_file, float *passed_train_data, size_t num_train, unsigned num_centers, unsigned num_pq_chunks, std::string pq_pivots_path) {
  _u64 read_blk_size = 64 * 1024 * 1024;
  cached_ifstream base_reader(data_file, read_blk_size, (uint32_t) 0);
  _u32 npts32;
  _u32 basedim32;
  base_reader.read((char *) &npts32, sizeof(uint32_t));
  base_reader.read((char *) &basedim32, sizeof(uint32_t));
  size_t num_points = npts32;
  size_t dim = basedim32;

  std::unique_ptr<float[]> full_pivot_data;
  std::unique_ptr<float[]> centroid;
  std::unique_ptr<uint32_t[]> rearrangement;
  std::unique_ptr<uint32_t[]> chunk_offsets;

  if (!file_exists(pq_pivots_path)) {
    LOG(INFO) << "ERROR: PQ k-means pivot file not found";
    crash();
  } else {
    _u64 nr, nc;
    std::unique_ptr<_u64[]> file_offset_data;

    pipeann::load_bin<_u64>(pq_pivots_path.c_str(), file_offset_data, nr, nc, 0);

    if (nr != 5) {
      LOG(INFO) << "Error reading pq_pivots file " << pq_pivots_path
                << ". Offsets dont contain correct metadata, # offsets = " << nr << ", but expecting 5.";
      crash();
    }

    pipeann::load_bin<float>(pq_pivots_path.c_str(), full_pivot_data, nr, nc, file_offset_data[0]);

    if ((nr != num_centers) || (nc != dim)) {
      LOG(INFO) << "Error reading pq_pivots file " << pq_pivots_path << ". file_num_centers  = " << nr
                << ", file_dim = " << nc << " but expecting " << num_centers << " centers in " << dim << " dimensions.";
      crash();
    }

    pipeann::load_bin<float>(pq_pivots_path.c_str(), centroid, nr, nc, file_offset_data[1]);

    if ((nr != dim) || (nc != 1)) {
      LOG(INFO) << "Error reading pq_pivots file " << pq_pivots_path << ". file_dim  = " << nr << ", file_cols = " << nc
                << " but expecting " << dim << " entries in 1 dimension.";
      crash();
    }

    pipeann::load_bin<uint32_t>(pq_pivots_path.c_str(), rearrangement, nr, nc, file_offset_data[2]);

    if ((nr != dim) || (nc != 1)) {
      LOG(INFO) << "Error reading pq_pivots file " << pq_pivots_path << ". file_dim  = " << nr << ", file_cols = " << nc
                << " but expecting " << dim << " entries in 1 dimension.";
      crash();
    }

    pipeann::load_bin<uint32_t>(pq_pivots_path.c_str(), chunk_offsets, nr, nc, file_offset_data[3]);

    if (nr != (uint64_t) num_pq_chunks + 1 || nc != 1) {
      LOG(INFO) << "Error reading pq_pivots file at chunk offsets; file has nr=" << nr << ",nc=" << nc
                << ", expecting nr=" << num_pq_chunks + 1 << ", nc=1.";
      crash();
    }

    LOG(INFO) << "Loaded PQ pivot information";
  }

  _u32 num_pq_chunks_u32 = num_pq_chunks;

  // LOG(ERROR) << "1";
  std::unique_ptr<T[]> all_data = std::make_unique<T[]>(num_points * dim);
  std::unique_ptr<float[]> all_data_tmp = std::make_unique<float[]>(num_points * dim);
  std::unique_ptr<float[]> all_data_float = std::make_unique<float[]>(num_points * dim);
  base_reader.read((char *) (all_data.get()), sizeof(T) * (num_points * dim)); //读所有向量
  pipeann::convert_types<T, float>(all_data.get(), all_data_tmp.get(), num_points, dim);
  // LOG(ERROR) << "1";

  for (uint64_t p = 0; p < num_points; p++) { //均值化
    for (uint64_t d = 0; d < dim; d++) {
      all_data_tmp[p * dim + d] -= centroid[d];
    }
  }
  // LOG(ERROR) << "1";

  for (uint64_t p = 0; p < num_points; p++) {// 根据rearrangement重排，但其实没有用，但最好还是加上
    for (uint64_t d = 0; d < dim; d++) {
      all_data_float[p * dim + d] = all_data_tmp[p * dim + rearrangement[d]];
    }
  }
  // LOG(ERROR) << "1";

  std::unique_ptr<uint32_t[]> closest_center = std::make_unique<uint32_t[]>(num_points);
  std::unique_ptr<float[]> dist_error = std::make_unique<float[]>(num_points);
  std::vector<std::pair<float, uint32_t>> dist_error_pair;
  // LOG(ERROR) << "1";

//之前的都是读入元数据
  for (size_t i = 0; i < num_pq_chunks; i++) { // 对于每个pq_chunk
    size_t cur_chunk_size = chunk_offsets[i + 1] - chunk_offsets[i];
    if (cur_chunk_size == 0)
      continue;

    std::unique_ptr<float[]> cur_pivot_data = std::make_unique<float[]>(num_centers * cur_chunk_size);
    std::unique_ptr<float[]> cur_data = std::make_unique<float[]>(num_points * cur_chunk_size);
  // LOG(ERROR) << "1";
    
#pragma omp parallel for schedule(static, 8192)
    for (int64_t j = 0; j < (_s64) num_points; j++) { // 对于block里的每个点，拿出对应子空间的向量
      for (uint64_t k = 0; k < cur_chunk_size; k++)
        cur_data[j * cur_chunk_size + k] = all_data_float[j * dim + chunk_offsets[i] + k];
    }
    // LOG(ERROR) << "1";

#pragma omp parallel for schedule(static, 1)
    for (int64_t j = 0; j < (_s64) num_centers; j++) { //拿出对应子空间的256个质心
      std::memcpy(cur_pivot_data.get() + j * cur_chunk_size, full_pivot_data.get() + j * dim + chunk_offsets[i],
                  cur_chunk_size * sizeof(float));
    }
    // LOG(ERROR) << "1";

    math_utils::compute_closest_centers(cur_data.get(), num_points, cur_chunk_size, cur_pivot_data.get(),num_centers, 1, closest_center.get()
                                                      , nullptr, nullptr, dist_error.get()); // 计算出最近的

    dist_error_pair.clear();
    for (int64_t j = 0; j < (_s64) num_points; j++) { // 对于block里的每个点，拿出对应子空间的向量
      dist_error_pair.emplace_back(dist_error[j], j);
    }
    std::sort(dist_error_pair.begin(), dist_error_pair.end(), [](const std::pair<float, uint32_t>& a, const std::pair<float, uint32_t>& b) {
      return a.first > b.first;
    });
    // LOG(ERROR) << "1";
    // std::cerr << "num_train " << num_train << std::endl;

    dist_error_pair.resize(num_train);
    for (uint j = 0; j < num_train; j++) {
      uint32_t id = dist_error_pair[j].second;
      // std::cout << id << " " << j * dim + chunk_offsets[i] << " " << id * dim + chunk_offsets[i] << std::endl;
      // std::cout << id << " " << dim << " " << chunk_offsets[i] << std::endl;
      std::memcpy(passed_train_data + j * dim + chunk_offsets[i], all_data_float.get() + id * dim + chunk_offsets[i], cur_chunk_size * sizeof(float));
    }
  // LOG(ERROR) << "1";

  }
  // LOG(ERROR) << "1";

  return 0;
}

// streams the base file (data_file), and computes the closest centers in each
// chunk to generate the compressed data_file and stores it in
// pq_compressed_vectors_path.
// If the numbber of centers is < 256, it stores as byte vector, else as 4-byte
// vector in binary format.
template<typename T>
int generate_pq_data_from_pivots(const std::string data_file, unsigned num_centers, unsigned num_pq_chunks,
                                 std::string pq_pivots_path, std::string pq_compressed_vectors_path, size_t offset) {
  _u64 read_blk_size = 64 * 1024 * 1024;
  cached_ifstream base_reader(data_file, read_blk_size, (uint32_t) offset);
  _u32 npts32;
  _u32 basedim32;
  base_reader.read((char *) &npts32, sizeof(uint32_t));
  base_reader.read((char *) &basedim32, sizeof(uint32_t));
  size_t num_points = npts32;
  size_t dim = basedim32;

#ifdef SAVE_INFLATED_PQ
  std::string inflated_pq_file = pq_compressed_vectors_path + "_full.bin";
#endif

  size_t BLOCK_SIZE = (std::min)((size_t) MAX_BLOCK_SIZE, num_points);

  std::unique_ptr<float[]> full_pivot_data;
  std::unique_ptr<float[]> centroid;
  std::unique_ptr<uint32_t[]> rearrangement;
  std::unique_ptr<uint32_t[]> chunk_offsets;

  if (!file_exists(pq_pivots_path)) {
    LOG(INFO) << "ERROR: PQ k-means pivot file not found";
    crash();
  } else {
    _u64 nr, nc;
    std::unique_ptr<_u64[]> file_offset_data;

    pipeann::load_bin<_u64>(pq_pivots_path.c_str(), file_offset_data, nr, nc, 0);

    if (nr != 5) {
      LOG(INFO) << "Error reading pq_pivots file " << pq_pivots_path
                << ". Offsets dont contain correct metadata, # offsets = " << nr << ", but expecting 5.";
      crash();
    }

    pipeann::load_bin<float>(pq_pivots_path.c_str(), full_pivot_data, nr, nc, file_offset_data[0]);

    if ((nr != num_centers) || (nc != dim)) {
      LOG(INFO) << "Error reading pq_pivots file " << pq_pivots_path << ". file_num_centers  = " << nr
                << ", file_dim = " << nc << " but expecting " << num_centers << " centers in " << dim << " dimensions.";
      crash();
    }

    pipeann::load_bin<float>(pq_pivots_path.c_str(), centroid, nr, nc, file_offset_data[1]);

    if ((nr != dim) || (nc != 1)) {
      LOG(INFO) << "Error reading pq_pivots file " << pq_pivots_path << ". file_dim  = " << nr << ", file_cols = " << nc
                << " but expecting " << dim << " entries in 1 dimension.";
      crash();
    }

    pipeann::load_bin<uint32_t>(pq_pivots_path.c_str(), rearrangement, nr, nc, file_offset_data[2]);

    if ((nr != dim) || (nc != 1)) {
      LOG(INFO) << "Error reading pq_pivots file " << pq_pivots_path << ". file_dim  = " << nr << ", file_cols = " << nc
                << " but expecting " << dim << " entries in 1 dimension.";
      crash();
    }

    pipeann::load_bin<uint32_t>(pq_pivots_path.c_str(), chunk_offsets, nr, nc, file_offset_data[3]);

    if (nr != (uint64_t) num_pq_chunks + 1 || nc != 1) {
      LOG(INFO) << "Error reading pq_pivots file at chunk offsets; file has nr=" << nr << ",nc=" << nc
                << ", expecting nr=" << num_pq_chunks + 1 << ", nc=1.";
      crash();
    }

    LOG(INFO) << "Loaded PQ pivot information";
  }

  std::ofstream compressed_file_writer(pq_compressed_vectors_path, std::ios::binary);
  _u32 num_pq_chunks_u32 = num_pq_chunks;

  compressed_file_writer.write((char *) &num_points, sizeof(uint32_t));
  compressed_file_writer.write((char *) &num_pq_chunks_u32, sizeof(uint32_t));

#ifdef SAVE_INFLATED_PQ
  std::ofstream inflated_file_writer(inflated_pq_file, std::ios::binary);
  inflated_file_writer.write((char *) &npts32, sizeof(uint32_t));
  inflated_file_writer.write((char *) &basedim32, sizeof(uint32_t));

  std::unique_ptr<float[]> block_inflated_base = std::make_unique<float[]>(BLOCK_SIZE * (_u64) dim);
  std::memset(block_inflated_base.get(), 0, BLOCK_SIZE * (_u64) dim * sizeof(float));
#endif

  size_t block_size = num_points <= BLOCK_SIZE ? num_points : BLOCK_SIZE;
  std::unique_ptr<_u32[]> block_compressed_base = std::make_unique<_u32[]>(block_size * (_u64) num_pq_chunks);
  std::memset(block_compressed_base.get(), 0, block_size * (_u64) num_pq_chunks * sizeof(uint32_t));

  std::unique_ptr<T[]> block_data_T = std::make_unique<T[]>(block_size * dim);
  std::unique_ptr<float[]> block_data_float = std::make_unique<float[]>(block_size * dim);
  std::unique_ptr<float[]> block_data_tmp = std::make_unique<float[]>(block_size * dim);

  size_t num_blocks = DIV_ROUND_UP(num_points, block_size);

  for (size_t block = 0; block < num_blocks; block++) {
    size_t start_id = block * block_size;
    size_t end_id = (std::min)((block + 1) * block_size, num_points);
    size_t cur_blk_size = end_id - start_id;

    base_reader.read((char *) (block_data_T.get()), sizeof(T) * (cur_blk_size * dim));
    pipeann::convert_types<T, float>(block_data_T.get(), block_data_tmp.get(), cur_blk_size, dim);

    for (uint64_t p = 0; p < cur_blk_size; p++) {
      for (uint64_t d = 0; d < dim; d++) {
        block_data_tmp[p * dim + d] -= centroid[d];
      }
    }

    for (uint64_t p = 0; p < cur_blk_size; p++) {
      for (uint64_t d = 0; d < dim; d++) {
        block_data_float[p * dim + d] = block_data_tmp[p * dim + rearrangement[d]];
      }
    }

    for (size_t i = 0; i < num_pq_chunks; i++) {
      size_t cur_chunk_size = chunk_offsets[i + 1] - chunk_offsets[i];
      if (cur_chunk_size == 0)
        continue;

      std::unique_ptr<float[]> cur_pivot_data = std::make_unique<float[]>(num_centers * cur_chunk_size);
      std::unique_ptr<float[]> cur_data = std::make_unique<float[]>(cur_blk_size * cur_chunk_size);
      std::unique_ptr<uint32_t[]> closest_center = std::make_unique<uint32_t[]>(cur_blk_size);

    // LOG(INFO) << "thread: " << omp_get_thread_num();
#pragma omp parallel for schedule(static, 8192)
      for (int64_t j = 0; j < (_s64) cur_blk_size; j++) {
        for (uint64_t k = 0; k < cur_chunk_size; k++)
          cur_data[j * cur_chunk_size + k] = block_data_float[j * dim + chunk_offsets[i] + k];
      }

#pragma omp parallel for schedule(static, 1)
      for (int64_t j = 0; j < (_s64) num_centers; j++) {
        std::memcpy(cur_pivot_data.get() + j * cur_chunk_size, full_pivot_data.get() + j * dim + chunk_offsets[i],
                    cur_chunk_size * sizeof(float));
      }

      math_utils::compute_closest_centers(cur_data.get(), cur_blk_size, cur_chunk_size, cur_pivot_data.get(),
                                          num_centers, 1, closest_center.get());
#pragma omp parallel for schedule(static, 8192)
      for (int64_t j = 0; j < (_s64) cur_blk_size; j++) {
        block_compressed_base[j * num_pq_chunks + i] = closest_center[j];
#ifdef SAVE_INFLATED_PQ
        for (uint64_t k = 0; k < cur_chunk_size; k++)
          block_inflated_base[j * dim + chunk_offsets[i] + k] =
              cur_pivot_data[closest_center[j] * cur_chunk_size + k] + centroid[chunk_offsets[i] + k];
#endif
      }
    }

#ifdef SAVE_INFLATED_PQ
    inflated_file_writer.write((char *) block_inflated_base.get(), cur_blk_size * dim * sizeof(float));
#endif

    if (num_centers > 256) {
      compressed_file_writer.write((char *) (block_compressed_base.get()),
                                   cur_blk_size * num_pq_chunks * sizeof(uint32_t));
    } else {
      std::unique_ptr<uint8_t[]> pVec = std::make_unique<uint8_t[]>(cur_blk_size * num_pq_chunks);
      pipeann::convert_types<uint32_t, uint8_t>(block_compressed_base.get(), pVec.get(), cur_blk_size, num_pq_chunks);
      compressed_file_writer.write((char *) (pVec.get()), cur_blk_size * num_pq_chunks * sizeof(uint8_t));
    }
    // LOG(INFO) << ".done.";
  }
  // Splittng diskann_dll into separate DLLs for search and build.
  // This code should only be available in the "build" DLL.
  compressed_file_writer.close();
#ifdef SAVE_INFLATED_PQ
  inflated_file_writer.close();
#endif
  return 0;
}

template<typename T>
int refine_pq_pivots_from_errors(
    const std::string data_file, const std::string initial_pq_pivots_path, const std::string refined_pq_pivots_path,
    const std::string refined_pq_compressed_vectors_path, unsigned num_centers, /* 256 */
    unsigned num_pq_chunks, unsigned max_k_means_reps, double error_percentage, /* 0.20 */
    size_t offset) {
    _u64 read_blk_size = 64 * 1024 * 1024;
    cached_ifstream base_reader(data_file, read_blk_size, (uint32_t) offset);
    _u32 npts32;
    _u32 basedim32;
    base_reader.read((char *) &npts32, sizeof(uint32_t));
    base_reader.read((char *) &basedim32, sizeof(uint32_t));
    size_t num_points = npts32;
    size_t dim = basedim32;

    LOG(INFO) << "Starting PQ refinement (512 -> 256 strategy). Loading initial pivots...";

    // 1. 加载 256 个 "旧" 质心 (C_old) 和元数据
    std::unique_ptr<float[]> full_pivot_data_old; // 256 * dim
    std::unique_ptr<float[]> centroid;
    std::vector<uint32_t>  rearrangement;
    std::vector<uint32_t>  chunk_offsets;
    _u64                   nr_old, nc_old;

    if (!file_exists(initial_pq_pivots_path)) {
        LOG(ERROR) << "Initial PQ pivot file not found: " << initial_pq_pivots_path;
        return -1;
    } else {
        std::unique_ptr<_u64[]> file_offset_data;
        pipeann::load_bin<_u64>(initial_pq_pivots_path.c_str(), file_offset_data, nr_old, nc_old, 0);
        if (nr_old != 5) {
            LOG(ERROR) << "Initial pivot file metadata incorrect.";
            return -1;
        }

        // 加载 C_old
        pipeann::load_bin<float>(initial_pq_pivots_path.c_str(), full_pivot_data_old, nr_old, nc_old, file_offset_data[0]);
        if (nr_old != num_centers || nc_old != dim) {
            LOG(ERROR) << "Initial pivot file dimension mismatch.";
            return -1;
        }

        // 加载元数据
        std::unique_ptr<float[]>    centroid_ptr;
        std::unique_ptr<uint32_t[]> rearrangement_ptr;
        std::unique_ptr<uint32_t[]> chunk_offsets_ptr;
        _u64                        nr_meta, nc_meta;

        pipeann::load_bin<float>(initial_pq_pivots_path.c_str(), centroid_ptr, nr_meta, nc_meta, file_offset_data[1]);
        centroid.reset(centroid_ptr.release()); // 转移所有权

        pipeann::load_bin<uint32_t>(initial_pq_pivots_path.c_str(), rearrangement_ptr, nr_meta, nc_meta, file_offset_data[2]);
        rearrangement.assign(rearrangement_ptr.get(), rearrangement_ptr.get() + nr_meta);

        pipeann::load_bin<uint32_t>(initial_pq_pivots_path.c_str(), chunk_offsets_ptr, nr_meta, nc_meta, file_offset_data[3]);
        chunk_offsets.assign(chunk_offsets_ptr.get(), chunk_offsets_ptr.get() + nr_meta);

        if (chunk_offsets.size() != (uint64_t) num_pq_chunks + 1) {
            LOG(ERROR) << "Chunk offsets mismatch.";
            return -1;
        }
        LOG(INFO) << "Loaded " << num_centers << " initial pivots and metadata.";
    }

    // 2. 加载所有数据点
    std::unique_ptr<T[]>     all_data_T     = std::make_unique<T[]>(num_points * dim);
    std::unique_ptr<float[]> all_data_tmp   = std::make_unique<float[]>(num_points * dim);
    std::unique_ptr<float[]> all_data_float = std::make_unique<float[]>(num_points * dim); // 预处理后的数据
    base_reader.read((char *) (all_data_T.get()), sizeof(T) * (num_points * dim));
    pipeann::convert_types<T, float>(all_data_T.get(), all_data_tmp.get(), num_points, dim);
    all_data_T.reset(); // 释放内存

    // 预处理: 中心化和重排
    for (uint64_t p = 0; p < num_points; p++) {
        for (uint64_t d = 0; d < dim; d++) {
            all_data_tmp[p * dim + d] -= centroid[d];
        }
    }
    for (uint64_t p = 0; p < num_points; p++) {
        for (uint64_t d = 0; d < dim; d++) {
            all_data_float[p * dim + d] = all_data_tmp[p * dim + rearrangement[d]];
        }
    }
    all_data_tmp.reset(); // 释放内存
    LOG(INFO) << "Loaded and pre-processed " << num_points << " data points.";

    // 3. 准备存储最终的 256 个质心
    std::unique_ptr<float[]> full_pivot_data_final = std::make_unique<float[]>(num_centers * dim);
    size_t num_error_points = (size_t)(num_points * error_percentage);
    if (num_error_points == 0)
        num_error_points = 1;
    if (num_error_points > num_points)
        num_error_points = num_points;

    // 4. 按 chunk 独立执行细化
    for (size_t i = 0; i < num_pq_chunks; i++) {
        size_t cur_chunk_size = chunk_offsets[i + 1] - chunk_offsets[i];
        if (cur_chunk_size == 0)
            continue;

        LOG(INFO) << "Processing chunk " << i << "/" << num_pq_chunks << " (dim: " << cur_chunk_size << ")";

        // 4a. 提取当前 chunk 的所有数据
        std::unique_ptr<float[]> cur_data = std::make_unique<float[]>(num_points * cur_chunk_size);
#pragma omp parallel for schedule(static, 8192)
        for (int64_t j = 0; j < (_s64) num_points; j++) {
            std::memcpy(cur_data.get() + j * cur_chunk_size, all_data_float.get() + j * dim + chunk_offsets[i],
                        cur_chunk_size * sizeof(float));
        }

        // 4b. 提取 C_old (256)
        std::unique_ptr<float[]> cur_pivot_data_old = std::make_unique<float[]>(num_centers * cur_chunk_size);
#pragma omp parallel for schedule(static, 1)
        for (int64_t j = 0; j < (_s64) num_centers; j++) {
            std::memcpy(cur_pivot_data_old.get() + j * cur_chunk_size,
                        full_pivot_data_old.get() + j * dim + chunk_offsets[i], cur_chunk_size * sizeof(float));
        }

        // 4c. 找到 20% 误差最高的点
        std::unique_ptr<float[]> dist_error       = std::make_unique<float[]>(num_points);
        std::unique_ptr<uint32_t[]> closest_center = std::make_unique<uint32_t[]>(num_points);
        math_utils::compute_closest_centers(cur_data.get(), num_points, cur_chunk_size, cur_pivot_data_old.get(),
                                            num_centers, 1, closest_center.get(), nullptr, nullptr, dist_error.get());

        std::vector<std::pair<float, uint32_t>> dist_error_pair;
        dist_error_pair.reserve(num_points);
        for (int64_t j = 0; j < (_s64) num_points; j++) {
            dist_error_pair.emplace_back(dist_error[j], j);
        }
        std::sort(dist_error_pair.begin(), dist_error_pair.end(),
                  [](const std::pair<float, uint32_t> &a, const std::pair<float, uint32_t> &b) {
                      return a.first > b.first; // 降序排列
                  });

        // 提取 20% 误差点的数据
        std::unique_ptr<float[]> cur_error_data = std::make_unique<float[]>(num_error_points * cur_chunk_size);
        for (uint j = 0; j < num_error_points; j++) {
            uint32_t id = dist_error_pair[j].second;
            std::memcpy(cur_error_data.get() + j * cur_chunk_size, cur_data.get() + id * cur_chunk_size,
                        cur_chunk_size * sizeof(float));
        }
        dist_error.reset();
        closest_center.reset();
        dist_error_pair.clear();
        LOG(INFO) << "  - Isolated " << num_error_points << " high-error points for chunk " << i;

        // 4d. 训练 C_new (256)
        std::unique_ptr<float[]> cur_pivot_data_new = std::make_unique<float[]>(num_centers * cur_chunk_size);
        kmeans::selecting_pivots(cur_error_data.get(), num_error_points, cur_chunk_size, cur_pivot_data_new.get(),
                                 num_centers);
        kmeans::run_lloyds(cur_error_data.get(), num_error_points, cur_chunk_size, cur_pivot_data_new.get(),
                           num_centers, max_k_means_reps, nullptr, nullptr);
        cur_error_data.reset(); // 释放内存
        LOG(INFO) << "  - Trained 256 'new' pivots for chunk " << i;
        
        // 4e. 合并 C_old 和 C_new 为 512 个种子点
        unsigned combined_num_centers = num_centers * 2; // 512
        std::unique_ptr<float[]> cur_pivot_data_combined =
            std::make_unique<float[]>(combined_num_centers * cur_chunk_size);
        
        std::memcpy(cur_pivot_data_combined.get(), cur_pivot_data_old.get(), num_centers * cur_chunk_size * sizeof(float));
        std::memcpy(cur_pivot_data_combined.get() + (num_centers * cur_chunk_size), cur_pivot_data_new.get(),
                    num_centers * cur_chunk_size * sizeof(float));
        
        cur_pivot_data_old.reset();
        cur_pivot_data_new.reset();

        // 4f. **关键步骤**: 以 512 为种子, 聚类得到 C_final (256)
        std::unique_ptr<float[]> cur_pivot_data_final = std::make_unique<float[]>(num_centers * cur_chunk_size);

        // 我们需要从 512 个种子中选出 256 个最好的作为 Lloyd's 迭代的起点
        // 这里我们简单地使用 K-means++ 从 512 个点中选 256 个
        kmeans::kmeanspp_selecting_pivots(cur_pivot_data_combined.get(), combined_num_centers, cur_chunk_size,
                                          cur_pivot_data_final.get(), num_centers);

        // 现在, 使用所有数据点, 用这 256 个种子点运行 Lloyd's 迭代
        kmeans::run_lloyds(cur_data.get(), num_points, cur_chunk_size, cur_pivot_data_final.get(), num_centers,
                           max_k_means_reps, nullptr, nullptr);
        
        LOG(INFO) << "  - Refined 512 seeds into 256 'final' pivots for chunk " << i;
        cur_data.reset(); // 释放内存
        cur_pivot_data_combined.reset();

        // 4g. 将 C_final (256) 复制到最终的码本中
        for (uint64_t j = 0; j < num_centers; j++) {
            std::memcpy(full_pivot_data_final.get() + j * dim + chunk_offsets[i],
                        cur_pivot_data_final.get() + j * cur_chunk_size, cur_chunk_size * sizeof(float));
        }
    }

    all_data_float.reset(); // 释放所有数据
    full_pivot_data_old.reset();
    LOG(INFO) << "All chunks refined. Saving final 256-center codebook...";

    // 5. 保存新的 256 质心码本 (C_final)
    std::vector<size_t> cumul_bytes(5, 0);
    cumul_bytes[0] = METADATA_SIZE;
    cumul_bytes[1] = cumul_bytes[0] + pipeann::save_bin<float>(refined_pq_pivots_path.c_str(), full_pivot_data_final.get(),
                                                              (size_t) num_centers, dim, cumul_bytes[0]);
    cumul_bytes[2] = cumul_bytes[1] +
                     pipeann::save_bin<float>(refined_pq_pivots_path.c_str(), centroid.get(), (size_t) dim, 1, cumul_bytes[1]);
    cumul_bytes[3] = cumul_bytes[2] + pipeann::save_bin<uint32_t>(refined_pq_pivots_path.c_str(), rearrangement.data(),
                                                                 rearrangement.size(), 1, cumul_bytes[2]);
    cumul_bytes[4] = cumul_bytes[3] + pipeann::save_bin<uint32_t>(refined_pq_pivots_path.c_str(), chunk_offsets.data(),
                                                                 chunk_offsets.size(), 1, cumul_bytes[3]);
    pipeann::save_bin<_u64>(refined_pq_pivots_path.c_str(), cumul_bytes.data(), cumul_bytes.size(), 1, 0);

    LOG(INFO) << "Saved refined 256-center pivots to " << refined_pq_pivots_path;

    // 6. 使用 C_final (256) 重新编码所有数据
    LOG(INFO) << "Re-encoding all data using refined 256-center pivots...";
    return generate_pq_data_from_pivots<T>(data_file, num_centers, num_pq_chunks, refined_pq_pivots_path,
                                           refined_pq_compressed_vectors_path, offset);
}

template<typename T>
int refine_pq_pivots_from_errors_toppercent_in_one_file(const std::string data_file,
                                          const std::string initial_pq_pivots_path,
                                          const std::string refined_pq_pivots_path,
                                          const std::string map_out_path,
                                          const std::string refined_pq_compressed_vectors_path, // 输出将是唯一的混合 code
                                          unsigned num_centers,    // e.g. 256
                                          unsigned num_pq_chunks,
                                          unsigned max_k_means_reps,
                                          double error_percentage, // 0.20 for top 20%
                                          size_t offset) {
  LOG(INFO) << "[refine_pq_toppercent] start: data=" << data_file
            << " init_pivots=" << initial_pq_pivots_path
            << " refined_pivots_out=" << refined_pq_pivots_path
            << " map_out=" << map_out_path
            << " refined_compressed_out=" << refined_pq_compressed_vectors_path
            << " centers=" << num_centers
            << " chunks=" << num_pq_chunks
            << " error_pct=" << error_percentage;

  if (error_percentage <= 0.0 || error_percentage >= 1.0) {
    LOG(ERROR) << "error_percentage must be in (0,1). got " << error_percentage;
    return -1;
  }

  // 1) read data header (不变)
  _u64 read_blk_size = 64ull * 1024ull * 1024ull;
  cached_ifstream base_reader(data_file, read_blk_size, (uint32_t) offset);
  _u32 npts32 = 0, nd32 = 0;
  base_reader.read((char *)&npts32, sizeof(uint32_t));
  base_reader.read((char *)&nd32, sizeof(uint32_t));
  size_t N = (size_t) npts32;
  size_t dim = (size_t) nd32;
  if (N == 0 || dim == 0) {
    LOG(ERROR) << "Bad data file metadata.";
    return -1;
  }
  LOG(INFO) << "Dataset: N=" << N << " dim=" << dim;

  // 2) load original pivots... (不变)
  std::unique_ptr<float[]> orig_pivots; // num_centers x dim
  std::unique_ptr<float[]> centroid;
  std::vector<uint32_t> rearrangement;
  std::vector<uint32_t> chunk_offsets;
  {
    _u64 nr=0,nc=0;
    std::unique_ptr<_u64[]> file_offsets;
    pipeann::load_bin<_u64>(initial_pq_pivots_path.c_str(), file_offsets, nr, nc, 0);
    if (nr != 5) {
      LOG(ERROR) << "pivots file offsets != 5";
      return -1;
    }
    pipeann::load_bin<float>(initial_pq_pivots_path.c_str(), orig_pivots, nr, nc, file_offsets[0]);
    if ((size_t)nr != num_centers || (size_t)nc != dim) {
      LOG(ERROR) << "orig_pivots shape mismatch.";
      return -1;
    }
    pipeann::load_bin<float>(initial_pq_pivots_path.c_str(), centroid, nr, nc, file_offsets[1]);
    if ((size_t)nr != dim || (size_t)nc != 1) {
      LOG(ERROR) << "centroid shape mismatch.";
      return -1;
    }
    std::unique_ptr<uint32_t[]> rearr_ptr;
    pipeann::load_bin<uint32_t>(initial_pq_pivots_path.c_str(), rearr_ptr, nr, nc, file_offsets[2]);
    rearrangement.assign(rearr_ptr.get(), rearr_ptr.get() + dim);
    std::unique_ptr<uint32_t[]> chunk_ptr;
    pipeann::load_bin<uint32_t>(initial_pq_pivots_path.c_str(), chunk_ptr, nr, nc, file_offsets[3]);
    chunk_offsets.assign(chunk_ptr.get(), chunk_ptr.get() + (num_pq_chunks + 1));
  }

  // 3) read all vectors -> convert to float -> center -> rearrange (不变)
  std::unique_ptr<T[]> all_T = std::make_unique<T[]>(N * dim);
  base_reader.read((char *)all_T.get(), sizeof(T) * (N * dim));
  base_reader.close();

  std::unique_ptr<float[]> all_tmp = std::make_unique<float[]>(N * dim);
  std::unique_ptr<float[]> all_f = std::make_unique<float[]>(N * dim);
  pipeann::convert_types<T, float>(all_T.get(), all_tmp.get(), N, dim);
  all_T.reset();

// (这些预处理步骤仍然可以使用并行)
#pragma omp parallel for schedule(static, 8192)
  for (int64_t i=0;i<(_s64)N;i++){
    for (uint64_t d=0; d<dim; d++){
      all_tmp[i*dim + d] -= centroid[d];
    }
  }

#pragma omp parallel for schedule(static, 8192)
  for (int64_t i=0;i<(_s64)N;i++){
    for (uint64_t d=0; d<dim; d++){
      all_f[i*dim + d] = all_tmp[i*dim + rearrangement[d]];
    }
  }
  all_tmp.reset();

  // 4) prepare refined_pivots (不变)
  std::unique_ptr<float[]> refined_pivots = std::make_unique<float[]>((size_t)num_centers * dim);
  std::memcpy(refined_pivots.get(), orig_pivots.get(), (size_t)num_centers * dim * sizeof(float));

  // 5) prepare map (不变)
  std::vector<uint8_t> map_bytes;
  map_bytes.assign(N * (size_t) num_pq_chunks, 0);

  // *NEW*: 准备 混合 PQ code 缓冲区 (不变)
  std::unique_ptr<uint8_t[]> hybrid_compressed_vectors = std::make_unique<uint8_t[]>(N * (size_t)num_pq_chunks);


  // 6) per-chunk processing
  // *MODIFIED*: 在 chunk 粒度上并行化
  // 使用 dynamic 调度以实现负载均衡，以防某些 chunk 的 k-means 收敛得更慢
#pragma omp parallel for schedule(dynamic, 1) num_threads(16)
  for (int c=0; c < (int)num_pq_chunks; ++c) {
    size_t s = (size_t)chunk_offsets[c];
    size_t e = (size_t)chunk_offsets[c+1];
    size_t csz = e - s;
    if (csz == 0) {
      LOG(INFO) << "chunk " << c << " size 0 skip";
      continue; // (在 OMP for 循环中，用 continue)
    }
    LOG(INFO) << "Refine chunk " << c << " dims [" << s << "," << e << ") size=" << csz << " (Thread " << omp_get_thread_num() << ")";

    // cur_data: N x csz
    std::unique_ptr<float[]> cur_data = std::make_unique<float[]>((size_t)N * csz);
    // *MODIFIED*: 移除内部的 #pragma omp
    for (int64_t i=0;i<(_s64)N;i++){
      std::memcpy(cur_data.get() + (size_t)i * csz, all_f.get() + (size_t)i * dim + s, csz * sizeof(float));
    }

    // extract orig pivots for this chunk: shape num_centers x csz
    std::unique_ptr<float[]> orig_chunk_pivots = std::make_unique<float[]>((size_t)num_centers * csz);
    // *MODIFIED*: 移除内部的 #pragma omp
    for (int64_t k=0;k<(_s64)num_centers;k++){
      std::memcpy(orig_chunk_pivots.get() + (size_t)k * csz, orig_pivots.get() + (size_t)k * dim + s, csz * sizeof(float));
    }

    // compute closest centers & dist_error_old (这是耗时操作 1)
    std::unique_ptr<uint32_t[]> closest_old = std::make_unique<uint32_t[]>((size_t)N);
    std::unique_ptr<float[]> dist_old = std::make_unique<float[]>((size_t)N);
    math_utils::compute_closest_centers(cur_data.get(), N, csz, orig_chunk_pivots.get(), num_centers, 1,
                                        closest_old.get(), nullptr, nullptr, dist_old.get());

    // pick top error_percentage points
    size_t num_sel = (size_t) std::max<size_t>(1, (size_t) std::floor((double)N * error_percentage));
    num_sel = std::min(num_sel, (size_t) 256000);
    if (num_sel > N) num_sel = N;
    LOG(INFO) << "Chunk " << c << " selecting top " << num_sel << " points (of " << N << ")";

    // use pair vector + nth_element to find top num_sel
    std::vector<std::pair<float, uint32_t>> errs;
    errs.reserve(N);
    for (uint32_t i=0;i<(uint32_t)N;i++) errs.emplace_back(dist_old[i], i);
    std::nth_element(errs.begin(), errs.begin() + num_sel, errs.end(),
                     [](const std::pair<float,uint32_t>& a, const std::pair<float,uint32_t>& b){
                       return a.first > b.first; // descending
                     });
    errs.resize(num_sel);

    // build training set (num_sel x csz)
    std::unique_ptr<float[]> train = std::make_unique<float[]>((size_t)num_sel * csz);
    for (size_t i=0;i<num_sel;i++){
      uint32_t id = errs[i].second;
      std::memcpy(train.get() + i * csz, cur_data.get() + (size_t)id * csz, csz * sizeof(float));
    }

    // train new pivots (这是耗时操作 2)
    std::unique_ptr<float[]> new_pivots = std::make_unique<float[]>((size_t)num_centers * csz);
    kmeans::selecting_pivots(train.get(), num_sel, csz, new_pivots.get(), num_centers);
    kmeans::run_lloyds(train.get(), num_sel, csz, new_pivots.get(), num_centers, max_k_means_reps, nullptr, nullptr);

    // compute dist_new for all points (这是耗时操作 3)
    std::unique_ptr<uint32_t[]> closest_new = std::make_unique<uint32_t[]>((size_t)N);
    std::unique_ptr<float[]> dist_new = std::make_unique<float[]>((size_t)N);
    math_utils::compute_closest_centers(cur_data.get(), N, csz, new_pivots.get(), num_centers, 1,
                                        closest_new.get(), nullptr, nullptr, dist_new.get());

    // set map
    // *MODIFIED*: 移除内部的 #pragma omp
    for (int64_t i=0;i<(_s64)N;i++){
      size_t map_idx = (size_t)i * (size_t)num_pq_chunks + (size_t)c;
      if (dist_new[i] < dist_old[i]) {
        map_bytes[map_idx] = 1;
      } else {
        map_bytes[map_idx] = 0;
      }
    }

    // *NEW*: 填充 混合 PQ code
    // *MODIFIED*: 移除内部的 #pragma omp
    for (int64_t i = 0; i < (_s64)N; i++) {
        size_t map_idx = (size_t)i * (size_t)num_pq_chunks + (size_t)c;
        size_t code_idx = (size_t)i * (size_t)num_pq_chunks + (size_t)c;
        
        if (map_bytes[map_idx] == 1) {
            hybrid_compressed_vectors[code_idx] = (uint8_t)closest_new[i];
        } else {
            hybrid_compressed_vectors[code_idx] = (uint8_t)closest_old[i];
        }
    }

    // copy new pivots into refined_pivots
    // *MODIFIED*: 移除内部的 #pragma omp
    for (int64_t k=0;k<(_s64)num_centers;k++){
      std::memcpy(refined_pivots.get() + (size_t)k * dim + s, new_pivots.get() + (size_t)k * csz, csz * sizeof(float));
    }

    // free chunk-local buffers (scoped unique_ptrs handle this)
    LOG(INFO) << "Finished chunk " << c;
  } // end parallel per-chunk

  // 7) save refined pivots (不变)
  std::vector<size_t> cumul_bytes(5,0);
  cumul_bytes[0] = METADATA_SIZE;
  cumul_bytes[1] = cumul_bytes[0] + pipeann::save_bin<float>(refined_pq_pivots_path.c_str(), refined_pivots.get(), (size_t)num_centers, dim, cumul_bytes[0]);
  cumul_bytes[2] = cumul_bytes[1] + pipeann::save_bin<float>(refined_pq_pivots_path.c_str(), centroid.get(), (size_t)dim, 1, cumul_bytes[1]);
  cumul_bytes[3] = cumul_bytes[2] + pipeann::save_bin<uint32_t>(refined_pq_pivots_path.c_str(), rearrangement.data(), rearrangement.size(), 1, cumul_bytes[2]);
  cumul_bytes[4] = cumul_bytes[3] + pipeann::save_bin<uint32_t>(refined_pq_pivots_path.c_str(), chunk_offsets.data(), chunk_offsets.size(), 1, cumul_bytes[3]);
  pipeann::save_bin<_u64>(refined_pq_pivots_path.c_str(), cumul_bytes.data(), cumul_bytes.size(), 1, 0);
  LOG(INFO) << "Saved refined pivots to " << refined_pq_pivots_path;

  // 8) save map file (不变)
  {
    std::ofstream fout(map_out_path, std::ios::binary | std::ios::out);
    if (!fout) {
      LOG(ERROR) << "Cannot open map_out_path: " << map_out_path;
      return -1;
    }
    uint32_t N32 = (uint32_t)N;
    uint32_t C32 = (uint32_t)num_pq_chunks;
    fout.write((char *)&N32, sizeof(uint32_t));
    fout.write((char *)&C32, sizeof(uint32_t));
    fout.write((char *)map_bytes.data(), map_bytes.size() * sizeof(uint8_t));
    fout.close();
  }
  LOG(INFO) << "Saved map (" << map_bytes.size() << " bytes) to " << map_out_path;

  // 9) *REPLACED*: 保存 混合 PQ code 文件 (不变)
  LOG(INFO) << "Saving hybrid compressed vectors to " << refined_pq_compressed_vectors_path;
  {
      std::ofstream fout(refined_pq_compressed_vectors_path, std::ios::binary | std::ios::out);
      if (!fout) {
          LOG(ERROR) << "Cannot open refined_pq_compressed_vectors_path: " << refined_pq_compressed_vectors_path;
          return -1;
      }
      uint32_t N32 = (uint32_t)N;
      uint32_t C32 = (uint32_t)num_pq_chunks;
      fout.write((char *)&N32, sizeof(uint32_t)); // 写入 N
      fout.write((char *)&C32, sizeof(uint32_t)); // 写入 M (num_chunks)
      fout.write((char *)hybrid_compressed_vectors.get(), N * (size_t)num_pq_chunks * sizeof(uint8_t));
      fout.close();
  }
  LOG(INFO) << "Saved hybrid compressed vectors (" << (N * (size_t)num_pq_chunks) << " bytes) to " << refined_pq_compressed_vectors_path;

  LOG(INFO) << "[refine_pq_toppercent] finished successfully.";
  return 0;
}
template<typename T>
int refine_pq_pivots_from_errors_toppercent(const std::string data_file,
                                            const std::string initial_pq_pivots_path,
                                            const std::string refined_pq_pivots_path,
                                            const std::string map_out_path,
                                            const std::string refined_pq_compressed_vectors_path,
                                            unsigned num_centers,      // e.g. 256
                                            unsigned num_pq_chunks,
                                            unsigned max_k_means_reps,
                                            double error_percentage,   // 0.20 for top 20%
                                            size_t offset) {
  LOG(INFO) << "[refine_pq_toppercent] start: data=" << data_file
            << " init_pivots=" << initial_pq_pivots_path
            << " refined_pivots_out=" << refined_pq_pivots_path
            << " map_out=" << map_out_path
            << " refined_compressed_out=" << refined_pq_compressed_vectors_path
            << " centers=" << num_centers
            << " chunks=" << num_pq_chunks
            << " error_pct=" << error_percentage;

  if (error_percentage <= 0.0 || error_percentage >= 1.0) {
    LOG(ERROR) << "error_percentage must be in (0,1). got " << error_percentage;
    return -1;
  }

  // 1) read data header
  _u64 read_blk_size = 64ull * 1024ull * 1024ull;
  {
    cached_ifstream base_reader(data_file, read_blk_size, (uint32_t) offset);
    _u32 npts32 = 0, nd32 = 0;
    base_reader.read((char *)&npts32, sizeof(uint32_t));
    base_reader.read((char *)&nd32, sizeof(uint32_t));
    size_t N = (size_t) npts32;
    size_t dim = (size_t) nd32;
    if (N == 0 || dim == 0) {
      LOG(ERROR) << "Bad data file metadata.";
      return -1;
    }
    LOG(INFO) << "Dataset: N=" << N << " dim=" << dim;
  }

  // 再读一遍 header，把 N, dim 存起来（方便后面用）
  _u32 npts32 = 0, nd32 = 0;
  {
    cached_ifstream base_reader(data_file, read_blk_size, (uint32_t) offset);
    base_reader.read((char *)&npts32, sizeof(uint32_t));
    base_reader.read((char *)&nd32, sizeof(uint32_t));
  }
  const size_t N   = (size_t)npts32;
  const size_t dim = (size_t)nd32;

  if (N == 0 || dim == 0) {
    LOG(ERROR) << "Bad data file metadata (second check).";
    return -1;
  }

  // 2) load original pivots/centroid/rearrangement/chunk_offsets
  std::unique_ptr<float[]> orig_pivots; // num_centers x dim
  std::unique_ptr<float[]> centroid;
  std::vector<uint32_t>    rearrangement;
  std::vector<uint32_t>    chunk_offsets;
  {
    _u64 nr=0,nc=0;
    std::unique_ptr<_u64[]> file_offsets;
    pipeann::load_bin<_u64>(initial_pq_pivots_path.c_str(), file_offsets, nr, nc, 0);
    if (nr != 5) {
      LOG(ERROR) << "pivots file offsets != 5";
      return -1;
    }

    // pivots
    pipeann::load_bin<float>(initial_pq_pivots_path.c_str(), orig_pivots, nr, nc, file_offsets[0]);
    if ((size_t)nr != num_centers || (size_t)nc != dim) {
      LOG(ERROR) << "orig_pivots shape mismatch. nr=" << nr << " nc=" << nc
                 << " expected (" << num_centers << ", " << dim << ")";
      return -1;
    }

    // centroid
    pipeann::load_bin<float>(initial_pq_pivots_path.c_str(), centroid, nr, nc, file_offsets[1]);
    if ((size_t)nr != dim || (size_t)nc != 1) {
      LOG(ERROR) << "centroid shape mismatch.";
      return -1;
    }

    // rearrangement
    std::unique_ptr<uint32_t[]> rearr_ptr;
    pipeann::load_bin<uint32_t>(initial_pq_pivots_path.c_str(), rearr_ptr, nr, nc, file_offsets[2]);
    if ((size_t)nr != dim) {
      LOG(ERROR) << "rearrangement length mismatch: got " << nr << " expected " << dim;
      return -1;
    }
    rearrangement.assign(rearr_ptr.get(), rearr_ptr.get() + dim);

    // chunk offsets (num_pq_chunks+1 rows)
    std::unique_ptr<uint32_t[]> chunk_ptr;
    pipeann::load_bin<uint32_t>(initial_pq_pivots_path.c_str(), chunk_ptr, nr, nc, file_offsets[3]);
    if ((size_t)nr != (size_t)(num_pq_chunks + 1)) {
      LOG(ERROR) << "chunk_offsets rows mismatch: got " << nr
                 << " expected " << (num_pq_chunks + 1);
      return -1;
    }
    chunk_offsets.assign(chunk_ptr.get(), chunk_ptr.get() + (num_pq_chunks + 1));
  }

  // 3) prepare refined_pivots (same shape: num_centers x dim) — we will replace per-chunk
  std::unique_ptr<float[]> refined_pivots =
      std::make_unique<float[]>((size_t)num_centers * dim);
  // initialize with original pivots (so chunks not retrained remain identical)
  std::memcpy(refined_pivots.get(), orig_pivots.get(),
              (size_t)num_centers * dim * sizeof(float));

  // 4) prepare map (N * num_pq_chunks), initially zero
  std::vector<uint8_t> map_bytes;
  map_bytes.assign((size_t)N * (size_t)num_pq_chunks, 0);

  // 为了流式构造 cur_data，用 batch 方式读取数据，避免 all_f
  const _u64 MAX_POINTS_PER_BATCH = 1000000; // 可按实际内存/IO 调整

  // 5) per-chunk processing
  for (unsigned c = 0; c < num_pq_chunks; ++c) {
    size_t s   = (size_t)chunk_offsets[c];
    size_t e   = (size_t)chunk_offsets[c+1];
    size_t csz = e - s;

    if (csz == 0) {
      LOG(INFO) << "chunk " << c << " size 0 skip";
      continue;
    }
    LOG(INFO) << "Refine chunk " << c << " dims [" << s << "," << e << ") size=" << csz;

    // ---------------------------------------------------------
    // 5.1) 构造本 chunk 的 cur_data: N x csz
    //      从 data_file 流式读取：T -> float -> 中心化 -> 重排 -> 取 [s,e)
    // ---------------------------------------------------------
    std::unique_ptr<float[]> cur_data =
        std::make_unique<float[]>((size_t)N * csz);

    {
      cached_ifstream base_reader(data_file, read_blk_size, (uint32_t) offset);
      _u32 npts32_r = 0, nd32_r = 0;
      base_reader.read((char *)&npts32_r, sizeof(uint32_t));
      base_reader.read((char *)&nd32_r, sizeof(uint32_t));
      if ((size_t)npts32_r != N || (size_t)nd32_r != dim) {
        LOG(ERROR) << "Header changed when re-reading data for chunk " << c;
        return -1;
      }

      const _u64 total_points = (_u64)N;
      const _u64 batch_points =
          std::min<_u64>(MAX_POINTS_PER_BATCH, total_points);

      std::unique_ptr<T[]>     batch_T   =
          std::make_unique<T[]>((size_t)batch_points * dim);
      std::unique_ptr<float[]> batch_tmp =
          std::make_unique<float[]>((size_t)batch_points * dim);
      std::unique_ptr<float[]> batch_f   =
          std::make_unique<float[]>((size_t)batch_points * dim);

      _u64 processed = 0;
      while (processed < total_points) {
        _u64 cur_batch_size =
            std::min<_u64>(batch_points, total_points - processed);

        // 读取原始数据
        base_reader.read((char *)batch_T.get(),
                         sizeof(T) * (size_t)cur_batch_size * dim);

        // 转 float
        pipeann::convert_types<T, float>(batch_T.get(), batch_tmp.get(),
                                         (size_t)cur_batch_size, dim);

        // 中心化 + 重排
#pragma omp parallel for schedule(static, 8192)
        for (int64_t i = 0; i < (_s64)cur_batch_size; i++) {
          float *src = batch_tmp.get() + (size_t)i * dim;
          float *dst = batch_f.get()   + (size_t)i * dim;

          // 中心化
          for (size_t d = 0; d < dim; d++) {
            src[d] -= centroid[d];
          }
          // 重排
          for (size_t d = 0; d < dim; d++) {
            dst[d] = src[rearrangement[d]];
          }
        }

        // 取本 chunk 对应的维度 [s,e)，写到 cur_data
#pragma omp parallel for schedule(static, 8192)
        for (int64_t i = 0; i < (_s64)cur_batch_size; i++) {
          size_t global_idx = (size_t)processed + (size_t)i;
          float *dst_chunk = cur_data.get() + global_idx * csz;
          float *src_full  = batch_f.get()   + (size_t)i * dim;
          std::memcpy(dst_chunk, src_full + s, csz * sizeof(float));
        }

        processed += cur_batch_size;
      } // while processed < total_points
    }   // 构造 cur_data 的作用域结束

    // ---------------------------------------------------------
    // 5.2) extract orig pivots for this chunk: shape num_centers x csz
    // ---------------------------------------------------------
    std::unique_ptr<float[]> orig_chunk_pivots =
        std::make_unique<float[]>((size_t)num_centers * csz);
#pragma omp parallel for schedule(static,1)
    for (int64_t k = 0; k < (_s64)num_centers; k++) {
      std::memcpy(orig_chunk_pivots.get() + (size_t)k * csz,
                  orig_pivots.get()       + (size_t)k * dim + s,
                  csz * sizeof(float));
    }

    // ---------------------------------------------------------
    // 5.3) compute closest centers & dist_error_old
    // ---------------------------------------------------------
    std::unique_ptr<uint32_t[]> closest_old =
        std::make_unique<uint32_t[]>((size_t)N);
    std::unique_ptr<float[]> dist_old =
        std::make_unique<float[]>((size_t)N);

    math_utils::compute_closest_centers(cur_data.get(), N, csz,
                                        orig_chunk_pivots.get(), num_centers, 1,
                                        closest_old.get(), nullptr, nullptr,
                                        dist_old.get());

    // ---------------------------------------------------------
    // 5.4) pick top error_percentage points (largest dist_old)
    // ---------------------------------------------------------
    size_t num_sel = (size_t)std::max<size_t>(
        1, (size_t)std::floor((double)N * error_percentage));
    if (num_sel > N) num_sel = N;
    LOG(INFO) << "Chunk " << c << " selecting top " << num_sel
              << " points (of " << N << ")";

    // 为了尽量少改逻辑，仍然用 pair<float, uint32_t>，只是 all_f 已移除，整体内存安全很多
    std::vector<std::pair<float, uint32_t>> errs;
    errs.reserve(N);
    for (uint32_t i = 0; i < (uint32_t)N; i++)
      errs.emplace_back(dist_old[i], i);

    std::nth_element(
        errs.begin(), errs.begin() + num_sel, errs.end(),
        [](const std::pair<float, uint32_t> &a,
           const std::pair<float, uint32_t> &b) {
          return a.first > b.first; // descending by error
        });
    errs.resize(num_sel);

    // build training set (num_sel x csz)
    std::unique_ptr<float[]> train =
        std::make_unique<float[]>((size_t)num_sel * csz);
    for (size_t i = 0; i < num_sel; i++) {
      uint32_t id = errs[i].second;
      std::memcpy(train.get() + i * csz,
                  cur_data.get() + (size_t)id * csz,
                  csz * sizeof(float));
    }

    // ---------------------------------------------------------
    // 5.5) train new pivots (num_centers x csz)
    // ---------------------------------------------------------
    std::unique_ptr<float[]> new_pivots =
        std::make_unique<float[]>((size_t)num_centers * csz);

    kmeans::selecting_pivots(train.get(), num_sel, csz,
                             new_pivots.get(), num_centers);
    kmeans::run_lloyds(train.get(), num_sel, csz,
                       new_pivots.get(), num_centers,
                       max_k_means_reps, nullptr, nullptr);

    // ---------------------------------------------------------
    // 5.6) compute dist_new for all points
    // ---------------------------------------------------------
    std::unique_ptr<uint32_t[]> closest_new =
        std::make_unique<uint32_t[]>((size_t)N);
    std::unique_ptr<float[]> dist_new =
        std::make_unique<float[]>((size_t)N);

    math_utils::compute_closest_centers(cur_data.get(), N, csz,
                                        new_pivots.get(), num_centers, 1,
                                        closest_new.get(), nullptr, nullptr,
                                        dist_new.get());

    // ---------------------------------------------------------
    // 5.7) set map: if dist_new < dist_old -> use refined pivots for this point/chunk
    // ---------------------------------------------------------
#pragma omp parallel for schedule(static, 8192)
    for (int64_t i = 0; i < (_s64)N; i++) {
      if (dist_new[i] < dist_old[i]) {
        map_bytes[(size_t)i * (size_t)num_pq_chunks + (size_t)c] = 1;
      } else {
        map_bytes[(size_t)i * (size_t)num_pq_chunks + (size_t)c] = 0;
      }
    }

    // ---------------------------------------------------------
    // 5.8) copy new pivots into refined_pivots at the correct dimension locations
    // ---------------------------------------------------------
#pragma omp parallel for schedule(static,1)
    for (int64_t k = 0; k < (_s64)num_centers; k++) {
      std::memcpy(refined_pivots.get() + (size_t)k * dim + s,
                  new_pivots.get()     + (size_t)k * csz,
                  csz * sizeof(float));
    }

    // chunk-local的大数组在这里出作用域，内存释放，然后处理下一个 chunk
  } // end per-chunk

  // 6) save refined pivots (same layout as original pivots file)
  {
    std::vector<size_t> cumul_bytes(5, 0);
    cumul_bytes[0] = METADATA_SIZE;
    cumul_bytes[1] =
        cumul_bytes[0] +
        pipeann::save_bin<float>(refined_pq_pivots_path.c_str(),
                                 refined_pivots.get(),
                                 (size_t)num_centers, dim,
                                 cumul_bytes[0]);
    cumul_bytes[2] =
        cumul_bytes[1] +
        pipeann::save_bin<float>(refined_pq_pivots_path.c_str(),
                                 centroid.get(),
                                 (size_t)dim, 1,
                                 cumul_bytes[1]);
    cumul_bytes[3] =
        cumul_bytes[2] +
        pipeann::save_bin<uint32_t>(refined_pq_pivots_path.c_str(),
                                    rearrangement.data(),
                                    rearrangement.size(), 1,
                                    cumul_bytes[2]);
    cumul_bytes[4] =
        cumul_bytes[3] +
        pipeann::save_bin<uint32_t>(refined_pq_pivots_path.c_str(),
                                    chunk_offsets.data(),
                                    chunk_offsets.size(), 1,
                                    cumul_bytes[3]);
    pipeann::save_bin<_u64>(refined_pq_pivots_path.c_str(),
                            cumul_bytes.data(),
                            cumul_bytes.size(), 1, 0);
    LOG(INFO) << "Saved refined pivots to " << refined_pq_pivots_path;
  }

  // 7) save map file: format -> uint32_t N, uint32_t n_chunks, bytes[N * n_chunks]
  {
    std::ofstream fout(map_out_path, std::ios::binary | std::ios::out);
    if (!fout) {
      LOG(ERROR) << "Cannot open map_out_path: " << map_out_path;
      return -1;
    }
    uint32_t N32 = (uint32_t)N;
    uint32_t C32 = (uint32_t)num_pq_chunks;
    fout.write((char *)&N32, sizeof(uint32_t));
    fout.write((char *)&C32, sizeof(uint32_t));
    fout.write((char *)map_bytes.data(),
               map_bytes.size() * sizeof(uint8_t));
    fout.close();
    LOG(INFO) << "Saved map (" << map_bytes.size() << " bytes) to "
              << map_out_path;
  }

  // 8) generate refined compressed vectors file (完整编码 LUT2)
  int enc_rc = generate_pq_data_from_pivots<T>(
      data_file, num_centers, num_pq_chunks,
      refined_pq_pivots_path, refined_pq_compressed_vectors_path, offset);
  if (enc_rc != 0) {
    LOG(ERROR) << "generate_pq_data_from_pivots failed for refined pivots";
    return -1;
  }
  LOG(INFO) << "Saved refined compressed vectors to "
            << refined_pq_compressed_vectors_path;

  LOG(INFO) << "[refine_pq_toppercent] finished successfully.";
  return 0;
}

template<typename T>
int build_dual_pq_512_from_scratch(const std::string data_file,
                                   const std::string initial_pq_pivots_path, // 仅用于读取 centroid/rearrangement/chunk_offsets
                                   const std::string pivots_1_out_path,      // 导出：前半（0..255）pivots
                                   const std::string pivots_2_out_path,      // 导出：后半（256..511）pivots
                                   const std::string map_out_path,           // 导出：map（N × num_pq_chunks；字节）
                                   const std::string compressed_1_out_path,  // 导出：用 pivots_1 编码的全量数据
                                   const std::string compressed_2_out_path,  // 导出：用 pivots_2 编码的全量数据
                                   unsigned num_centers,       // 例如 256（实际每 chunk 训练 2*num_centers）
                                   unsigned num_pq_chunks,
                                   unsigned max_k_means_reps,
                                   size_t offset) {
  LOG(INFO) << "[build_dual_pq_512_from_scratch] data=" << data_file
            << " pivots1_out=" << pivots_1_out_path
            << " pivots2_out=" << pivots_2_out_path
            << " map_out=" << map_out_path
            << " comp1_out=" << compressed_1_out_path
            << " comp2_out=" << compressed_2_out_path
            << " centers(each)=" << num_centers
            << " chunks=" << num_pq_chunks;

  // 1) 读数据头（只读一次，之后按需重新打开文件流）
  _u64 read_blk_size = 64ull * 1024ull * 1024ull;
  {
    cached_ifstream base_reader(data_file, read_blk_size, (uint32_t) offset);
    _u32 npts32 = 0, nd32 = 0;
    base_reader.read((char *)&npts32, sizeof(uint32_t));
    base_reader.read((char *)&nd32, sizeof(uint32_t));
    size_t N = (size_t) npts32;
    size_t dim = (size_t) nd32;
    if (N == 0 || dim == 0) {
      LOG(ERROR) << "Bad data file metadata.";
      return -1;
    }
    LOG(INFO) << "Dataset: N=" << N << " dim=" << dim;
  }

  // 再次读取 N 和 dim（方便后续使用）
  _u32 npts32 = 0, nd32 = 0;
  {
    cached_ifstream base_reader(data_file, read_blk_size, (uint32_t) offset);
    base_reader.read((char *)&npts32, sizeof(uint32_t));
    base_reader.read((char *)&nd32, sizeof(uint32_t));
  }
  const size_t N   = (size_t) npts32;
  const size_t dim = (size_t) nd32;

  if (N == 0 || dim == 0) {
    LOG(ERROR) << "Bad data file metadata (second check).";
    return -1;
  }

  // 2) 仅从 initial_pq_pivots_path 读取 centroid/rearrangement/chunk_offsets
  std::unique_ptr<float[]> centroid;
  std::vector<uint32_t>    rearrangement;
  std::vector<uint32_t>    chunk_offsets;
  {
    _u64 nr=0,nc=0;
    std::unique_ptr<_u64[]> file_offsets;
    pipeann::load_bin<_u64>(initial_pq_pivots_path.c_str(), file_offsets, nr, nc, 0);
    if (nr != 5) {
      LOG(ERROR) << "pivots file offsets != 5";
      return -1;
    }
    // 跳过旧 pivots，仅用 metadata
    std::unique_ptr<float[]> dummy;
    pipeann::load_bin<float>(initial_pq_pivots_path.c_str(), dummy, nr, nc, file_offsets[0]);

    // centroid
    pipeann::load_bin<float>(initial_pq_pivots_path.c_str(), centroid, nr, nc, file_offsets[1]);
    if ((size_t)nr != dim || (size_t)nc != 1) {
      LOG(ERROR) << "centroid shape mismatch.";
      return -1;
    }

    // rearrangement
    std::unique_ptr<uint32_t[]> rearr_ptr;
    pipeann::load_bin<uint32_t>(initial_pq_pivots_path.c_str(), rearr_ptr, nr, nc, file_offsets[2]);
    if ((size_t)nr != dim) {
      LOG(ERROR) << "rearrangement length mismatch.";
      return -1;
    }
    rearrangement.assign(rearr_ptr.get(), rearr_ptr.get() + dim);

    // chunk offsets
    std::unique_ptr<uint32_t[]> chunk_ptr;
    pipeann::load_bin<uint32_t>(initial_pq_pivots_path.c_str(), chunk_ptr, nr, nc, file_offsets[3]);
    if ((size_t)nr != (size_t)(num_pq_chunks + 1)) {
      LOG(ERROR) << "chunk_offsets rows mismatch: got " << nr
                 << " expected " << (num_pq_chunks+1);
      return -1;
    }
    chunk_offsets.assign(chunk_ptr.get(), chunk_ptr.get() + (num_pq_chunks + 1));
  }

  // 3) 为两套 pivots 分配空间（每套 num_centers × dim，按全维布局）
  std::unique_ptr<float[]> pivots_1 = std::make_unique<float[]>((size_t)num_centers * dim);
  std::unique_ptr<float[]> pivots_2 = std::make_unique<float[]>((size_t)num_centers * dim);

  // 4) map：N × num_pq_chunks（0 表示用 pivots_1，1 表示用 pivots_2）
  //    Deep100M + 32 chunks ≈ 3.2GB，这里仍然是一个大块内存，但远小于原版总峰值。
  std::vector<uint8_t> map_bytes((size_t)N * (size_t)num_pq_chunks, 0);

  // 5) 逐 chunk：直接训练 2*num_centers 个质心，再拆成两套
  const unsigned K512 = 2 * num_centers;

  // 设定一个“批大小”用于流式读取数据并作类型转换/中心化/重排，避免一次性 N×dim 的大数组
  const _u64 MAX_POINTS_PER_BATCH = 1000000;  // 你可以按机器内存调整

  for (unsigned c = 0; c < num_pq_chunks; ++c) {
    size_t s   = (size_t)chunk_offsets[c];
    size_t e   = (size_t)chunk_offsets[c+1];
    size_t csz = e - s;   // 当前 chunk 维度数

    if (csz == 0) {
      LOG(INFO) << "chunk " << c << " size 0 skip";
      continue;
    }

    LOG(INFO) << "Train chunk " << c << " dims [" << s << "," << e << ") size=" << csz
              << " with K=" << K512;

    // cur_data: N × csz，仅保存该 chunk 的数据（已中心化 + 重排后）
    std::unique_ptr<float[]> cur_data =
        std::make_unique<float[]>((size_t)N * csz);

    // ---------------------------------------------------------
    // 5.1) 流式读取数据构造 cur_data（按 batch 完成类型转换/中心化/重排）
    // ---------------------------------------------------------
    {
      cached_ifstream base_reader(data_file, read_blk_size, (uint32_t) offset);
      _u32 npts32_r = 0, nd32_r = 0;
      base_reader.read((char *)&npts32_r, sizeof(uint32_t));
      base_reader.read((char *)&nd32_r, sizeof(uint32_t));
      if ((size_t)npts32_r != N || (size_t)nd32_r != dim) {
        LOG(ERROR) << "Header changed when re-reading data.";
        return -1;
      }

      const _u64 total_points = (_u64)N;
      const _u64 batch_points =
          std::min<_u64>(MAX_POINTS_PER_BATCH, total_points);

      std::unique_ptr<T[]>     batch_T   = std::make_unique<T[]>((_u64)batch_points * dim);
      std::unique_ptr<float[]> batch_f   = std::make_unique<float[]>((_u64)batch_points * dim);
      std::unique_ptr<float[]> batch_tmp = std::make_unique<float[]>((_u64)batch_points * dim);

      _u64 processed = 0;
      while (processed < total_points) {
        _u64 cur_batch_size = std::min<_u64>(batch_points,
                                             total_points - processed);
        // 读取原始数据（T 类型）
        base_reader.read((char *)batch_T.get(),
                         sizeof(T) * (size_t)cur_batch_size * dim);

        // T -> float
        pipeann::convert_types<T, float>(batch_T.get(), batch_tmp.get(),
                                         (size_t)cur_batch_size, dim);

        // 中心化 + 重排（得到完整维度的 batch_f）
#pragma omp parallel for schedule(static, 8192)
        for (int64_t i = 0; i < (_s64)cur_batch_size; i++) {
          float *dst_full = batch_f.get() + (size_t)i * dim;
          float *src_full = batch_tmp.get() + (size_t)i * dim;

          // 中心化
          for (size_t d = 0; d < dim; d++) {
            src_full[d] -= centroid[d];
          }
          // 重排
          for (size_t d = 0; d < dim; d++) {
            dst_full[d] = src_full[rearrangement[d]];
          }
        }

        // 抽取当前 chunk 的维度 [s,e) → 写入 cur_data 对应位置
#pragma omp parallel for schedule(static, 8192)
        for (int64_t i = 0; i < (_s64)cur_batch_size; i++) {
          size_t global_idx = (size_t)processed + (size_t)i;  // 全局第几条向量
          float *dst_chunk = cur_data.get() + global_idx * csz;
          float *src_full  = batch_f.get() + (size_t)i * dim;
          // 拷贝 [s,e)
          std::memcpy(dst_chunk, src_full + s, csz * sizeof(float));
        }

        processed += cur_batch_size;
      }  // end while processed < total_points
    }    // 结束构造 cur_data 的作用域，batch_* 内存释放

    // ---------------------------------------------------------
    // 5.2) 用 cur_data 训练 K512 个 pivots（K-means）
    // ---------------------------------------------------------
    std::unique_ptr<float[]> pivots512 =
        std::make_unique<float[]>((size_t)K512 * csz);

    kmeans::selecting_pivots(cur_data.get(), N, csz, pivots512.get(), K512);
    kmeans::run_lloyds(cur_data.get(), N, csz, pivots512.get(),
                       K512, max_k_means_reps, nullptr, nullptr);

    // ---------------------------------------------------------
    // 5.3) 将 pivots512 拆成两半，写入全维 pivots_1 / pivots_2 对应位置
    //      A: [0..num_centers-1], B: [num_centers..K512-1]
    // ---------------------------------------------------------
#pragma omp parallel for schedule(static,1)
    for (int64_t k = 0; k < (_s64)num_centers; k++) {
      // A：写入 pivots_1
      std::memcpy(pivots_1.get() + (size_t)k * dim + s,
                  pivots512.get() + (size_t)k * csz,
                  csz * sizeof(float));
      // B：写入 pivots_2
      std::memcpy(pivots_2.get() + (size_t)k * dim + s,
                  pivots512.get() + ((size_t)num_centers + (size_t)k) * csz,
                  csz * sizeof(float));
    }

    // ---------------------------------------------------------
    // 5.4) 计算每个点到 A/B 两套 pivots 的最近距离，生成 map
    //      这里也按 batch 做，避免分配 N 规模的临时数组
    // ---------------------------------------------------------
    {
      // 构造按 chunk 排好的一块 pivots_1 / pivots_2：num_centers × csz
      std::unique_ptr<float[]> pivots1_chunk =
          std::make_unique<float[]>((size_t)num_centers * csz);
      std::unique_ptr<float[]> pivots2_chunk =
          std::make_unique<float[]>((size_t)num_centers * csz);

#pragma omp parallel for schedule(static,1)
      for (int64_t k = 0; k < (_s64)num_centers; k++) {
        std::memcpy(pivots1_chunk.get() + (size_t)k * csz,
                    pivots_1.get() + (size_t)k * dim + s,
                    csz * sizeof(float));
        std::memcpy(pivots2_chunk.get() + (size_t)k * csz,
                    pivots_2.get() + (size_t)k * dim + s,
                    csz * sizeof(float));
      }

      const _u64 total_points = (_u64)N;
      const _u64 batch_points =
          std::min<_u64>(MAX_POINTS_PER_BATCH, total_points);

      std::unique_ptr<uint32_t[]> closest_A =
          std::make_unique<uint32_t[]>((size_t)batch_points);
      std::unique_ptr<uint32_t[]> closest_B =
          std::make_unique<uint32_t[]>((size_t)batch_points);
      std::unique_ptr<float[]> dist_A =
          std::make_unique<float[]>((size_t)batch_points);
      std::unique_ptr<float[]> dist_B =
          std::make_unique<float[]>((size_t)batch_points);

      _u64 processed = 0;
      while (processed < total_points) {
        _u64 cur_batch_size = std::min<_u64>(batch_points,
                                             total_points - processed);

        float *batch_data_ptr =
            cur_data.get() + (size_t)processed * csz;

        // 最近 A
        math_utils::compute_closest_centers(batch_data_ptr,
                                            (size_t)cur_batch_size,
                                            csz,
                                            pivots1_chunk.get(), num_centers, 1,
                                            closest_A.get(),
                                            nullptr, nullptr,
                                            dist_A.get());

        // 最近 B
        math_utils::compute_closest_centers(batch_data_ptr,
                                            (size_t)cur_batch_size,
                                            csz,
                                            pivots2_chunk.get(), num_centers, 1,
                                            closest_B.get(),
                                            nullptr, nullptr,
                                            dist_B.get());

        // 生成 map：比较 A/B 的最近距离，哪个更小选哪个（0=A，1=B）
#pragma omp parallel for schedule(static, 8192)
        for (int64_t i = 0; i < (_s64)cur_batch_size; i++) {
          size_t global_idx = (size_t)processed + (size_t)i;
          uint8_t use_B     = dist_B[i] < dist_A[i] ? 1 : 0;
          map_bytes[global_idx * (size_t)num_pq_chunks + (size_t)c] = use_B;
        }

        processed += cur_batch_size;
      }
    }  // end 生成 map 的作用域

    // cur_data 此处生命周期结束，内存释放，进入下一个 chunk
  } // end for chunks

  // 6) 分别保存 pivots_1 / pivots_2（沿用你原来的 5 段布局）
  auto save_pivots_like_original = [&](const std::string& out_path,
                                       float* pivots_mat) -> int {
    std::vector<size_t> cumul_bytes(5,0);
    cumul_bytes[0] = METADATA_SIZE;
    cumul_bytes[1] = cumul_bytes[0] + pipeann::save_bin<float>(
        out_path.c_str(), pivots_mat,
        (size_t)num_centers, dim, cumul_bytes[0]);

    cumul_bytes[2] = cumul_bytes[1] + pipeann::save_bin<float>(
        out_path.c_str(), centroid.get(),
        (size_t)dim, 1, cumul_bytes[1]);

    cumul_bytes[3] = cumul_bytes[2] + pipeann::save_bin<uint32_t>(
        out_path.c_str(), rearrangement.data(),
        rearrangement.size(), 1, cumul_bytes[2]);

    cumul_bytes[4] = cumul_bytes[3] + pipeann::save_bin<uint32_t>(
        out_path.c_str(), chunk_offsets.data(),
        chunk_offsets.size(), 1, cumul_bytes[3]);

    pipeann::save_bin<_u64>(out_path.c_str(), cumul_bytes.data(),
                            cumul_bytes.size(), 1, 0);
    return 0;
  };

  if (save_pivots_like_original(pivots_1_out_path, pivots_1.get()) != 0) {
    LOG(ERROR) << "Failed to save pivots_1";
    return -1;
  }
  if (save_pivots_like_original(pivots_2_out_path, pivots_2.get()) != 0) {
    LOG(ERROR) << "Failed to save pivots_2";
    return -1;
  }
  LOG(INFO) << "Saved pivots_1 to " << pivots_1_out_path;
  LOG(INFO) << "Saved pivots_2 to " << pivots_2_out_path;

  // 7) 保存 map（头：N、num_pq_chunks，之后 N×num_pq_chunks 字节）
  {
    std::ofstream fout(map_out_path, std::ios::binary | std::ios::out);
    if (!fout) {
      LOG(ERROR) << "Cannot open map_out_path: " << map_out_path;
      return -1;
    }
    uint32_t N32 = (uint32_t)N, C32 = (uint32_t)num_pq_chunks;
    fout.write((char *)&N32, sizeof(uint32_t));
    fout.write((char *)&C32, sizeof(uint32_t));
    fout.write((char *)map_bytes.data(),
               map_bytes.size() * sizeof(uint8_t));
    fout.close();
  }
  LOG(INFO) << "Saved map (" << map_bytes.size() << " bytes) to "
            << map_out_path;

  // 8) 分别用两份 pivots 生成两份压缩编码文件（与你现有 encode 逻辑完全一致）
  {
    int rc1 = generate_pq_data_from_pivots<T>(data_file, num_centers, num_pq_chunks,
                                              pivots_1_out_path, compressed_1_out_path, offset);
    if (rc1 != 0) {
      LOG(ERROR) << "generate_pq_data_from_pivots failed for pivots_1";
      return -1;
    }
    int rc2 = generate_pq_data_from_pivots<T>(data_file, num_centers, num_pq_chunks,
                                              pivots_2_out_path, compressed_2_out_path, offset);
    if (rc2 != 0) {
      LOG(ERROR) << "generate_pq_data_from_pivots failed for pivots_2";
      return -1;
    }
    LOG(INFO) << "Saved compressed_1 to " << compressed_1_out_path;
    LOG(INFO) << "Saved compressed_2 to " << compressed_2_out_path;
  }

  LOG(INFO) << "[build_dual_pq_512_from_scratch] finished successfully.";
  return 0;
}



template<typename T>
int estimate_cluster_sizes(const std::string data_file, float *pivots, const size_t num_centers, const size_t dim,
                           const size_t k_base, std::vector<size_t> &cluster_sizes) {
  cluster_sizes.clear();

  size_t num_test, test_dim;
  float *test_data_float;
  double sampling_rate = 0.01;

  gen_random_slice<T>(data_file, sampling_rate, test_data_float, num_test, test_dim);

  if (test_dim != dim) {
    LOG(INFO) << "Error. dimensions dont match for pivot set and base set";
    return -1;
  }

  size_t *shard_counts = new size_t[num_centers];

  for (size_t i = 0; i < num_centers; i++) {
    shard_counts[i] = 0;
  }

  size_t BLOCK_SIZE = (std::min)((size_t) MAX_BLOCK_SIZE, num_test);
  size_t num_points = 0, num_dim = 0;
  pipeann::get_bin_metadata(data_file, num_points, num_dim);
  size_t block_size = num_points <= BLOCK_SIZE ? num_points : BLOCK_SIZE;
  _u32 *block_closest_centers = new _u32[block_size * k_base];
  float *block_data_float;

  size_t num_blocks = DIV_ROUND_UP(num_test, block_size);

  for (size_t block = 0; block < num_blocks; block++) {
    size_t start_id = block * block_size;
    size_t end_id = (std::min)((block + 1) * block_size, num_test);
    size_t cur_blk_size = end_id - start_id;

    block_data_float = test_data_float + start_id * test_dim;

    math_utils::compute_closest_centers(block_data_float, cur_blk_size, dim, pivots, num_centers, k_base,
                                        block_closest_centers);

    for (size_t p = 0; p < cur_blk_size; p++) {
      for (size_t p1 = 0; p1 < k_base; p1++) {
        size_t shard_id = block_closest_centers[p * k_base + p1];
        shard_counts[shard_id]++;
      }
    }
  }

  LOG(INFO) << "Estimated cluster sizes: ";
  for (size_t i = 0; i < num_centers; i++) {
    _u32 cur_shard_count = (_u32) shard_counts[i];
    cluster_sizes.push_back(size_t(((double) cur_shard_count) * (1.0 / sampling_rate)));
    std::cerr << cur_shard_count * (1.0 / sampling_rate) << " ";
  }
  std::cerr << "\n";
  delete[] shard_counts;
  delete[] block_closest_centers;
  return 0;
}

template<typename T>
int shard_data_into_clusters(const std::string data_file, float *pivots, const size_t num_centers, const size_t dim,
                             const size_t k_base, std::string prefix_path) {
  _u64 read_blk_size = 64 * 1024 * 1024;
  //  _u64 write_blk_size = 64 * 1024 * 1024;
  // create cached reader + writer
  cached_ifstream base_reader(data_file, read_blk_size);
  _u32 npts32;
  _u32 basedim32;
  base_reader.read((char *) &npts32, sizeof(uint32_t));
  base_reader.read((char *) &basedim32, sizeof(uint32_t));
  size_t num_points = npts32;
  if (basedim32 != dim) {
    LOG(INFO) << "Error. dimensions dont match for train set and base set";
    return -1;
  }

  std::unique_ptr<size_t[]> shard_counts = std::make_unique<size_t[]>(num_centers);
  std::vector<std::ofstream> shard_data_writer(num_centers);
  std::vector<std::ofstream> shard_idmap_writer(num_centers);
  _u32 dummy_size = 0;
  _u32 const_one = 1;

  for (size_t i = 0; i < num_centers; i++) {
    std::string data_filename = prefix_path + "_subshard-" + std::to_string(i) + ".bin";
    std::string idmap_filename = prefix_path + "_subshard-" + std::to_string(i) + "_ids_uint32.bin";
    shard_data_writer[i] = std::ofstream(data_filename.c_str(), std::ios::binary);
    shard_idmap_writer[i] = std::ofstream(idmap_filename.c_str(), std::ios::binary);
    shard_data_writer[i].write((char *) &dummy_size, sizeof(uint32_t));
    shard_data_writer[i].write((char *) &basedim32, sizeof(uint32_t));
    shard_idmap_writer[i].write((char *) &dummy_size, sizeof(uint32_t));
    shard_idmap_writer[i].write((char *) &const_one, sizeof(uint32_t));
    shard_counts[i] = 0;
  }

  size_t BLOCK_SIZE = (std::min)((size_t) MAX_BLOCK_SIZE, num_points);
  size_t block_size = num_points <= BLOCK_SIZE ? num_points : BLOCK_SIZE;
  std::unique_ptr<_u32[]> block_closest_centers = std::make_unique<_u32[]>(block_size * k_base);
  std::unique_ptr<T[]> block_data_T = std::make_unique<T[]>(block_size * dim);
  std::unique_ptr<float[]> block_data_float = std::make_unique<float[]>(block_size * dim);

  size_t num_blocks = DIV_ROUND_UP(num_points, block_size);

  for (size_t block = 0; block < num_blocks; block++) {
    size_t start_id = block * block_size;
    size_t end_id = (std::min)((block + 1) * block_size, num_points);
    size_t cur_blk_size = end_id - start_id;

    base_reader.read((char *) block_data_T.get(), sizeof(T) * (cur_blk_size * dim));
    pipeann::convert_types<T, float>(block_data_T.get(), block_data_float.get(), cur_blk_size, dim);

    math_utils::compute_closest_centers(block_data_float.get(), cur_blk_size, dim, pivots, num_centers, k_base,
                                        block_closest_centers.get());

    for (size_t p = 0; p < cur_blk_size; p++) {
      for (size_t p1 = 0; p1 < k_base; p1++) {
        size_t shard_id = block_closest_centers[p * k_base + p1];
        uint32_t original_point_map_id = (uint32_t) (start_id + p);
        shard_data_writer[shard_id].write((char *) (block_data_T.get() + p * dim), sizeof(T) * dim);
        shard_idmap_writer[shard_id].write((char *) &original_point_map_id, sizeof(uint32_t));
        shard_counts[shard_id]++;
      }
    }
  }

  size_t total_count = 0;
  LOG(INFO) << "Actual shard sizes: ";
  for (size_t i = 0; i < num_centers; i++) {
    _u32 cur_shard_count = (_u32) shard_counts[i];
    total_count += cur_shard_count;
    LOG(INFO) << cur_shard_count << " ";
    shard_data_writer[i].seekp(0);
    shard_data_writer[i].write((char *) &cur_shard_count, sizeof(uint32_t));
    shard_data_writer[i].close();
    shard_idmap_writer[i].seekp(0);
    shard_idmap_writer[i].write((char *) &cur_shard_count, sizeof(uint32_t));
    shard_idmap_writer[i].close();
  }

  LOG(INFO) << "\n Partitioned " << num_points << " with replication factor " << k_base << " to get " << total_count
            << " points across " << num_centers << " shards ";
  return 0;
}

template<typename T1, typename T2>
int shard_two_data_into_clusters(const std::string data_file1, const std::string data_file2, 
                                 float *pivots, const size_t num_centers, 
                                 const size_t dim1, // 第一类向量维度
                                 const size_t dim2, // 第二类向量维度
                                 const size_t k_base, std::string prefix_path) {
  _u64 read_blk_size = 64 * 1024 * 1024; // 64MB I/O 块大小

  // 1. 初始化两个 Reader 并校验文件头
  cached_ifstream base_reader1(data_file1, read_blk_size);
  cached_ifstream base_reader2(data_file2, read_blk_size);
  
  _u32 npts1, basedim1, npts2, basedim2;
  base_reader1.read((char *) &npts1, sizeof(uint32_t));
  base_reader1.read((char *) &basedim1, sizeof(uint32_t));
  base_reader2.read((char *) &npts2, sizeof(uint32_t));
  base_reader2.read((char *) &basedim2, sizeof(uint32_t));

  if (npts1 != npts2 || basedim1 != dim1 || basedim2 != dim2) {
    LOG(INFO) << "Error: dimensions or point counts don't match!";
    return -1;
  }
  size_t num_points = npts1;

  // 2. 准备三套 Writer：第一类数据、第二类数据、原始 ID 映射
  std::unique_ptr<size_t[]> shard_counts = std::make_unique<size_t[]>(num_centers);
  std::vector<std::ofstream> shard_data1_writer(num_centers);
  std::vector<std::ofstream> shard_data2_writer(num_centers);
  std::vector<std::ofstream> shard_idmap_writer(num_centers);
  _u32 dummy_size = 0;
  _u32 const_one = 1;

  for (size_t i = 0; i < num_centers; i++) {
    shard_data1_writer[i] = std::ofstream(prefix_path + "_subshard-" + std::to_string(i) + ".bin", std::ios::binary);
    shard_data2_writer[i] = std::ofstream(prefix_path + "_subshard_loc-" + std::to_string(i) + ".bin", std::ios::binary); // 第二类文件
    shard_idmap_writer[i] = std::ofstream(prefix_path + "_subshard-" + std::to_string(i) + "_ids_uint32.bin", std::ios::binary);
    
    // 写入占位用的文件头 (dummy_size 后续会回填)
    shard_data1_writer[i].write((char *) &dummy_size, sizeof(uint32_t));
    shard_data1_writer[i].write((char *) &basedim1, sizeof(uint32_t));
    shard_data2_writer[i].write((char *) &dummy_size, sizeof(uint32_t));
    shard_data2_writer[i].write((char *) &basedim2, sizeof(uint32_t));
    shard_idmap_writer[i].write((char *) &dummy_size, sizeof(uint32_t));
    shard_idmap_writer[i].write((char *) &const_one, sizeof(uint32_t));
    shard_counts[i] = 0;
  }

  size_t BLOCK_SIZE = (std::min)((size_t) MAX_BLOCK_SIZE, num_points);
  // 3. 准备读写缓冲区：包含 T1, T2 和用于计算距离的 float
  std::unique_ptr<_u32[]> block_closest_centers = std::make_unique<_u32[]>(BLOCK_SIZE * k_base);
  std::unique_ptr<T1[]> block_data_T1 = std::make_unique<T1[]>(BLOCK_SIZE * dim1);
  std::unique_ptr<T2[]> block_data_T2 = std::make_unique<T2[]>(BLOCK_SIZE * dim2);
  std::unique_ptr<float[]> block_data_float = std::make_unique<float[]>(BLOCK_SIZE * dim1);

  size_t num_blocks = DIV_ROUND_UP(num_points, BLOCK_SIZE);

  // 4. 分块读取与同步切分流水线
  for (size_t block = 0; block < num_blocks; block++) {
    size_t start_id = block * BLOCK_SIZE;
    size_t end_id = (std::min)((block + 1) * BLOCK_SIZE, num_points);
    size_t cur_blk_size = end_id - start_id;

    // 同步读取一块数据
    base_reader1.read((char *) block_data_T1.get(), sizeof(T1) * (cur_blk_size * dim1));
    base_reader2.read((char *) block_data_T2.get(), sizeof(T2) * (cur_blk_size * dim2));
    
    // 将主特征向量转为 float 用于求距离
    pipeann::convert_types<T1, float>(block_data_T1.get(), block_data_float.get(), cur_blk_size, dim1);

    // 【核心】只用第一类向量 (block_data_float) 去计算归属哪个簇
    math_utils::compute_closest_centers(block_data_float.get(), cur_blk_size, dim1, pivots, num_centers, k_base,
                                        block_closest_centers.get());

    for (size_t p = 0; p < cur_blk_size; p++) {
      for (size_t p1 = 0; p1 < k_base; p1++) {
        size_t shard_id = block_closest_centers[p * k_base + p1];
        uint32_t original_point_map_id = (uint32_t) (start_id + p);
        
        // 【核心同步写入】将 T1 和 T2 一起写入对应的分片！
        shard_data1_writer[shard_id].write((char *) (block_data_T1.get() + p * dim1), sizeof(T1) * dim1);
        shard_data2_writer[shard_id].write((char *) (block_data_T2.get() + p * dim2), sizeof(T2) * dim2);
        shard_idmap_writer[shard_id].write((char *) &original_point_map_id, sizeof(uint32_t));
        
        shard_counts[shard_id]++; // 只要写一次 ID map，统计一次即可
      }
    }
  }

  // 5. 结尾处理：回填真实的节点数量并关闭文件
  size_t total_count = 0;
  for (size_t i = 0; i < num_centers; i++) {
    // _u32 cur_shard_count = (_u32) (shard_counts[i] / k_base); // 注意：由于 k_base 循环里++了，如果是全局 count 需处理，但这里每个 shard 的 count 是准确的，所以：
    _u32 real_shard_count = (_u32) shard_counts[i]; 
    total_count += real_shard_count;
    
    // 回填 data1
    shard_data1_writer[i].seekp(0);
    shard_data1_writer[i].write((char *) &real_shard_count, sizeof(uint32_t));
    shard_data1_writer[i].close();
    // 回填 data2
    shard_data2_writer[i].seekp(0);
    shard_data2_writer[i].write((char *) &real_shard_count, sizeof(uint32_t));
    shard_data2_writer[i].close();
    // 回填 idmap
    shard_idmap_writer[i].seekp(0);
    shard_idmap_writer[i].write((char *) &real_shard_count, sizeof(uint32_t));
    shard_idmap_writer[i].close();
  }

  LOG(INFO) << "\n Partitioned " << num_points << " with replication factor " << k_base << " to get " << total_count
            << " points across " << num_centers << " shards ";
  return 0;
}
template<typename T1, typename T2>
int shard_two_data_randomly(const std::string data_file1, const std::string data_file2, 
                             const uint32_t num_shards, 
                             const size_t dim1, const size_t dim2,
                             const size_t k_base, std::string prefix_path) {
  _u64 read_blk_size = 64 * 1024 * 1024; 

  // 1. 初始化 Reader (逻辑同前)
  cached_ifstream base_reader1(data_file1, read_blk_size);
  cached_ifstream base_reader2(data_file2, read_blk_size);
  
  _u32 npts1, basedim1, npts2, basedim2;
  base_reader1.read((char *) &npts1, sizeof(uint32_t));
  base_reader1.read((char *) &basedim1, sizeof(uint32_t));
  base_reader2.read((char *) &npts2, sizeof(uint32_t));
  base_reader2.read((char *) &basedim2, sizeof(uint32_t));

  size_t num_points = npts1;
  // 安全检查：副本数不能超过总分片数
  size_t actual_k = std::min((size_t)num_shards, k_base);

  // 2. 初始化 Writers (逻辑同前)
  std::vector<size_t> shard_counts(num_shards, 0);
  std::vector<std::ofstream> shard_data1_writer(num_shards);
  std::vector<std::ofstream> shard_data2_writer(num_shards);
  std::vector<std::ofstream> shard_idmap_writer(num_shards);

  for (size_t i = 0; i < num_shards; i++) {
    shard_data1_writer[i].open(prefix_path + "_subshard-" + std::to_string(i) + ".bin", std::ios::binary);
    shard_data2_writer[i].open(prefix_path + "_subshard_loc-" + std::to_string(i) + ".bin", std::ios::binary);
    shard_idmap_writer[i].open(prefix_path + "_subshard-" + std::to_string(i) + "_ids_uint32.bin", std::ios::binary);
    
    // 写入初始文件头 (4字节 count + 4字节 dim)
    uint32_t dummy = 0;
    shard_data1_writer[i].write((char *) &dummy, sizeof(uint32_t));
    shard_data1_writer[i].write((char *) &basedim1, sizeof(uint32_t));
    shard_data2_writer[i].write((char *) &dummy, sizeof(uint32_t));
    shard_data2_writer[i].write((char *) &basedim2, sizeof(uint32_t));
    shard_idmap_writer[i].write((char *) &dummy, sizeof(uint32_t));
    uint32_t const_one = 1; 
    shard_idmap_writer[i].write((char *) &const_one, sizeof(uint32_t));
  }

  // 3. 分块读取与多副本分发
  size_t BLOCK_SIZE = (std::min)((size_t) MAX_BLOCK_SIZE, num_points);
  std::unique_ptr<T1[]> block_data_T1 = std::make_unique<T1[]>(BLOCK_SIZE * dim1);
  std::unique_ptr<T2[]> block_data_T2 = std::make_unique<T2[]>(BLOCK_SIZE * dim2);

  for (size_t block = 0; block < (num_points + BLOCK_SIZE - 1) / BLOCK_SIZE; block++) {
    size_t start_id = block * BLOCK_SIZE;
    size_t end_id = (std::min)((block + 1) * BLOCK_SIZE, num_points);
    size_t cur_blk_size = end_id - start_id;

    base_reader1.read((char *) block_data_T1.get(), sizeof(T1) * (cur_blk_size * dim1));
    base_reader2.read((char *) block_data_T2.get(), sizeof(T2) * (cur_blk_size * dim2));

    for (size_t p = 0; p < cur_blk_size; p++) {
      uint32_t original_id = (uint32_t)(start_id + p);
      
      // 【核心改动：多副本分发】
      // 为每个点分配 actual_k 个互不相同的分片
      for (size_t k = 0; k < actual_k; k++) {
        // 使用偏移法确保不重复：(ID + k) % num_shards
        // 如果需要更随机，可以使用：(hash(ID) + k) % num_shards
        size_t shard_id = (original_id + k) % num_shards;

        shard_data1_writer[shard_id].write((char *)(block_data_T1.get() + p * dim1), sizeof(T1) * dim1);
        shard_data2_writer[shard_id].write((char *)(block_data_T2.get() + p * dim2), sizeof(T2) * dim2);
        shard_idmap_writer[shard_id].write((char *)&original_id, sizeof(uint32_t));
        
        shard_counts[shard_id]++;
      }
    }
  }

  // 4. 回填文件头 (逻辑同前，确保 count 正确)
  for (size_t i = 0; i < num_shards; i++) {
    uint32_t final_count = (uint32_t)shard_counts[i];
    shard_data1_writer[i].seekp(0);
    shard_data1_writer[i].write((char *)&final_count, sizeof(uint32_t));
    shard_data2_writer[i].seekp(0);
    shard_data2_writer[i].write((char *)&final_count, sizeof(uint32_t));
    shard_idmap_writer[i].seekp(0);
    shard_idmap_writer[i].write((char *)&final_count, sizeof(uint32_t));
    
    shard_data1_writer[i].close();
    shard_data2_writer[i].close();
    shard_idmap_writer[i].close();
  }

  return 0;
}

template<typename T>
int partition_with_ram_budget(const std::string data_file, const double sampling_rate, double ram_budget,
                              size_t graph_degree, const std::string prefix_path, size_t k_base) {
  size_t train_dim;
  size_t num_train;
  float *train_data_float;
  size_t max_k_means_reps = 20;

  int num_parts = 3;
  bool fit_in_ram = false;

  gen_random_slice<T>(data_file, sampling_rate, train_data_float, num_train, train_dim);

  float *pivot_data = nullptr;

  std::string cur_file = std::string(prefix_path);
  std::string output_file;

  // kmeans_partitioning on training data

  //  cur_file = cur_file + "_kmeans_partitioning-" + std::to_string(num_parts);
  output_file = cur_file + "_centroids.bin";

  while (!fit_in_ram) {
    fit_in_ram = true;

    double max_ram_usage = 0;
    if (pivot_data != nullptr)
      delete[] pivot_data;

    pivot_data = new float[num_parts * train_dim];
    // Process Global k-means for kmeans_partitioning Step
    LOG(INFO) << "Processing global k-means (kmeans_partitioning Step)";
    kmeans::kmeanspp_selecting_pivots(train_data_float, num_train, train_dim, pivot_data, num_parts);

    kmeans::run_lloyds(train_data_float, num_train, train_dim, pivot_data, num_parts, max_k_means_reps, NULL, NULL);

    // now pivots are ready. need to stream base points and assign them to
    // closest clusters.

    std::vector<size_t> cluster_sizes;
    estimate_cluster_sizes<T>(data_file, pivot_data, num_parts, train_dim, k_base, cluster_sizes);

    for (auto &p : cluster_sizes) {
      double cur_shard_ram_estimate = pipeann::estimate_ram_usage(p, train_dim, sizeof(T), graph_degree);

      if (cur_shard_ram_estimate > max_ram_usage)
        max_ram_usage = cur_shard_ram_estimate;
    }
    LOG(INFO) << "With " << num_parts << " parts, max estimated RAM usage: " << max_ram_usage / (1024 * 1024 * 1024)
              << "GB, budget given is " << ram_budget;
    if (max_ram_usage > 1024 * 1024 * 1024 * ram_budget) {
      fit_in_ram = false;
      num_parts++;
    }
  }

  LOG(INFO) << "Saving global k-center pivots";
  pipeann::save_bin<float>(output_file.c_str(), pivot_data, (size_t) num_parts, train_dim);

  shard_data_into_clusters<T>(data_file, pivot_data, num_parts, train_dim, k_base, prefix_path);
  delete[] pivot_data;
  delete[] train_data_float;
  return num_parts;
}

template<typename T1, typename T2>
int partition_two_vectors_with_ram_budget(const std::string data_file1, 
                                          const std::string data_file2, // [新增] 第二类向量文件
                                          uint32_t dim1, uint32_t dim2,
                                          const double sampling_rate, double ram_budget,
                                          size_t graph_degree, const std::string prefix_path, size_t k_base) {
  size_t train_dim;
  size_t num_train;
  float *train_data_float_ptr = nullptr;
  size_t max_k_means_reps = 20;

  // 建议：给一个更合理的初始分片预估值，避免从 3 开始做无效的 O(N^2) 循环
  int num_parts = 3; 
  bool fit_in_ram = false;

  // 1. 采样阶段：只抽取主向量 (data_file1) 用于训练 K-Means
  gen_random_slice<T1>(data_file1, sampling_rate, train_data_float_ptr, num_train, train_dim);

  // [优化] 使用 RAII 智能指针接管裸指针，确保发生异常时自动释放内存
  std::unique_ptr<float[]> train_data_float(train_data_float_ptr);
  std::unique_ptr<float[]> pivot_data;

  std::string cur_file = std::string(prefix_path);
  std::string output_file = cur_file + "_centroids.bin";

  while (!fit_in_ram) {
    fit_in_ram = true;
    double max_ram_usage = 0;

    // 重新分配中心点内存
    pivot_data.reset(new float[num_parts * train_dim]);

    LOG(INFO) << "Processing global k-means (kmeans_partitioning Step) with " << num_parts << " parts";
    
    // 2. 聚类阶段：纯粹在主向量 (train_data_float) 的空间内寻找聚类中心
    kmeans::kmeanspp_selecting_pivots(train_data_float.get(), num_train, train_dim, pivot_data.get(), num_parts);
    kmeans::run_lloyds(train_data_float.get(), num_train, train_dim, pivot_data.get(), num_parts, max_k_means_reps, NULL, NULL);

    // 3. 统计预估：扫描主向量全量数据，计算出每个聚类簇会被分到多少个点
    std::vector<size_t> cluster_sizes;
    estimate_cluster_sizes<T1>(data_file1, pivot_data.get(), num_parts, train_dim, k_base, cluster_sizes);

    for (auto &p : cluster_sizes) {
      // [⚠️ 核心注意点] 这里的 RAM 估算可能需要修改。
      // 如果后续在内存中构建子图时，两类向量都要常驻内存，你需要在这里把 T2 的内存开销也加进去。
      double cur_shard_ram_estimate = pipeann::estimate_ram_usage(p, train_dim, dim2, sizeof(T1), graph_degree);
      // 例如修改为： estimate_ram_usage_for_two_types(p, train_dim, dim2, sizeof(T1), sizeof(T2), graph_degree);

      if (cur_shard_ram_estimate > max_ram_usage)
        max_ram_usage = cur_shard_ram_estimate;
    }
    
    LOG(INFO) << "With " << num_parts << " parts, max estimated RAM usage: " << max_ram_usage / (1024 * 1024 * 1024)
              << "GB, budget given is " << ram_budget;
              
    if (max_ram_usage > 1024 * 1024 * 1024 * ram_budget) {
      fit_in_ram = false;
      num_parts++; // 或者使用 num_parts = static_cast<int>(num_parts * 1.5) 倍增策略加速
    }
  }

  LOG(INFO) << "Saving global k-center pivots";
  pipeann::save_bin<float>(output_file.c_str(), pivot_data.get(), (size_t) num_parts, train_dim);

  // 4. 切分阶段：调用双文件同步切分接口
  // 这个底层函数需要保证：当 data_file1 中的第 i 个点被分配到 shard_X 时，
  // data_file2 中的第 i 个点也会被同步写入附属的 shard_X 文件中。
  uint32_t num_parts_fixed = static_cast<uint32_t>(num_parts);
  shard_two_data_into_clusters<T1, T2>(data_file1, data_file2, pivot_data.get(), num_parts_fixed, train_dim, dim2, k_base, prefix_path);

  // 由于使用了 unique_ptr，函数结束或提前 return/throw 时，内存会自动清理
  return num_parts;
}

template<typename T1, typename T2>
int partition_two_vectors_with_ram_budget_r(const std::string data_file1, 
                                          const std::string data_file2, 
                                          uint32_t dim1, uint32_t dim2,
                                          const double sampling_rate, double ram_budget,
                                          size_t graph_degree, const std::string prefix_path, size_t k_base) {
  // 获取总向量数（用于随机分片的内存预估）
  uint32_t npts32, ndims32;
  std::ifstream base_reader(data_file1.c_str());
  // metadata: npts, ndims
  base_reader.read((char *) &npts32, sizeof(unsigned));
  base_reader.read((char *) &ndims32, sizeof(unsigned));
  size_t total_points = npts32; 
  
  // 初始分片数预估：直接根据 RAM 预算算出一个合理的起始值，而不是从 3 开始
  // 假设平均分配，单个点的内存占用大约为：(dim1*sizeof(T1) + dim2*sizeof(T2) + 索引开销)
  // 这里保留你的循环结构以确保严谨性
  int num_parts = 1; 
  bool fit_in_ram = false;

  while (!fit_in_ram) {
    fit_in_ram = true;
    
    // 1. 随机分片的预估逻辑（考虑 K 倍复制）：
    // 总有效点数现在是 total_points * k_base
    // 每个分片分得的期望点数 = (N * K) / num_parts
    size_t total_replicated_points = total_points * k_base;
    size_t expected_points_per_shard = (total_replicated_points + num_parts - 1) / num_parts;
    
    // 对于随机分片，10% 的 buffer 通常绰绰有余，因为随机分布非常均匀
    size_t safety_buffer_points = static_cast<size_t>(expected_points_per_shard * 1.05);

    // 2. RAM 估算
    // 假设在构建子图阶段，T1 和 T2 都会常驻内存
    // 注意：如果 estimate_ram_usage 内部只算了一种维度，
    // 你可能需要手动计算：Points * (dim1*sizeof(T1) + dim2*sizeof(T2) + graph_degree*sizeof(uint32_t))
    double max_ram_usage = pipeann::estimate_ram_usage(
        safety_buffer_points, 
        dim1, 
        dim2, 
        sizeof(T1), // 这里取决于你的 estimate_ram_usage 是否支持双类型大小，
        graph_degree
    );

    LOG(INFO) << "Testing random partition with " << num_parts << " parts and K=" << k_base;
    LOG(INFO) << "Max shard size (est): " << safety_buffer_points 
              << ", RAM: " << max_ram_usage / (1024.0 * 1024.0 * 1024.0) << " GB, Budget: " << ram_budget << " GB";

    if (max_ram_usage > 1024.0 * 1024.0 * 1024.0 * ram_budget) {
      fit_in_ram = false;
      
      // 优化：与其一个一个加，不如直接计算出理论所需的最小 num_parts
      // 这里的 1.1 是为了给刚才的 safety_buffer 留出余量
      double single_point_ram = max_ram_usage / safety_buffer_points;
      double max_points_allowed = (ram_budget * 1024.0 * 1024.0 * 1024.0) / single_point_ram;
      int suggested_parts = static_cast<int>(std::ceil(total_replicated_points / (max_points_allowed * 0.95)));
      
      num_parts = std::max(num_parts + 1, suggested_parts);
    }
  }

  LOG(INFO) << "Finalizing random partition into " << num_parts << " shards.";

  // 3. 执行随机切分
  // 注意：你需要实现/调用一个随机切分的函数，或者将 pivot_data 传为空。
  // 如果底层函数 shard_two_data_into_clusters 强依赖 pivot_data，
  // 你可以传入一个空的 unique_ptr，并在底层判断：如果 pivots 为空，则执行 ID % num_parts。
  uint32_t num_parts_fixed = static_cast<uint32_t>(num_parts);
  
  // 这里假设你修改后的底层函数支持随机模式（例如通过一个 flag 或检测 pivot 为空）
  shard_two_data_randomly<T1, T2>(data_file1, data_file2, num_parts_fixed, dim1, dim2, k_base, prefix_path);

  return num_parts;
}

// Instantations of supported templates
template void gen_random_slice<int8_t>(const std::string data_file, double p_val,
                                       std::unique_ptr<float[]> &sampled_data, size_t &slice_size, size_t &ndims);
template void gen_random_slice<uint8_t>(const std::string data_file, double p_val,
                                        std::unique_ptr<float[]> &sampled_data, size_t &slice_size, size_t &ndims);
template void gen_random_slice<float>(const std::string data_file, double p_val, std::unique_ptr<float[]> &sampled_data,
                                      size_t &slice_size, size_t &ndims);

template void gen_random_slice<int8_t>(const std::string base_file, const std::string output_prefix,
                                       double sampling_rate, size_t offset);
template void gen_random_slice<uint8_t>(const std::string base_file, const std::string output_prefix,
                                        double sampling_rate, size_t offset);
template void gen_random_slice<float>(const std::string base_file, const std::string output_prefix,
                                      double sampling_rate, size_t offset);

template void gen_random_slice<float>(const std::string data_file, double p_val, float *&sampled_data,
                                      size_t &slice_size, size_t &ndims);
template void gen_random_slice<uint8_t>(const std::string data_file, double p_val, float *&sampled_data,
                                        size_t &slice_size, size_t &ndims);
template void gen_random_slice<int8_t>(const std::string data_file, double p_val, float *&sampled_data,
                                       size_t &slice_size, size_t &ndims);

template int partition_with_ram_budget<int8_t>(const std::string data_file, const double sampling_rate,
                                               double ram_budget, size_t graph_degree, const std::string prefix_path,
                                               size_t k_base);
template int partition_with_ram_budget<uint8_t>(const std::string data_file, const double sampling_rate,
                                                double ram_budget, size_t graph_degree, const std::string prefix_path,
                                                size_t k_base);
template int partition_with_ram_budget<float>(const std::string data_file, const double sampling_rate,
                                              double ram_budget, size_t graph_degree, const std::string prefix_path,
                                              size_t k_base);

template int partition_two_vectors_with_ram_budget<int8_t, int8_t>(const std::string data_file1, const std::string data_file2,
                                          uint32_t dim1, uint32_t dim2,
                                          const double sampling_rate, double ram_budget,
                                          size_t graph_degree, const std::string prefix_path, size_t k_base);
template int partition_two_vectors_with_ram_budget<uint8_t, uint8_t>(const std::string data_file1, const std::string data_file2,
                                          uint32_t dim1, uint32_t dim2,
                                          const double sampling_rate, double ram_budget,
                                          size_t graph_degree, const std::string prefix_path, size_t k_base);
template int partition_two_vectors_with_ram_budget<float, float>(const std::string data_file1, const std::string data_file2,
                                          uint32_t dim1, uint32_t dim2,
                                          const double sampling_rate, double ram_budget,
                                          size_t graph_degree, const std::string prefix_path, size_t k_base);
template int partition_two_vectors_with_ram_budget_r<int8_t, int8_t>(const std::string data_file1, 
                                          const std::string data_file2, 
                                          uint32_t dim1, uint32_t dim2,
                                          const double sampling_rate, double ram_budget,
                                          size_t graph_degree, const std::string prefix_path, size_t k_base);
template int partition_two_vectors_with_ram_budget_r<uint8_t, uint8_t>(const std::string data_file1, 
                                          const std::string data_file2, 
                                          uint32_t dim1, uint32_t dim2,
                                          const double sampling_rate, double ram_budget,
                                          size_t graph_degree, const std::string prefix_path, size_t k_base);
template int partition_two_vectors_with_ram_budget_r<float, float>(const std::string data_file1, 
                                          const std::string data_file2, 
                                          uint32_t dim1, uint32_t dim2,
                                          const double sampling_rate, double ram_budget,
                                          size_t graph_degree, const std::string prefix_path, size_t k_base);

template int generate_pq_pivots<float>(const std::unique_ptr<float[]> &passed_train_data, size_t num_train,
                                       unsigned dim, unsigned num_centers, unsigned num_pq_chunks,
                                       unsigned max_k_means_reps, std::string pq_pivots_path);
template int generate_pq_pivots<int8_t>(const std::unique_ptr<int8_t[]> &passed_train_data, size_t num_train,
                                        unsigned dim, unsigned num_centers, unsigned num_pq_chunks,
                                        unsigned max_k_means_reps, std::string pq_pivots_path);
template int generate_pq_pivots<uint8_t>(const std::unique_ptr<uint8_t[]> &passed_train_data, size_t num_train,
                                         unsigned dim, unsigned num_centers, unsigned num_pq_chunks,
                                         unsigned max_k_means_reps, std::string pq_pivots_path);

template int generate_pq_data_from_pivots<int8_t>(const std::string data_file, unsigned num_centers,
                                                  unsigned num_pq_chunks, std::string pq_pivots_path,
                                                  std::string pq_compressed_vectors_path, size_t offset);
template int generate_pq_data_from_pivots<uint8_t>(const std::string data_file, unsigned num_centers,
                                                   unsigned num_pq_chunks, std::string pq_pivots_path,
                                                   std::string pq_compressed_vectors_path, size_t offset);
template int generate_pq_data_from_pivots<float>(const std::string data_file, unsigned num_centers,
                                                 unsigned num_pq_chunks, std::string pq_pivots_path,
                                                 std::string pq_compressed_vectors_path, size_t offset);

template int generate_train_data<int8_t>(const std::string data_file, float *passed_train_data, size_t num_train, unsigned num_centers,
  unsigned num_pq_chunks, std::string pq_pivots_path);
template int generate_train_data<uint8_t>(const std::string data_file, float *passed_train_data, size_t num_train, unsigned num_centers,
     unsigned num_pq_chunks, std::string pq_pivots_path);
template int generate_train_data<float>(const std::string data_file, float *passed_train_data, size_t num_train, unsigned num_centers,
     unsigned num_pq_chunks, std::string pq_pivots_path);

template int refine_pq_pivots_from_errors<int8_t>(const std::string data_file, 
      const std::string initial_pq_pivots_path,
      const std::string refined_pq_pivots_path,
      const std::string refined_pq_compressed_vectors_path, 
      unsigned num_centers, unsigned num_pq_chunks,
      unsigned max_k_means_reps, 
      double error_percentage, size_t offset);

template int refine_pq_pivots_from_errors<uint8_t>(const std::string data_file, 
      const std::string initial_pq_pivots_path,
      const std::string refined_pq_pivots_path,
      const std::string refined_pq_compressed_vectors_path, 
      unsigned num_centers, unsigned num_pq_chunks,
      unsigned max_k_means_reps, 
      double error_percentage, size_t offset);

template int refine_pq_pivots_from_errors<float>(const std::string data_file, 
      const std::string initial_pq_pivots_path,
      const std::string refined_pq_pivots_path,
      const std::string refined_pq_compressed_vectors_path, 
      unsigned num_centers, unsigned num_pq_chunks,
      unsigned max_k_means_reps, 
      double error_percentage, size_t offset);

template int refine_pq_pivots_from_errors_toppercent<float>(const std::string data_file,
        const std::string initial_pq_pivots_path,
        const std::string refined_pq_pivots_path,
        const std::string map_out_path,
        const std::string refined_pq_compressed_vectors_path,
        unsigned num_centers, unsigned num_pq_chunks, unsigned max_k_means_reps, double error_percentage, size_t offset);

template int refine_pq_pivots_from_errors_toppercent<uint8_t>(const std::string data_file,
        const std::string initial_pq_pivots_path,
        const std::string refined_pq_pivots_path,
        const std::string map_out_path,
        const std::string refined_pq_compressed_vectors_path,
        unsigned num_centers, unsigned num_pq_chunks, unsigned max_k_means_reps, double error_percentage, size_t offset);

template int refine_pq_pivots_from_errors_toppercent<int8_t>(const std::string data_file,
        const std::string initial_pq_pivots_path,
        const std::string refined_pq_pivots_path,
        const std::string map_out_path,
        const std::string refined_pq_compressed_vectors_path,
        unsigned num_centers, unsigned num_pq_chunks, unsigned max_k_means_reps, double error_percentage, size_t offset);

template int refine_pq_pivots_from_errors_toppercent_in_one_file<float>(const std::string data_file,
        const std::string initial_pq_pivots_path,
        const std::string refined_pq_pivots_path,
        const std::string map_out_path,
        const std::string refined_pq_compressed_vectors_path,
        unsigned num_centers, unsigned num_pq_chunks, unsigned max_k_means_reps, double error_percentage, size_t offset);

template int refine_pq_pivots_from_errors_toppercent_in_one_file<int8_t>(const std::string data_file,
        const std::string initial_pq_pivots_path,
        const std::string refined_pq_pivots_path,
        const std::string map_out_path,
        const std::string refined_pq_compressed_vectors_path,
        unsigned num_centers, unsigned num_pq_chunks, unsigned max_k_means_reps, double error_percentage, size_t offset);

template int refine_pq_pivots_from_errors_toppercent_in_one_file<uint8_t>(const std::string data_file,
        const std::string initial_pq_pivots_path,
        const std::string refined_pq_pivots_path,
        const std::string map_out_path,
        const std::string refined_pq_compressed_vectors_path,
        unsigned num_centers, unsigned num_pq_chunks, unsigned max_k_means_reps, double error_percentage, size_t offset);

template int build_dual_pq_512_from_scratch<float>(const std::string data_file,
  const std::string initial_pq_pivots_path,
  const std::string pivots_1_out_path,
  const std::string pivots_2_out_path,
  const std::string map_out_path,
  const std::string compressed_1_out_path,
  const std::string compressed_2_out_path,
  unsigned num_centers,
  unsigned num_pq_chunks,
  unsigned max_k_means_reps,
  size_t offset);
template int build_dual_pq_512_from_scratch<int8_t>(const std::string data_file,
  const std::string initial_pq_pivots_path,
  const std::string pivots_1_out_path,
  const std::string pivots_2_out_path,
  const std::string map_out_path,
  const std::string compressed_1_out_path,
  const std::string compressed_2_out_path,
  unsigned num_centers,
  unsigned num_pq_chunks,
  unsigned max_k_means_reps,
  size_t offset);
template int build_dual_pq_512_from_scratch<uint8_t>(const std::string data_file,
  const std::string initial_pq_pivots_path,
  const std::string pivots_1_out_path,
  const std::string pivots_2_out_path,
  const std::string map_out_path,
  const std::string compressed_1_out_path,
  const std::string compressed_2_out_path,
  unsigned num_centers,
  unsigned num_pq_chunks,
  unsigned max_k_means_reps,
  size_t offset);