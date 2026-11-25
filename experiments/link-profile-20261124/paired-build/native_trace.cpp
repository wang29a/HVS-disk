#include "linux_aligned_file_reader.h"
#include "ssd_index.h"
#include "trace_hooks.h"

#include <array>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <fcntl.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace phase0 {
Pages topology, coordinates;
thread_local std::vector<uint32_t> *expanded = nullptr;
thread_local bool coarse_only = false;
} // namespace phase0

namespace {
constexpr uint32_t top_k = 10;
constexpr int threads = 4;

void require(bool ok, const std::string &message) {
  if (!ok)
    throw std::runtime_error(message);
}

struct Mapped {
  int fd;
  size_t bytes;
  const char *data;

  explicit Mapped(const std::string &path) : fd(open(path.c_str(), O_RDONLY)) {
    struct stat st {};
    require(fd >= 0 && fstat(fd, &st) == 0 && st.st_size >= 8, "map " + path);
    bytes = size_t(st.st_size);
    data = static_cast<const char *>(mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0));
    require(data != MAP_FAILED, "mmap " + path);
  }

  Mapped(const Mapped &) = delete;
  Mapped &operator=(const Mapped &) = delete;

  ~Mapped() {
    munmap(const_cast<char *>(data), bytes);
    close(fd);
  }
};

uint32_t u32(const char *value) {
  uint32_t result;
  std::memcpy(&result, value, sizeof(result));
  return result;
}

struct Matrix {
  Mapped map;
  uint32_t rows, dim;

  explicit Matrix(const std::string &path)
      : map(path), rows(u32(map.data)), dim(u32(map.data + 4)) {
    require(dim == 512 && map.bytes == 8 + uint64_t(rows) * dim * sizeof(float),
            "matrix shape " + path);
  }

  const float *row(uint32_t id) const {
    return reinterpret_cast<const float *>(map.data + 8) + size_t(id) * dim;
  }
};

std::vector<char> resident(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  const auto bytes = file.tellg();
  require(bytes > 0, "open for resident load " + path);
  std::vector<char> data(static_cast<size_t>(bytes));
  file.seekg(0);
  require(bool(file.read(data.data(), bytes)), "resident load " + path);
  return data;
}

template <typename T>
void write(std::ofstream &file, const T *items, size_t count) {
  file.write(reinterpret_cast<const char *>(items), count * sizeof(T));
}

struct Result {
  std::vector<uint32_t> expanded;
  std::array<uint32_t, top_k> ids{};
  std::array<float, top_k> distances{};
  uint32_t pages = 0, hits = 0;
};

void trace_job(pipeann::SSDIndex<float> &index, float alpha, uint32_t depth, bool only_coarse,
               const Matrix &x, const Matrix &y, const std::string &id_path, const std::string &out) {
  std::ifstream selection(id_path);
  require(bool(selection), "query selection " + id_path);
  std::vector<uint32_t> ids;
  uint32_t id;
  while (selection >> id) {
    require(id < x.rows, "query selection outside matrix");
    ids.push_back(id);
  }
  require(selection.eof() && !ids.empty(), "invalid query selection");
  std::vector<Result> results(ids.size());
  std::string error;
#pragma omp parallel for schedule(dynamic, 8)
  for (int64_t i = 0; i < static_cast<int64_t>(ids.size()); ++i) {
    auto &row = results[i];
    phase0::expanded = &row.expanded;
    phase0::coarse_only = only_coarse;
    try {
      pipeann::QueryStats stats{};
      auto k = index.pq_page_rerank_search(x.row(ids[i]), y.row(ids[i]), alpha, top_k,
                                          0, depth, row.ids.data(), row.distances.data(),
                                          4, &stats);
      require(k == (only_coarse ? 0 : top_k), "native search returned wrong result count");
      row.pages = stats.n_ios;
      row.hits = stats.n_cache_hits;
      require(row.pages + row.hits == row.expanded.size(), "native cache and expansion counts differ");
    } catch (const std::exception &err) {
#pragma omp critical
      {
        if (error.empty())
          error = err.what();
      }
    }
    phase0::expanded = nullptr;
  }
  require(error.empty(), error);
  std::ofstream file(out, std::ios::binary);
  require(bool(file), "trace output");
  file.write("HVSTRC1\0", 8);
  const uint32_t header[] = {static_cast<uint32_t>(ids.size()), only_coarse ? 0 : top_k};
  write(file, header, 2);
  for (size_t i = 0; i < results.size(); ++i) {
    const auto &row = results[i];
    const uint32_t fields[] = {ids[i], static_cast<uint32_t>(row.expanded.size()), row.pages, row.hits};
    write(file, fields, 4);
    write(file, row.expanded.data(), row.expanded.size());
    if (!only_coarse) {
      write(file, row.ids.data(), top_k);
      write(file, row.distances.data(), top_k);
    }
  }
  file.close();
  require(bool(file), "could not close output trace");
}
} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 6, "usage: native-trace directory x_matrix y_matrix dram|disk jobs.tsv");
    const std::string prefix = std::string(argv[1]) + "/";
    const std::string medium = argv[4];
    require(medium == "dram" || medium == "disk", "unknown read medium");
    Matrix x(argv[2]), y(argv[3]);
    require(x.rows == y.rows, "unpaired query matrices");
    const auto topology = medium == "dram" ? resident(prefix + "disk_index_graph") : std::vector<char>();
    const auto coords = medium == "dram" ? resident(prefix + "disk_index_data") : std::vector<char>();
    if (medium == "dram") {
      phase0::topology = {topology.data(), topology.size()};
      phase0::coordinates = {coords.data(), coords.size()};
    }
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    auto file_reader = std::make_shared<LinuxAlignedFileReader>();
    file_reader->strategy = 17;
    std::shared_ptr<AlignedFileReader> reader = file_reader;
    // The frozen library keeps detached update workers alive. Like the stock
    // read-only query driver, release its resources when this process exits.
    auto *index = new pipeann::SSDIndex<float>(pipeann::Metric::L2, reader,
                                                PQ_PAGE_RERANK_SEARCH, false);
    require(index->load(prefix.c_str(), threads, true, true) == 0, "native index load failed");
    std::ifstream jobs(argv[5]);
    require(bool(jobs), "jobs file");
    float alpha;
    uint32_t depth;
    unsigned only_coarse, count = 0;
    std::string ids, out;
    while (jobs >> alpha >> depth >> only_coarse >> ids >> out) {
      require(std::isfinite(alpha) && alpha >= 0 && alpha <= 1 && depth >= top_k &&
              depth <= 1200 && only_coarse <= 1, "invalid trace job");
      trace_job(*index, alpha, depth, only_coarse != 0, x, y, ids, out);
      ++count;
      std::cout << "TRACE_COMPLETE " << out << std::endl;
    }
    require(jobs.eof() && count != 0, "bad or empty trace job file");
    std::cout.flush();
    std::_Exit(EXIT_SUCCESS);
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    std::cerr.flush();
    std::_Exit(EXIT_FAILURE);
  }
}
