// Exhaustive top-40 under the frozen two-space distance, with resumable ID rows.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#include <sys/stat.h>
#include <unistd.h>

namespace {
constexpr int top = 40, weights = 5, qtile = 128, btile = 32768;
constexpr std::array<double, weights> alpha{.1, .3, .5, .7, .9};

void check(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}

struct Matrix {
  uint32_t n, dim;
  std::vector<double> data, norm;

  explicit Matrix(const char *path) {
    std::ifstream in(path, std::ios::binary);
    uint32_t shape[2]{};
    check(bool(in.read(reinterpret_cast<char *>(shape), sizeof(shape))), "matrix header");
    n = shape[0];
    dim = shape[1];
    check(n > top && dim > 0, "matrix dimensions");
    data.resize(size_t(n) * dim);
    norm.resize(n);
    std::vector<float> row(dim);
    for (uint32_t i = 0; i < n; ++i) {
      check(bool(in.read(reinterpret_cast<char *>(row.data()), dim * sizeof(float))), "matrix data");
      for (uint32_t j = 0; j < dim; ++j) {
        double value = row[j];
        data[size_t(i) * dim + j] = value;
        norm[i] += value * value;
      }
      check(std::isfinite(norm[i]), "nonfinite input vector");
    }
    check(in.peek() == EOF, "matrix trailing input");
  }

  const double *row(uint32_t i) const { return data.data() + size_t(i) * dim; }

  double distance(uint32_t q, uint32_t p) const {
    double total = 0;
    auto *a = row(q);
    auto *b = row(p);
#pragma omp simd reduction(+ : total)
    for (uint32_t j = 0; j < dim; ++j) {
      double delta = a[j] - b[j];
      total += delta * delta;
    }
    return std::sqrt(total);
  }

  void block(const std::vector<uint32_t> &ids, size_t start, int count, std::vector<double> &out) const {
    out.resize(size_t(count) * dim);
    for (int r = 0; r < count; ++r)
      std::copy_n(row(ids[start + r]), dim, out.data() + size_t(r) * dim);
  }
};

using Candidate = std::pair<double, uint32_t>;
using Heap = std::vector<Candidate>;
using Result = std::array<Heap, weights>;

void push(Heap &heap, Candidate c) {
  if (heap.size() == top) {
    if (c >= heap.front())
      return;
    std::pop_heap(heap.begin(), heap.end());
    heap.pop_back();
  }
  heap.push_back(c);
  std::push_heap(heap.begin(), heap.end());
}

void closest(const Matrix &x, const Matrix &y, const std::vector<uint32_t> &ids,
             size_t start, int count, std::ofstream &out) {
  std::vector<Result> best(count);
  std::vector<double> qx, qy, sx(size_t(count) * btile), sy(size_t(count) * btile);
  x.block(ids, start, count, qx);
  y.block(ids, start, count, qy);
  for (uint32_t first = 0; first < x.n; first += btile) {
    const int width = std::min<uint32_t>(btile, x.n - first);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, count, width, x.dim, -2., qx.data(), x.dim,
                x.row(first), x.dim, 0., sx.data(), width);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, count, width, y.dim, -2., qy.data(), y.dim,
                y.row(first), y.dim, 0., sy.data(), width);
#pragma omp parallel for schedule(static)
    for (int r = 0; r < count; ++r) {
      const uint32_t q = ids[start + r];
      auto &heaps = best[r];
      std::array<double, weights> cutoff{};
      auto update_cutoff = [&]() {
        for (int a = 0; a < weights; ++a)
          cutoff[a] = heaps[a].size() == top ? heaps[a].front().first
                                             : std::numeric_limits<double>::infinity();
      };
      update_cutoff();
      const auto *px = sx.data() + size_t(r) * width;
      const auto *py = sy.data() + size_t(r) * width;
      for (int i = 0; i < width; ++i) {
        const uint32_t p = first + i;
        if (p == q)
          continue;
        // Bound cancellation in the FP64 dot-product screen. Score every contender
        // using coordinate differences before including it in any top-40.
        const double nx = x.norm[q] + x.norm[p], ny = y.norm[q] + y.norm[p];
        const double margin = 1.0e-10;
        const double lx = std::sqrt(std::max(0., px[i] + nx - margin * nx));
        const double ly = std::sqrt(std::max(0., py[i] + ny - margin * ny));
        bool possibly_near = false;
        for (int a = 0; a < weights; ++a)
          possibly_near |= alpha[a] * lx + (1 - alpha[a]) * ly <= cutoff[a];
        if (!possibly_near)
          continue;
        const double dx = x.distance(q, p), dy = y.distance(q, p);
        for (int a = 0; a < weights; ++a)
          push(heaps[a], {alpha[a] * dx + (1 - alpha[a]) * dy, p});
        update_cutoff();
      }
    }
  }
  for (auto &result : best) {
    for (auto &heap : result) {
      check(heap.size() == top, "short top-40");
      std::sort_heap(heap.begin(), heap.end());
      for (const auto &hit : heap)
        out.write(reinterpret_cast<const char *>(&hit.second), sizeof(uint32_t));
    }
  }
}
} // namespace

int main(int argc, char **argv) {
  try {
    check(argc == 5, "usage: exact_top40 X Y sample_ids.u32 out.bin");
    Matrix x(argv[1]), y(argv[2]);
    check(x.n == y.n, "X/Y row count");
    std::ifstream input(argv[3], std::ios::binary);
    std::vector<uint32_t> ids;
    uint32_t id;
    while (input.read(reinterpret_cast<char *>(&id), sizeof(id))) {
      check(id < x.n, "sample ID");
      ids.push_back(id);
    }
    check(input.eof() && input.gcount() == 0 && !ids.empty(), "malformed sample");
    const uint32_t header[]{uint32_t(ids.size()), weights, top};
    constexpr size_t row_bytes = top * weights * sizeof(uint32_t);
    size_t done = 0;
    struct stat st {};
    if (stat(argv[4], &st) == 0) {
      uint32_t saved[3]{};
      std::ifstream old(argv[4], std::ios::binary);
      check(bool(old.read(reinterpret_cast<char *>(saved), sizeof(saved))) &&
                std::equal(saved, saved + 3, header), "resume header");
      done = (st.st_size - sizeof(header)) / row_bytes;
      check(done <= ids.size() && truncate(argv[4], sizeof(header) + done * row_bytes) == 0,
            "resume size");
    } else {
      std::ofstream initial(argv[4], std::ios::binary);
      initial.write(reinterpret_cast<const char *>(header), sizeof(header));
      check(bool(initial), "write header");
    }
    std::ofstream out(argv[4], std::ios::binary | std::ios::app);
    for (size_t r = done; r < ids.size(); r += qtile) {
      int count = std::min<size_t>(qtile, ids.size() - r);
      closest(x, y, ids, r, count, out);
      out.flush();
      check(bool(out), "write exact results");
      fprintf(stderr, "EXACT_TOP40 %zu/%zu\n", r + count, ids.size());
    }
    return 0;
  } catch (const std::exception &e) {
    fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
