#include "pq_table.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
static void check(bool b, const char *s) {
  if (!b)
    throw std::runtime_error(s);
}
static std::vector<float> matrix(const std::string &p, uint32_t &n,
                                 uint32_t &d) {
  std::ifstream f(p, std::ios::binary);
  f.read(reinterpret_cast<char *>(&n), 4);
  f.read(reinterpret_cast<char *>(&d), 4);
  check(bool(f) && n && d, "matrix header");
  std::vector<float> a(uint64_t(n) * d);
  f.read(reinterpret_cast<char *>(a.data()), a.size() * 4);
  check(bool(f) && f.peek() == EOF, "matrix length");
  return a;
}
int main(int argc, char **argv) {
  try {
    check(argc == 4, "inputs frozen-pq output");
    std::string input = argv[1], pqroot = argv[2], out = argv[3];
    for (auto pair :
         { std::make_pair("x", "emb"), std::make_pair("y", "loc") }) {
      uint32_t n, d;
      auto data = matrix(input + "/insert-" + pair.first + ".bin", n, d);
      pipeann::FixedChunkPQTable<float> table;
      table.load_pq_centroid_bin(
          (pqroot + "/" + pair.second + "_pq_pivots.bin").c_str(), 64);
      check(d == table.get_dim(), "dimension mismatch");
      std::vector<uint8_t> codes(uint64_t(n) * 64);
// Audit output only: never injected into the update executor.
#pragma omp parallel for schedule(static)
      for (uint32_t i = 0; i < n; ++i)
        table.deflate_vec(data.data() + uint64_t(i) * d,
                          codes.data() + uint64_t(i) * 64);
      std::ofstream f(out + "/" + pair.second + "_pq_compressed.bin",
                      std::ios::binary);
      uint32_t chunks = 64;
      f.write(reinterpret_cast<char *>(&n), 4);
      f.write(reinterpret_cast<char *>(&chunks), 4);
      f.write(reinterpret_cast<char *>(codes.data()), codes.size());
      check(bool(f), "code write");
      std::cout << pair.second << " ONLINE_PQ_AUDIT_ENCODED " << n << std::endl;
    }
  }
  catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
