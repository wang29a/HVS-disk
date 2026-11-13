#include <omp.h>
#include "hot_reader.h"
#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  const std::string dir = argv[1];
  std::vector<char> expected(2 * SECTOR_LEN);
  for (size_t i = 0; i < expected.size(); ++i)
    expected[i] = static_cast<char>((i * 7 + i / SECTOR_LEN) % 251);
  for (const auto &name : { "coord.bin", "merge-topo.bin" }) {
    std::ofstream f(dir + "/" + name, std::ios::binary);
    f.write(expected.data(), expected.size());
    if (!f)
      return 3;
  }
  HotReader reader;
  reader.coord_file_desc = ::open((dir + "/coord.bin").c_str(), O_RDONLY);
  reader.topo_file_desc = ::open((dir + "/merge-topo.bin").c_str(), O_RDONLY);
  reader.attach("all-hot", dir, "merge");
  omp_set_num_threads(4);
  int errors = 0;
#pragma omp parallel for reduction(+ : errors)
  for (int i = 0; i < 40; ++i) {
    std::vector<char> a(SECTOR_LEN), b(SECTOR_LEN);
    std::vector<IORequest> topo, coord;
    const uint64_t offset = (i % 2) * SECTOR_LEN;
    topo.emplace_back(offset, SECTOR_LEN, a.data(), offset, SECTOR_LEN);
    coord.emplace_back(offset, SECTOR_LEN, b.data(), offset, SECTOR_LEN);
    reader.send_io(topo, nullptr, false, reader.topo_file_desc);
    reader.poll_wait(nullptr);
    reader.read_fd(reader.coord_file_desc, coord, nullptr);
    errors += std::memcmp(a.data(), expected.data() + offset, SECTOR_LEN) != 0;
    errors += std::memcmp(b.data(), expected.data() + offset, SECTOR_LEN) != 0;
  }
  auto count = reader.totals();
  if (errors || count.memory_topology != 40 || count.memory_coordinate != 40 ||
      count.ssd_topology || count.ssd_coordinate ||
      count.memory_bytes != 80 * SECTOR_LEN)
    return 4;
  reader.reset_counters();
  bool bounds_rejected = false, completion_rejected = false;
  std::vector<char> b(SECTOR_LEN);
  std::vector<IORequest> invalid;
  invalid.emplace_back(expected.size(), SECTOR_LEN, b.data(), 0, SECTOR_LEN);
  try {
    reader.read_fd(reader.coord_file_desc, invalid, nullptr);
  }
  catch (const std::runtime_error &) {
    bounds_rejected = true;
  }
  try {
    reader.poll_wait(nullptr);
  }
  catch (const std::runtime_error &) {
    completion_rejected = true;
  }
  if (!bounds_rejected || !completion_rejected)
    return 5;
  ::close(reader.coord_file_desc);
  ::close(reader.topo_file_desc);
  std::cout << "HOT_READER_TEST_PASS: concurrent copies, exact bytes, request "
               "accounting, bounds and completions" << std::endl;
  return 0;
}
