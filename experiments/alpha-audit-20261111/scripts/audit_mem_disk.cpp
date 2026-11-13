// Audit the serialized int8 memory graph against the Full fixed-page graph.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using U = uint64_t;
using Bits = unsigned __int128;
constexpr U PAGE = 8192;

struct Map {
  int fd;
  U size;
  const uint8_t *data;
  explicit Map(const char *path) {
    fd = open(path, O_RDONLY);
    struct stat st {};
    if (fd < 0 || fstat(fd, &st) || !st.st_size)
      throw std::runtime_error("cannot open nonempty input");
    size = st.st_size;
    void *p = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED)
      throw std::runtime_error("cannot map input");
    data = static_cast<const uint8_t *>(p);
  }
  ~Map() {
    munmap(const_cast<uint8_t *>(data), size);
    close(fd);
  }
  template <class T> T read(U offset) const {
    if (offset > size || sizeof(T) > size - offset)
      throw std::runtime_error("truncated record");
    T value;
    std::memcpy(&value, data + offset, sizeof(T));
    return value;
  }
};

unsigned count(Bits bits) {
  return __builtin_popcountll(uint64_t(bits)) +
         __builtin_popcountll(uint64_t(bits >> 64));
}

Bits interval(unsigned lo, unsigned hi, bool endpoints) {
  unsigned width = hi - lo + unsigned(endpoints);
  return width ? ((Bits(1) << width) - 1) << lo : 0;
}

int main(int argc, char **argv) {
  try {
    if (argc != 5)
      throw std::runtime_error(
          "usage: audit_mem_disk mem.index meta topo report.json");
    Map mem(argv[1]), meta(argv[2]), topo(argv[3]);
    U nodes = mem.read<U>(0);
    unsigned degree = mem.read<unsigned>(8), slots = mem.read<unsigned>(12);
    U disk_npp = meta.read<U>(20);
    U width = 4 + U(degree) * (4 + 2 * slots);
    if (meta.size != PAGE || nodes != meta.read<unsigned>(0) ||
        degree != meta.read<unsigned>(12) || slots != meta.read<unsigned>(16) ||
        !nodes || !degree || !slots || width > PAGE ||
        disk_npp != PAGE / width ||
        topo.size != ((nodes + disk_npp - 1) / disk_npp) * PAGE)
      throw std::runtime_error("Full layout metadata mismatch");
    U pos = 16, edges = 0, pairs = 0, id_mismatch = 0, range_mismatch = 0;
    U narrowed_edges = 0, widened_edges = 0, lost_units = 0, added_units = 0;
    U lost_grid_points = 0, added_grid_points = 0, padding_nonzero = 0;
    U declared_zero_pairs = 0, reader_hidden_pairs = 0;
    for (U id = 0; id < nodes; ++id) {
      unsigned nn = mem.read<unsigned>(pos);
      pos += 4;
      U disk_pos = (id / disk_npp) * PAGE + (id % disk_npp) * width;
      if (nn > degree || nn != topo.read<unsigned>(disk_pos))
        throw std::runtime_error("node degree mismatch");
      edges += nn;
      for (unsigned j = 0; j < nn; ++j) {
        unsigned nbr = mem.read<unsigned>(pos),
                 nrange = mem.read<unsigned>(pos + 4);
        pos += 8;
        U edge = disk_pos + 4 + U(j) * (4 + 2 * slots);
        if (nbr >= nodes || nrange > slots)
          throw std::runtime_error("invalid memory graph ID or interval count");
        id_mismatch += nbr != topo.read<unsigned>(edge);
        pairs += nrange;
        Bits before = 0, after = 0, grid_before = 0, grid_after = 0;
        bool stopped = false, edge_ranges_differ = false;
        unsigned previous_start = 0;
        for (unsigned k = 0; k < slots; ++k) {
          unsigned lo = 0, hi = 0;
          if (k < nrange) {
            lo = mem.read<uint8_t>(pos++);
            hi = mem.read<uint8_t>(pos++);
            if (lo > hi || hi > 100 || (k && lo < previous_start))
              throw std::runtime_error("invalid source int8 interval");
            previous_start = lo;
            declared_zero_pairs += lo == 0 && hi == 0;
            before |= interval(lo, hi, false);
            grid_before |= interval(lo, hi, true);
          }
          unsigned a = topo.read<uint8_t>(edge + 4 + 2 * k);
          unsigned b = topo.read<uint8_t>(edge + 5 + 2 * k);
          if (a > b || b > 100)
            throw std::runtime_error("invalid disk int8 interval");
          edge_ranges_differ |= lo != a || hi != b;
          if (a == 0 && b == 0)
            stopped = true;
          if (!stopped) {
            after |= interval(a, b, false);
            grid_after |= interval(a, b, true);
          } else if (k < nrange) {
            reader_hidden_pairs++;
          }
        }
        range_mismatch += edge_ranges_differ;
        U lost = count(before & ~after), added = count(after & ~before);
        U glost = count(grid_before & ~grid_after),
          gadded = count(grid_after & ~grid_before);
        narrowed_edges += lost || glost;
        widened_edges += added || gadded;
        lost_units += lost;
        added_units += added;
        lost_grid_points += glost;
        added_grid_points += gadded;
      }
      for (U p = disk_pos + 4 + U(nn) * (4 + 2 * slots); p < disk_pos + width;
           ++p)
        padding_nonzero += topo.read<uint8_t>(p) != 0;
    }
    if (pos != mem.size)
      throw std::runtime_error(
          "memory graph has trailing bytes or wrong encoding");
    for (U page = 0; page < topo.size / PAGE; ++page) {
      U used = std::min(disk_npp, nodes - page * disk_npp) * width;
      for (U p = used; p < PAGE; ++p)
        padding_nonzero += topo.read<uint8_t>(page * PAGE + p) != 0;
    }
    bool pass = !id_mismatch && !range_mismatch && !narrowed_edges &&
                !widened_edges && !padding_nonzero;
    std::ofstream out(argv[4]);
    out << "{\n  \"status\":\"" << (pass ? "PASS" : "FAIL")
        << "\",\n  \"source_encoding\":\"int8 endpoints, uint32 range counts\""
        << ",\n  \"float_to_int8_audit\":\"unavailable without "
           "pre-serialization float intervals\""
        << ",\n  \"nodes\":" << nodes << ",\n  \"edges\":" << edges
        << ",\n  \"stored_pairs\":" << pairs
        << ",\n  \"neighbor_id_mismatches\":" << id_mismatch
        << ",\n  \"range_byte_mismatches\":" << range_mismatch
        << ",\n  \"coverage_narrowed_edges\":" << narrowed_edges
        << ",\n  \"coverage_widened_edges\":" << widened_edges
        << ",\n  \"summed_lost_alpha_width\":" << lost_units / 100.0
        << ",\n  \"summed_added_alpha_width\":" << added_units / 100.0
        << ",\n  \"lost_integer_alpha_grid_points\":" << lost_grid_points
        << ",\n  \"added_integer_alpha_grid_points\":" << added_grid_points
        << ",\n  \"declared_zero_pairs\":" << declared_zero_pairs
        << ",\n  \"reader_hidden_pairs\":" << reader_hidden_pairs
        << ",\n  \"nonzero_padding_bytes\":" << padding_nonzero << "\n}\n";
    if (!out)
      throw std::runtime_error("cannot write audit report");
    return pass ? 0 : 2;
  }
  catch (const std::exception &error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
