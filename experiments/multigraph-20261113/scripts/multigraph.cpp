// Disk adaptation of the pointer beam engine. DRAM fields are direct pointers;
// SSD fields use 8192-byte O_DIRECT AIO pages, cached only within one query.
#include "ssd_index.h"
#include "neighbor.h"
#include "tsl/robin_set.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <libaio.h>
#include <memory>
#include <numeric>
#include <set>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unordered_map>
#include <unistd.h>
using U = uint64_t;
using Clock = std::chrono::steady_clock;
constexpr U PAGE = 8192;
double usec(Clock::time_point p) {
  return std::chrono::duration<double, std::micro>(Clock::now() - p).count();
}
double epoch() {
  return std::chrono::duration<double>(
      std::chrono::system_clock::now().time_since_epoch()).count();
}
template <class T> T get(std::istream &f) {
  T x{};
  if (!f.read((char *)&x, sizeof(x)))
    throw std::runtime_error("truncated input");
  return x;
}
template <class T> void put(std::ostream &f, T x) {
  f.write((char *)&x, sizeof(x));
}
unsigned u32(const void *p) {
  unsigned x;
  memcpy(&x, p, 4);
  return x;
}
template <class T>
std::vector<T> matrix(const std::string &p, unsigned &n, unsigned &d) {
  std::ifstream f(p, std::ios::binary);
  n = get<unsigned>(f);
  d = get<unsigned>(f);
  if (!n || !d || U(n) * d > (U(1) << 36))
    throw std::runtime_error("matrix shape");
  std::vector<T> x(U(n) * d);
  if (!f.read((char *)x.data(), x.size() * sizeof(T)) || f.peek() != EOF)
    throw std::runtime_error("matrix size");
  return x;
}
struct File {
  std::string path;
  int fd = -1;
  U bytes = 0;
  uint8_t *ram = nullptr;
  File(const std::string &p, bool resident) : path(p) {
    fd = open(p.c_str(), O_RDONLY | O_DIRECT);
    struct stat s {};
    if (fd < 0 || fstat(fd, &s) || s.st_size % PAGE)
      throw std::runtime_error("aligned file " + p);
    bytes = s.st_size;
    if (resident) {
      if (posix_memalign((void **)&ram, PAGE, bytes))
        throw std::runtime_error("resident allocation");
      for (U a = 0; a < bytes;) {
        U k = std::min(U(8 << 20), bytes - a);
        if (pread(fd, ram + a, k, a) != ssize_t(k))
          throw std::runtime_error("resident read");
        a += k;
      }
    }
  }
  ~File() {
    free(ram);
    if (fd >= 0)
      close(fd);
  }
  File(const File &) = delete;
};
struct GStat {
  U expanded = 0, pq = 0, exact = 0, pages = 0, submitted = 0, inspected = 0,
    rejected = 0;
  double wait = 0;
};
struct Answer {
  std::array<unsigned, 10> ids{};
  std::array<float, 10> distances{};
  std::array<GStat, 8> graph{};
  U before = 0, after = 0, reranked = 0, coord_pages = 0, coord_submitted = 0;
  unsigned hits = 0;
  double latency = 0, coord_wait = 0, exact_us = 0;
  std::vector<std::vector<unsigned> > batches;
  std::vector<unsigned> candidates, coords;
};
// One workspace per OpenMP worker; the AIO context and buffer allocations are
// outside the measured query. Query reset clears every map, with no shared
// cache.
struct Workspace {
  io_context_t ctx = 0;
  std::vector<uint8_t *> buffers;
  U used = 0;
  std::unordered_map<U, uint8_t *> pages;
  std::vector<float> t1, t2, de, dy;
  std::vector<uint8_t> c1, c2;
  Workspace()
      : t1(64 * 256), t2(64 * 256), de(4096), dy(4096), c1(4096 * 64),
        c2(4096 * 64) {
    if (io_setup(128, &ctx))
      throw std::runtime_error("io_setup");
  }
  ~Workspace() {
    io_destroy(ctx);
    for (auto p : buffers)
      free(p);
  }
  void reset() {
    pages.clear();
    used = 0;
  }
  uint8_t *alloc() {
    if (used == buffers.size()) {
      uint8_t *p = nullptr;
      if (posix_memalign((void **)&p, PAGE, PAGE))
        throw std::runtime_error("page buffer");
      buffers.push_back(p);
    }
    return buffers[used++];
  }
  void fetch(File &f, unsigned key, const std::vector<U> &requested,
             U &distinct, U &submitted, double &wait) {
    std::vector<U> keys = requested;
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    std::vector<std::pair<U, uint8_t *> > miss;
    for (U id : keys) {
      if ((id + 1) * PAGE > f.bytes)
        throw std::runtime_error("page bounds");
      U k = id * 16 + key;
      if (!pages.count(k)) {
        ++distinct;
        if (f.ram)
          pages.emplace(k, f.ram + id * PAGE);
        else {
          auto p = alloc();
          pages.emplace(k, p);
          miss.emplace_back(id, p);
        }
      }
    }
    for (U first = 0; first < miss.size(); first += 128) {
      U n = std::min(U(128), U(miss.size()) - first);
      std::vector<iocb> cb(n);
      std::vector<iocb *> ptr(n);
      std::vector<io_event> ev(n);
      for (U j = 0; j < n; j++) {
        io_prep_pread(&cb[j], f.fd, miss[first + j].second, PAGE,
                      miss[first + j].first * PAGE);
        ptr[j] = &cb[j];
      }
      auto begin = Clock::now();
      if (io_submit(ctx, n, ptr.data()) != int(n))
        throw std::runtime_error("short AIO submit");
      U done = 0;
      while (done < n) {
        int got =
            io_getevents(ctx, n - done, n - done, ev.data() + done, nullptr);
        if (got <= 0)
          throw std::runtime_error("AIO wait");
        done += got;
      }
      wait += usec(begin);
      submitted += n;
      for (auto &e : ev)
        if (e.res != PAGE || e.res2)
          throw std::runtime_error("AIO completion");
    }
  }
  const uint8_t *field(File &f, unsigned key, U page, U off) const {
    return f.ram ? f.ram + page * PAGE + off : pages.at(page * 16 + key) + off;
  }
};
struct Graph {
  std::string name;
  unsigned n = 0, d1 = 0, d2 = 0, R = 0, M = 0, key;
  std::vector<unsigned> entries;
  std::unique_ptr<File> topo;
  std::vector<uint8_t> side;
  U edges = 0;
  Graph(const std::string &label, unsigned index, const std::string &meta,
        const std::string &path, const std::string &sidepath, unsigned slots,
        bool resident)
      : name(label), M(slots), key(index) {
    std::ifstream f(meta, std::ios::binary);
    n = get<unsigned>(f);
    d1 = get<unsigned>(f);
    d2 = get<unsigned>(f);
    R = get<unsigned>(f);
    if (get<unsigned>(f) != 0)
      throw std::runtime_error("topology must be compact");
    U npp = get<U>(f);
    unsigned ne = get<unsigned>(f);
    if (!n || !d1 || !d2 || !R || R > 1000 || npp != PAGE / (4 + R * 4) ||
        ne > 1000 || !ne)
      throw std::runtime_error("compact metadata");
    for (unsigned j = 0; j < ne; j++) {
      auto id = get<unsigned>(f);
      if (id >= n)
        throw std::runtime_error("entry");
      entries.push_back(id);
    }
    topo.reset(new File(path, resident));
    if (topo->bytes != ((U(n) + npp - 1) / npp) * PAGE)
      throw std::runtime_error("topology bytes");
    if (sidepath != "-") {
      std::ifstream s(sidepath, std::ios::binary);
      side.resize(U(n) * R * 2 * M);
      if (!M || !s.read((char *)side.data(), side.size()) || s.peek() != EOF)
        throw std::runtime_error("sidecar size");
    } else if (M)
      throw std::runtime_error("missing ranges");
  }
  U npp() const { return PAGE / (4 + R * 4); }
  const uint8_t *row(unsigned id, const Workspace &w) const {
    return w.field(*topo, key, id / npp(), (id % npp()) * (4 + 4 * R));
  }
  void fetch(const std::vector<unsigned> &ids, Workspace &w, GStat &s) {
    std::vector<U> pages;
    for (auto id : ids) {
      if (id >= n)
        throw std::runtime_error("node bounds");
      pages.push_back(id / npp());
    }
    w.fetch(*topo, key, pages, s.pages, s.submitted, s.wait);
  }
  bool active(unsigned id, unsigned edge, float alpha) const {
    if (side.empty())
      throw std::runtime_error("filter without sidecar");
    const auto *p = side.data() + (U(id) * R + edge) * 2 * M;
    for (unsigned j = 0; j < M; j++) {
      int lo = p[2 * j], hi = p[2 * j + 1];
      if ((!lo && !hi) || lo > 100 || hi > 100)
        break;
      float a100 = alpha * 100.f;
      if (a100 >= lo && a100 <= hi)
        return true;
      if (a100 < lo)
        break;
    }
    return false;
  }
};
struct Job {
  std::string label, method;
  unsigned L, kx, ky;
  float weight;
};
struct Engine {
  unsigned n, d1, d2;
  U cnpp;
  std::string mode, nav;
  std::array<std::unique_ptr<Graph>, 8> graphs;
  std::unique_ptr<File> coords;
  pipeann::FixedChunkPQTable<float> pq1, pq2;
  std::vector<uint8_t> code1, code2;
  std::unique_ptr<pipeann::Distance<float> > distance;
  U resident_bytes = 0, pq_bytes = 0, side_bytes = 0, topology_bytes = 0;
  static unsigned index(const std::string &name) {
    static const std::vector<std::string> names = { "x",   "y",   "a01", "a03",
                                                    "a05", "a07", "a09", "u" };
    auto it = std::find(names.begin(), names.end(), name);
    if (it == names.end())
      throw std::runtime_error("graph name");
    return it - names.begin();
  }
  static unsigned route(float alpha) {
    static const float a[] = { .1f, .3f, .5f, .7f, .9f };
    unsigned best = 0;
    for (unsigned j = 1; j < 5; j++)
      if (std::abs(alpha - a[j]) < std::abs(alpha - a[best]) - 1e-6f)
        best = j;
    return best + 2;
  }
  Engine(const std::string &manifest, const std::string &m,
         const std::string &v, const std::set<unsigned> &needed)
      : mode(m), nav(v) {
    if (mode != "dram" && mode != "ssd" && mode != "topo")
      throw std::runtime_error("medium");
    if (nav != "pq" && nav != "exact")
      throw std::runtime_error("navigation");
    if (nav == "exact" && mode != "dram")
      throw std::runtime_error(
          "exact navigation is a correctness-only DRAM arm");
    std::ifstream f(manifest);
    std::string magic, coordinate, prefix;
    unsigned count;
    f >> magic >> n >> d1 >> d2 >> count >> coordinate >> prefix;
    if (magic != "HVS_MULTI_V1" || count > 8 || d1 % 8 || d2 % 8 || !d1 || !d2)
      throw std::runtime_error("manifest");
    cnpp = PAGE / (4 * (d1 + d2));
    coords.reset(new File(coordinate, mode == "dram"));
    if (!cnpp || coords->bytes != ((U(n) + cnpp - 1) / cnpp) * PAGE)
      throw std::runtime_error("coordinate size");
    for (unsigned j = 0; j < count; j++) {
      std::string name, meta, topo, side;
      unsigned slots;
      f >> name >> meta >> topo >> side >> slots;
      auto i = index(name);
      if (needed.count(i)) {
        graphs[i]
            .reset(new Graph(name, i, meta, topo, side, slots, mode != "ssd"));
        auto &g = *graphs[i];
        if (g.n != n || g.d1 != d1 || g.d2 != d2)
          throw std::runtime_error("graph dimensions");
        topology_bytes += g.topo->bytes;
        side_bytes += g.side.size();
        if (g.topo->ram)
          resident_bytes += g.topo->bytes;
      }
    }
    if (!f)
      throw std::runtime_error("manifest read");
    for (auto i : needed)
      if (!graphs[i])
        throw std::runtime_error("needed graph missing");
    if (coords->ram)
      resident_bytes += coords->bytes;
    if (nav == "pq") {
      size_t rows, chunks;
      pipeann::load_bin<uint8_t>(prefix + "emb_pq_compressed.bin", code1, rows,
                                 chunks);
      if (rows != n || chunks != 64)
        throw std::runtime_error("X PQ codes");
      pipeann::load_bin<uint8_t>(prefix + "loc_pq_compressed.bin", code2, rows,
                                 chunks);
      if (rows != n || chunks != 64)
        throw std::runtime_error("Y PQ codes");
      pq1.load_pq_centroid_bin((prefix + "emb_pq_pivots.bin").c_str(), 64);
      pq2.load_pq_centroid_bin((prefix + "loc_pq_pivots.bin").c_str(), 64);
      if (pq1.ndims != d1 || pq2.ndims != d2)
        throw std::runtime_error("PQ dimensions");
      pq_bytes = code1.size() + code2.size();
    }
    distance.reset(pipeann::get_distance_function<float>(pipeann::Metric::L2));
  }
  float fuse(float x, float y, float alpha) const {
    return alpha * std::sqrt(std::max(0.f, x)) +
           (1 - alpha) * std::sqrt(std::max(0.f, y));
  }
  const float *vector(unsigned id, unsigned space, const Workspace &w) const {
    return reinterpret_cast<const float *>(
        w.field(*coords, 8, id / cnpp,
                (id % cnpp) * 4 * (d1 + d2) + 4 * (space ? d1 : 0)));
  }
  void coord_fetch(const std::vector<unsigned> &ids, Workspace &w, Answer &a) {
    std::vector<U> pages;
    for (auto id : ids)
      pages.push_back(id / cnpp);
    w.fetch(*coords, 8, pages, a.coord_pages, a.coord_submitted, a.coord_wait);
  }
  std::vector<pipeann::Neighbor> traverse(unsigned gi, int space, float alpha,
                                          unsigned L, bool filter,
                                          const float *qx, const float *qy,
                                          Workspace &w, Answer &a, bool trace) {
    auto &g = *graphs[gi];
    auto &s = a.graph[gi];
    auto scores = [&](const std::vector<unsigned> &ids) {
      std::vector<float> v(ids.size());
      if (ids.size() > 4096)
        throw std::runtime_error("neighbor batch");
      if (nav == "exact") {
        for (U j = 0; j < ids.size(); j++) {
          auto id = ids[j];
          float x =
              space == 1 ? 0 : distance->compare(qx, vector(id, 0, w), d1);
          float y =
              space == 0 ? 0 : distance->compare(qy, vector(id, 1, w), d2);
          v[j] = space == 0 ? std::sqrt(std::max(0.f, x))
                            : space == 1 ? std::sqrt(std::max(0.f, y))
                                         : fuse(x, y, alpha);
          s.exact += space < 0 ? 2 : 1;
        }
      } else {
        if (space != 1) {
          aggregate_coords(ids.data(), ids.size(), code1.data(), 64,
                           w.c1.data());
          pq_dist_lookup(w.c1.data(), ids.size(), 64, w.t1.data(), w.de.data());
        }
        if (space != 0) {
          aggregate_coords(ids.data(), ids.size(), code2.data(), 64,
                           w.c2.data());
          pq_dist_lookup(w.c2.data(), ids.size(), 64, w.t2.data(), w.dy.data());
        }
        for (U j = 0; j < ids.size(); j++)
          v[j] = space == 0 ? std::sqrt(std::max(0.f, w.de[j]))
                            : space == 1 ? std::sqrt(std::max(0.f, w.dy[j]))
                                         : fuse(w.de[j], w.dy[j], alpha);
        s.pq += ids.size() * (space < 0 ? 2 : 1);
      }
      return v;
    };
    if (g.entries.size() > L)
      throw std::runtime_error("L below entry count");
    std::vector<pipeann::Neighbor> ret(
        std::max(U(L) * 10, g.entries.size() + 1));
    tsl::robin_set<U> visited;
    unsigned cur = 0;
    auto entryscores = scores(g.entries);
    for (U j = 0; j < g.entries.size(); j++) {
      ret[cur++] = pipeann::Neighbor(g.entries[j], entryscores[j], true);
      visited.insert(g.entries[j]);
    }
    std::sort(ret.begin(), ret.begin() + cur);
    unsigned k = 0;
    while (k < cur) {
      unsigned nk = cur, marker = k, seen = 0;
      std::vector<unsigned> frontier;
      while (marker < cur && frontier.size() < 4 && seen < 4) {
        if (ret[marker].flag) {
          ++seen;
          frontier.push_back(ret[marker].id);
          ret[marker].flag = false;
        }
        ++marker;
      }
      if (trace) {
        std::vector<unsigned> b{ gi };
        b.insert(b.end(), frontier.begin(), frontier.end());
        a.batches.push_back(b);
      }
      g.fetch(frontier, w, s);
      s.expanded += frontier.size();
      for (auto id : frontier) {
        const auto *p = g.row(id, w);
        unsigned nn = u32(p);
        if (nn > g.R)
          throw std::runtime_error("degree");
        std::vector<unsigned> nbrs;
        for (unsigned m = 0; m < nn; m++) {
          unsigned nb = u32(p + 4 + 4 * m);
          if (nb >= n || nb == id)
            throw std::runtime_error("edge ID");
          ++s.inspected;
          if (filter && !g.active(id, m, alpha)) {
            ++s.rejected;
            continue;
          }
          if (visited.find(nb) != visited.end())
            continue;
          visited.insert(nb);
          nbrs.push_back(nb);
        }
        auto d = scores(nbrs);
        for (U j = 0; j < nbrs.size(); j++)
          if (cur < L || d[j] < ret[cur - 1].distance) {
            auto pos = pipeann::InsertIntoPool(
                ret.data(), cur, pipeann::Neighbor(nbrs[j], d[j], true));
            if (cur < L)
              ++cur;
            if (pos < nk)
              nk = pos;
          }
      }
      if (nk <= k)
        k = nk;
      else
        ++k;
    }
    ret.resize(cur);
    return ret;
  }
  Answer query(const Job &job, const float *qx, const float *qy, float alpha,
               Workspace &w, bool trace) {
    Answer a;
    auto begin = Clock::now();
    w.reset();
    if (nav == "pq") {
      pq1.populate_chunk_distances(qx, w.t1.data());
      pq2.populate_chunk_distances(qy, w.t2.data());
    }
    bool multi = job.method == "M_equal" || job.method == "M_weighted" ||
                 job.method == "M_endpoint";
    std::vector<unsigned> ids;
    std::vector<pipeann::Neighbor> ret;
    if (multi) {
      unsigned kx = job.kx, ky = job.ky;
      if (job.method == "M_endpoint" && alpha <= .05f) {
        ky = kx + ky;
        kx = 0;
      }
      if (job.method == "M_endpoint" && alpha >= .95f) {
        kx = kx + ky;
        ky = 0;
      }
      if (std::max(kx, ky) > job.L)
        throw std::runtime_error("candidate budget above L");
      if (kx) {
        ret = traverse(0, 0, alpha, job.L, false, qx, qy, w, a, trace);
        for (unsigned j = 0; j < std::min(kx, unsigned(ret.size())); j++)
          ids.push_back(ret[j].id);
      }
      if (ky) {
        ret = traverse(1, 1, alpha, job.L, false, qx, qy, w, a, trace);
        for (unsigned j = 0; j < std::min(ky, unsigned(ret.size())); j++)
          ids.push_back(ret[j].id);
      }
    } else {
      unsigned gi = job.method == "F" ? 4 : job.method == "DEG_on" ||
                                                    job.method == "DEG_off"
                                                ? 7
                                                : route(alpha);
      if (job.method == "Or_ideal") {
        const float values[] = { .1f, .3f, .5f, .7f, .9f };
        if (std::abs(alpha - values[gi - 2]) > 1e-6)
          throw std::runtime_error("oracle used off grid");
      }
      ret = traverse(gi, -1, alpha, job.L, job.method == "DEG_on", qx, qy, w, a,
                     trace);
      if (nav == "exact") {
        if (ret.size() < 10)
          throw std::runtime_error("exact top10");
        for (unsigned j = 0; j < 10; j++) {
          a.ids[j] = ret[j].id;
          a.distances[j] = ret[j].distance;
        }
        a.latency = usec(begin);
        return a;
      }
      for (unsigned j = 0; j < std::min(job.kx, unsigned(ret.size())); j++)
        ids.push_back(ret[j].id);
    }
    a.before = ids.size();
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    a.after = ids.size();
    a.reranked = ids.size();
    if (ids.size() < 10)
      throw std::runtime_error("insufficient merged candidates");
    if (trace) {
      a.candidates = ids;
      a.coords = ids;
    }
    std::vector<std::pair<unsigned, float> > exact;
    for (U first = 0; first < ids.size(); first += 256) {
      std::vector<unsigned> batch(ids.begin() + first,
                                  ids.begin() +
                                      std::min(U(ids.size()), first + 256));
      coord_fetch(batch, w, a);
      auto start = Clock::now();
      for (auto id : batch) {
        float x = distance->compare(qx, vector(id, 0, w), d1),
              y = distance->compare(qy, vector(id, 1, w), d2);
        exact.emplace_back(id, fuse(x, y, alpha));
      }
      a.exact_us += usec(start);
    }
    std::sort(exact.begin(), exact.end(), [](const auto &a, const auto &b) {
      return a.second < b.second || (a.second == b.second && a.first < b.first);
    });
    for (unsigned j = 0; j < 10; j++) {
      a.ids[j] = exact[j].first;
      a.distances[j] = exact[j].second;
    }
    a.latency = usec(begin);
    return a;
  }
};
std::set<unsigned> needed(const std::vector<Job> &jobs,
                          const std::vector<float> &alphas,
                          const std::vector<unsigned> &ids) {
  std::set<unsigned> s;
  for (auto &j : jobs) {
    if (j.method.find("M_") == 0) {
      s.insert(0);
      s.insert(1);
    } else if (j.method == "F")
      s.insert(4);
    else if (j.method == "DEG_on" || j.method == "DEG_off")
      s.insert(7);
    else if (j.method == "Or_ideal" || j.method == "Or_nearest")
      for (auto id : ids)
        s.insert(Engine::route(alphas[id]));
    else
      throw std::runtime_error("method");
  }
  return s;
}
void traces(const std::string &path, const std::vector<Answer> &answers) {
  std::ofstream f(path, std::ios::binary), ids(path + ".ids", std::ios::binary),
      d(path + ".dists", std::ios::binary),
      legacy(path + ".legacy", std::ios::binary);
  put<unsigned>(f, 0x4d475231);
  put<unsigned>(f, answers.size());
  put<unsigned>(legacy, 0x48565331);
  put<unsigned>(legacy, answers.size());
  for (auto &a : answers) {
    put<unsigned>(f, a.batches.size());
    put<unsigned>(legacy, a.batches.size());
    for (auto &b : a.batches) {
      put<unsigned>(f, b.size());
      for (auto x : b)
        put(f, x);
      put<unsigned>(legacy, b.size() - 1);
      for (U j = 1; j < b.size(); j++)
        put(legacy, b[j]);
    }
    for (const auto *v : { &a.candidates, &a.coords }) {
      put<unsigned>(f, v->size());
      for (auto x : *v)
        put(f, x);
    }
    put<unsigned>(legacy, 0);
    ids.write((char *)a.ids.data(), 40);
    d.write((char *)a.distances.data(), 40);
  }
  if (!f || !ids || !d || !legacy)
    throw std::runtime_error("trace write");
}
int main(int argc, char **argv) {
  try {
    if (argc != 15)
      throw std::runtime_error("manifest qx qy alpha gt queryIDs output medium "
                               "threads seconds nav jobs trace-prefix-or-- "
                               "audit-returns");
    unsigned q, d, q2, d2, na, da, ng, dg;
    auto qx = matrix<float>(argv[2], q, d), qy = matrix<float>(argv[3], q2, d2),
         alpha = matrix<float>(argv[4], na, da);
    auto gt = matrix<unsigned>(argv[5], ng, dg);
    if (q != q2 || q != na || q != ng || da != 1 || dg != 10)
      throw std::runtime_error("query matrices");
    for (float a : alpha)
      if (!std::isfinite(a) || a < 0 || a > 1)
        throw std::runtime_error("alpha");
    std::vector<unsigned> ids;
    if (std::string(argv[6]) == "-") {
      ids.resize(q);
      std::iota(ids.begin(), ids.end(), 0);
    } else {
      std::ifstream f(argv[6]);
      unsigned i;
      while (f >> i) {
        if (i >= q)
          throw std::runtime_error("query ID");
        ids.push_back(i);
      }
    }
    if (ids.empty())
      throw std::runtime_error("empty cohort");
    std::ifstream jf(argv[12]);
    std::vector<Job> jobs;
    Job j;
    while (jf >> j.label >> j.method >> j.L >> j.kx >> j.ky >> j.weight) {
      if (j.L < 10 || j.L > 20000 || j.kx > j.L || j.ky > j.L)
        throw std::runtime_error("job budget");
      jobs.push_back(j);
    }
    if (jobs.empty() || !jf.eof())
      throw std::runtime_error("jobs");
    unsigned threads = std::stoul(argv[9]);
    double duration = std::stod(argv[10]);
    bool trace = std::string(argv[13]) != "-", audit = std::stoi(argv[14]);
    if (!threads || threads > 8 || duration < 0 || duration > 60 ||
        (trace && duration))
      throw std::runtime_error("run limits");
    omp_set_num_threads(threads);
    Engine e(argv[1], argv[8], argv[11], needed(jobs, alpha, ids));
    if (e.d1 != d || e.d2 != d2)
      throw std::runtime_error("query dims");
    std::vector<std::unique_ptr<Workspace> > work;
    for (unsigned i = 0; i < threads; i++)
      work.emplace_back(new Workspace());
    std::ofstream out(argv[7]);
    out << std::setprecision(17) << "[";
    bool first = true;
    for (auto &job : jobs) {
      std::vector<Answer> answers(ids.size());
      std::vector<double> lat;
      std::string failure;
      auto batch = [&](bool measured) {
#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t k = 0; k < int64_t(ids.size()); k++)
          try {
            unsigned id = ids[k];
            answers[k] =
                e.query(job, qx.data() + U(id) * d, qy.data() + U(id) * d2,
                        alpha[id], *work[omp_get_thread_num()], trace);
            for (auto nb : answers[k].ids)
              for (unsigned z = 0; z < 10; z++)
                if (nb == gt[U(id) * 10 + z]) {
                  answers[k].hits++;
                  break;
                }
          }
        catch (const std::exception &ex) {
#pragma omp critical
          failure = ex.what();
        }
        if (!failure.empty())
          throw std::runtime_error(failure);
        if (measured)
          for (auto &a : answers)
            lat.push_back(a.latency);
      };
      if (duration)
        batch(false);
      double begin = epoch();
      auto tick = Clock::now();
      double seconds;
      do {
        batch(true);
        seconds = usec(tick) / 1e6;
      } while (seconds < duration);
      double error = 0, contrib = 0;
      U valid = 0;
      if (audit) {
        for (U k = 0; k < ids.size(); k++) {
          unsigned qi = ids[k];
          for (unsigned z = 0; z < 10; z++) {
            auto nb = answers[k].ids[z];
            auto &w = *work[0];
            if (!e.coords->ram) {
              w.reset();
              Answer a;
              e.coord_fetch({ nb }, w, a);
            }
            double sx = 0, sy = 0;
            const float *x = e.vector(nb, 0, w), *y = e.vector(nb, 1, w);
            for (unsigned l = 0; l < d; l++) {
              double delta = double(qx[U(qi) * d + l]) - x[l];
              sx += delta * delta;
            }
            for (unsigned l = 0; l < d2; l++) {
              double delta = double(qy[U(qi) * d2 + l]) - y[l];
              sy += delta * delta;
            }
            double ref = double(alpha[qi]) * std::sqrt(sx) +
                         (1. - double(alpha[qi])) * std::sqrt(sy),
                   got = answers[k].distances[z];
            if (ref == 0) {
              if (got != 0)
                throw std::runtime_error("zero distance changed");
            } else
              error = std::max(error, std::abs(got - ref) / ref);
          }
        }
        if (error > 5e-6)
          throw std::runtime_error("float64 distance gate");
      }
      // Axis is defined on native GT, independent of method-returned neighbors.
      {
        auto &w = *work[0];
        for (auto qi : ids) {
          double sum = 0;
          unsigned good = 0;
          w.reset();
          Answer a;
          std::vector<unsigned> truth(gt.begin() + U(qi) * 10,
                                      gt.begin() + U(qi) * 10 + 10);
          e.coord_fetch(truth, w, a);
          for (auto nb : truth) {
            double sx = 0, sy = 0;
            const float *x = e.vector(nb, 0, w), *y = e.vector(nb, 1, w);
            for (unsigned l = 0; l < d; l++) {
              double delta = double(qx[U(qi) * d + l]) - x[l];
              sx += delta * delta;
            }
            for (unsigned l = 0; l < d2; l++) {
              double delta = double(qy[U(qi) * d2 + l]) - y[l];
              sy += delta * delta;
            }
            double dx = std::sqrt(sx), dy = std::sqrt(sy),
                   total =
                       double(alpha[qi]) * dx + (1. - double(alpha[qi])) * dy;
            if (total > 0) {
              sum += double(alpha[qi]) * dx / total;
              ++good;
            }
          }
          if (good) {
            contrib += sum / good;
            ++valid;
          }
        }
      }
      std::sort(lat.begin(), lat.end());
      double count = ids.size(), hits = 0, pre = 0, post = 0, rerank = 0,
             cp = 0, cs = 0, cw = 0, ex = 0;
      std::array<GStat, 8> total{};
      for (auto &a : answers) {
        hits += a.hits;
        pre += a.before;
        post += a.after;
        rerank += a.reranked;
        cp += a.coord_pages;
        cs += a.coord_submitted;
        cw += a.coord_wait;
        ex += a.exact_us;
        for (unsigned g = 0; g < 8; g++) {
          auto &s = total[g];
          auto &t = a.graph[g];
          s.expanded += t.expanded;
          s.pq += t.pq;
          s.exact += t.exact;
          s.pages += t.pages;
          s.submitted += t.submitted;
          s.inspected += t.inspected;
          s.rejected += t.rejected;
          s.wait += t.wait;
        }
      }
      struct rusage usage {};
      getrusage(RUSAGE_SELF, &usage);
      if (!first)
        out << ",";
      first = false;
      out << "{\"label\":\"" << job.label << "\",\"method\":\"" << job.method
          << "\",\"L\":" << job.L << ",\"kx\":" << job.kx
          << ",\"ky\":" << job.ky << ",\"medium\":\"" << e.mode
          << "\",\"navigation\":\"" << e.nav << "\",\"query_count\":" << count
          << ",\"threads\":" << threads << ",\"start_unix\":" << begin
          << ",\"seconds\":" << seconds << ",\"samples\":" << lat.size()
          << ",\"recall\":" << hits / (count * 10) << ",\"latency_us\":"
          << std::accumulate(lat.begin(), lat.end(), 0.) / lat.size()
          << ",\"qps\":" << lat.size() / seconds << ",\"p95_us\":"
          << lat[std::min(lat.size() - 1, U(std::ceil(.95 * lat.size()) - 1))]
          << ",\"candidates_before_union\":" << pre / count
          << ",\"candidates_after_union\":" << post / count
          << ",\"reranked\":" << rerank / count
          << ",\"coordinate_pages\":" << cp / count
          << ",\"coordinate_submitted_pages\":" << cs / count
          << ",\"coordinate_wait_us\":" << cw / count
          << ",\"exact_us\":" << ex / count
          << ",\"max_return_relative_error\":" << error
          << ",\"contribution_valid_queries\":" << valid
          << ",\"image_contribution\":";
      if (valid)
        out << contrib / valid;
      else
        out << "null";
      out << ",\"topology_bytes\":" << e.topology_bytes
          << ",\"sidecar_bytes\":" << e.side_bytes
          << ",\"pq_code_bytes\":" << e.pq_bytes
          << ",\"resident_page_bytes\":" << e.resident_bytes
          << ",\"peak_rss_kib\":" << usage.ru_maxrss << ",\"graphs\":[";
      bool gf = true;
      for (unsigned g = 0; g < 8; g++)
        if (e.graphs[g]) {
          if (!gf)
            out << ",";
          gf = false;
          auto &s = total[g];
          out << "{\"name\":\"" << e.graphs[g]->name
              << "\",\"expanded\":" << s.expanded / count
              << ",\"pq_space_evals\":" << s.pq / count
              << ",\"exact_space_evals\":" << s.exact / count
              << ",\"topology_pages\":" << s.pages / count
              << ",\"topology_submitted_pages\":" << s.submitted / count
              << ",\"topology_wait_us\":" << s.wait / count
              << ",\"inspected\":" << s.inspected / count
              << ",\"rejected\":" << s.rejected / count << "}";
        }
      out << "],\"queries\":[";
      for (U k = 0; k < answers.size(); k++) {
        auto &a = answers[k];
        if (k)
          out << ",";
        out << "{\"query_id\":" << ids[k] << ",\"hits\":" << a.hits
            << ",\"ids\":[";
        for (unsigned z = 0; z < 10; z++) {
          if (z)
            out << ",";
          out << a.ids[z];
        }
        out << "],\"distances\":[";
        for (unsigned z = 0; z < 10; z++) {
          if (z)
            out << ",";
          out << a.distances[z];
        }
        out << "]}";
      }
      out << "]}";
      out.flush();
      if (trace)
        traces(std::string(argv[13]) + "." + job.label, answers);
    }
    out << "]\n";
    if (!out)
      throw std::runtime_error("output write");
    return 0;
  }
  catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
