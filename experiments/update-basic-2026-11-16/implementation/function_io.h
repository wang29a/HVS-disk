#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
namespace fnio {
template <class T> void put(std::ostream &s, T v) {
  s.write(reinterpret_cast<const char *>(&v), sizeof(v));
}
template <class T> T get(std::istream &s) {
  T v{};
  if (!s.read(reinterpret_cast<char *>(&v), sizeof(v)))
    throw std::runtime_error("truncated function record");
  return v;
}
struct Candidate {
  unsigned id;
  float e, g;
};
struct Edge {
  unsigned id;
  float e, g;
  unsigned layer;
  std::vector<std::pair<float, float> > ranges;
};
struct Record {
  unsigned target, new_id;
  uint64_t ordinal;
  std::vector<Candidate> candidates;
  std::vector<float> table;
  std::vector<Edge> expected;
};
inline void edge(std::ostream &s, const Edge &e) {
  put(s, e.id);
  put(s, e.e);
  put(s, e.g);
  put(s, e.layer);
  put(s, unsigned(e.ranges.size()));
  for (auto r : e.ranges) {
    put(s, r.first);
    put(s, r.second);
  }
}
inline Edge edge(std::istream &s) {
  Edge e;
  e.id = get<unsigned>(s);
  e.e = get<float>(s);
  e.g = get<float>(s);
  e.layer = get<unsigned>(s);
  auto n = get<unsigned>(s);
  if (n > 1000)
    throw std::runtime_error("range count");
  for (unsigned i = 0; i < n; ++i) {
    float a = get<float>(s), b = get<float>(s);
    e.ranges.emplace_back(a, b);
  }
  return e;
}
inline void edges(std::ostream &s, const std::vector<Edge> &v) {
  put(s, unsigned(v.size()));
  for (const auto &e : v)
    edge(s, e);
}
inline std::vector<Edge> edges(std::istream &s) {
  auto n = get<unsigned>(s);
  if (n > 1000)
    throw std::runtime_error("degree");
  std::vector<Edge> v;
  for (unsigned i = 0; i < n; ++i)
    v.push_back(edge(s));
  return v;
}
inline bool read(std::istream &s, Record &r) {
  if (s.peek() == EOF)
    return false;
  r = {};
  r.ordinal = get<uint64_t>(s);
  r.target = get<unsigned>(s);
  r.new_id = get<unsigned>(s);
  auto m = get<unsigned>(s);
  if (m > 1000 || !m)
    throw std::runtime_error("pool size");
  for (unsigned i = 0; i < m; ++i) {
    unsigned id = get<unsigned>(s);
    float e = get<float>(s), g = get<float>(s);
    r.candidates.push_back({ id, e, g });
  }
  r.table.resize(size_t(m) * m * 2);
  if (!s.read(reinterpret_cast<char *>(r.table.data()), r.table.size() * 4))
    throw std::runtime_error("distance oracle");
  r.expected = edges(s);
  return true;
}
inline void header(std::istream &s, const char *magic) {
  char h[8];
  if (!s.read(h, 8) || std::memcmp(h, magic, 8))
    throw std::runtime_error("record magic");
}
inline std::vector<float> matrix(const char *path, unsigned &n, unsigned &d) {
  std::ifstream f(path, std::ios::binary);
  n = get<unsigned>(f);
  d = get<unsigned>(f);
  std::vector<float> a(size_t(n) * d);
  if (!f.read(reinterpret_cast<char *>(a.data()), a.size() * 4) ||
      f.peek() != EOF)
    throw std::runtime_error("base matrix");
  return a;
}
struct Oracle {
  const Record *record = nullptr;
  const float *x = nullptr, *y = nullptr;
  unsigned n = 0, d1 = 0, d2 = 0;
  std::unordered_map<unsigned, unsigned> local;
  void select(const Record &r) {
    record = &r;
    local.clear();
    for (unsigned i = 0; i < r.candidates.size(); ++i)
      local.emplace(r.candidates[i].id, i);
  }
  float raw(const float *a, const float *b, unsigned dim) const {
    auto pos = [](const float *p, const float *base, unsigned n, unsigned d) {
      auto v = reinterpret_cast<uintptr_t>(p),
           lo = reinterpret_cast<uintptr_t>(base);
      return v >= lo && v < lo + size_t(n) * d * 4;
    };
    unsigned space;
    const float *base;
    unsigned d;
    if (pos(a, x, n, d1) && pos(b, x, n, d1)) {
      space = 0;
      base = x;
      d = d1;
    } else if (pos(a, y, n, d2) && pos(b, y, n, d2)) {
      space = 1;
      base = y;
      d = d2;
    } else
      throw std::runtime_error("oracle pointer space");
    if (dim != d || (a - base) % d || (b - base) % d)
      throw std::runtime_error("oracle stride");
    unsigned i = local.at(unsigned((a - base) / d)),
             j = local.at(unsigned((b - base) / d)),
             m = record->candidates.size();
    return record->table[(size_t(i) * m + j) * 2 + space];
  }
};
}
