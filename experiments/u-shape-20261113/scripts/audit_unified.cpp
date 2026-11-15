// Final retained-edge counterfactual audit. Not a reconstruction of transient
// candidate pools or an exact historical build-branch count.
#define main unused_pointer_driver
#include "pointer_search.cpp"
#undef main
#include "pruning_semantics.h"
#include <atomic>
#include <sstream>
namespace ps = pruning_semantics;
struct FloatGraph {
  struct Edge {
    unsigned id, start, count;
  };
  unsigned n, d1, d2;
  std::vector<U> offsets;
  std::vector<Edge> edges;
  std::vector<std::pair<float, float> > ranges;
  explicit FloatGraph(const char *path) {
    std::ifstream f(path, std::ios::binary);
    U nodes = get<U>(f);
    n = nodes;
    d1 = get<unsigned>(f);
    d2 = get<unsigned>(f);
    if (nodes != n || !n || !d1 || !d2)
      throw std::runtime_error("float graph shape");
    offsets.reserve(U(n) + 1);
    edges.reserve(U(n) * 34);
    ranges.reserve(U(n) * 45);
    for (unsigned i = 0; i < n; i++) {
      offsets.push_back(edges.size());
      unsigned degree = get<unsigned>(f);
      if (degree > 40)
        throw std::runtime_error("audit degree");
      for (unsigned k = 0; k < degree; k++) {
        unsigned id = get<unsigned>(f), count = get<unsigned>(f);
        if (id >= n || id == i || count > 64 || ranges.size() > UINT32_MAX)
          throw std::runtime_error("audit edge");
        edges.push_back({ id, unsigned(ranges.size()), count });
        for (unsigned j = 0; j < count; j++) {
          float lo = get<float>(f), hi = get<float>(f);
          if (lo < 0 || hi > 1 || lo > hi)
            throw std::runtime_error("audit range");
          ranges.emplace_back(lo, hi);
        }
      }
    }
    offsets.push_back(edges.size());
    if (f.peek() != std::char_traits<char>::eof())
      throw std::runtime_error("audit trailing graph bytes");
  }
};
struct Counts {
  U nodes = 0, edges = 0, unordered_pairs = 0, directions = 0,
    relevant_directions = 0;
  U first_zero_slope = 0, first_zero_rhs = 0, second_zero_slope = 0,
    second_zero_rhs = 0;
  U boundary_directions = 0, susceptible_edges = 0, susceptible_nodes = 0,
    endpoint0_edges = 0, endpoint1_edges = 0;
  U pair_distances = 0, pair_zero_x = 0, pair_zero_y = 0, pair_zero_both = 0;
  double potential_excess_width_union = 0, stored_width = 0;
  Counts &operator+=(const Counts &b) {
#define SUM(k) k += b.k
    SUM(nodes);
    SUM(edges);
    SUM(unordered_pairs);
    SUM(directions);
    SUM(relevant_directions);
    SUM(first_zero_slope);
    SUM(first_zero_rhs);
    SUM(second_zero_slope);
    SUM(second_zero_rhs);
    SUM(boundary_directions);
    SUM(susceptible_edges);
    SUM(susceptible_nodes);
    SUM(endpoint0_edges);
    SUM(endpoint1_edges);
    SUM(pair_distances);
    SUM(pair_zero_x);
    SUM(pair_zero_y);
    SUM(pair_zero_both);
    SUM(potential_excess_width_union);
    SUM(stored_width);
#undef SUM
    return *this;
  }
};
bool any_region(const FloatGraph &g, const FloatGraph::Edge &q,
                const FloatGraph::Edge &p, ps::Interval condition) {
  if (condition.empty())
    return false;
  for (unsigned i = 0; i < q.count; i++)
    for (unsigned j = 0; j < p.count; j++) {
      auto a = g.ranges[q.start + i], b = g.ranges[p.start + j];
      if (!ps::intersect(condition, ps::intersect({ a.first, a.second },
                                                  { b.first, b.second }))
               .empty())
        return true;
    }
  return false;
}
int main(int argc, char **argv) {
  try {
    if (argc != 6 && argc != 8)
      throw std::runtime_error(
          "float-graph baseX baseY output threads [first end]");
    FloatGraph g(argv[1]);
    unsigned n, d;
    auto x = matrix<float>(argv[2], n, d);
    if (n != g.n || d != g.d1)
      throw std::runtime_error("audit baseX");
    auto y = matrix<float>(argv[3], n, d);
    if (n != g.n || d != g.d2)
      throw std::runtime_error("audit baseY");
    unsigned threads = std::stoul(argv[5]);
    if (!threads || threads > 8)
      throw std::runtime_error("audit thread cap");
    omp_set_num_threads(threads);
    unsigned first = argc == 8 ? std::stoul(argv[6]) : 0,
             end = argc == 8 ? std::stoul(argv[7]) : n;
    if (first >= end || end > n)
      throw std::runtime_error("audit node bounds");
    std::unique_ptr<pipeann::Distance<float> > distance(
        pipeann::get_distance_function<float>(pipeann::Metric::L2));
    std::vector<Counts> totals(threads);
    std::vector<std::string> examples;
    std::atomic<U> done{ 0 };
    auto begin = Clock::now();
#pragma omp parallel for schedule(dynamic, 64)
    for (int64_t node = first; node < end; node++) {
      auto &c = totals[omp_get_thread_num()];
      U offset = g.offsets[node], nn = g.offsets[node + 1] - offset;
      std::vector<float> dx(nn), dy(nn);
      std::vector<std::vector<ps::Interval> > missing(nn);
      std::vector<bool> hit(nn, false), ep0(nn, false), ep1(nn, false);
      auto score = [&](unsigned a, unsigned b, bool second) {
        const auto &v = second ? y : x;
        unsigned dims = second ? g.d2 : g.d1;
        return std::sqrt(
            std::max(0.f, distance->compare(v.data() + U(a) * dims,
                                            v.data() + U(b) * dims, dims)));
      };
      for (unsigned k = 0; k < nn; k++) {
        dx[k] = score(node, g.edges[offset + k].id, false);
        dy[k] = score(node, g.edges[offset + k].id, true);
      }
      for (unsigned q = 0; q < nn; q++)
        for (unsigned p = 0; p < q; p++) {
          ++c.unordered_pairs;
          c.directions += 2;
          float slopes[2] = { dx[p] - dx[q] + dy[q] - dy[p],
                              dx[q] - dx[p] + dy[p] - dy[q] };
          float rhs[2] = { dy[q] - dy[p], dy[p] - dy[q] };
          bool relevant[2];
          for (unsigned z = 0; z < 2; z++) {
            c.first_zero_slope += slopes[z] == 0;
            c.first_zero_rhs += rhs[z] == 0;
            unsigned qi = z ? p : q, pi = z ? q : p;
            relevant[z] =
                any_region(g, g.edges[offset + qi], g.edges[offset + pi],
                           ps::solve(slopes[z], rhs[z]));
          }
          if (!relevant[0] && !relevant[1])
            continue;
          float pairx =
              score(g.edges[offset + p].id, g.edges[offset + q].id, false);
          float pairy =
              score(g.edges[offset + p].id, g.edges[offset + q].id, true);
          ++c.pair_distances;
          c.pair_zero_x += pairx == 0;
          c.pair_zero_y += pairy == 0;
          c.pair_zero_both += pairx == 0 && pairy == 0;
          for (unsigned z = 0; z < 2; z++) {
            if (!relevant[z])
              continue;
            unsigned qi = z ? p : q, pi = z ? q : p;
            ++c.relevant_directions;
            float a1 = slopes[z], b1 = rhs[z],
                  a2 = pairx - dx[qi] + dy[qi] - pairy, b2 = dy[qi] - pairy;
            c.second_zero_slope += a2 == 0;
            c.second_zero_rhs += b2 == 0;
            if (a1 != 0 && b1 != 0 && a2 != 0 && b2 != 0)
              continue;
            ++c.boundary_directions;
            auto corrected =
                ps::intersect(ps::solve(a1, b1), ps::solve(a2, b2));
            auto old = ps::intersect(ps::legacy(a1, b1), ps::legacy(a2, b2));
            const auto &qe = g.edges[offset + qi], &pe = g.edges[offset + pi];
            bool witness = false;
            for (unsigned i = 0; i < qe.count; i++)
              for (unsigned j = 0; j < pe.count; j++) {
                auto qr = g.ranges[qe.start + i], pr = g.ranges[pe.start + j];
                auto both = ps::intersect({ qr.first, qr.second },
                                          { pr.first, pr.second });
                auto now = ps::intersect(both, corrected),
                     before = ps::intersect(both, old);
                auto lost = ps::difference(now, before);
                for (auto interval : lost) {
                  // Width and endpoint tests are separate: interval
                  // subtraction's
                  // inclusive endpoints have zero measure and cannot inflate
                  // width.
                  if (interval.width() > 0) {
                    missing[qi].push_back(interval);
                    hit[qi] = true;
                    witness = true;
                  }
                }
                if (ps::active(now, 0) && !ps::active(before, 0)) {
                  ep0[qi] = true;
                  hit[qi] = true;
                  witness = true;
                }
                if (ps::active(now, 1) && !ps::active(before, 1)) {
                  ep1[qi] = true;
                  hit[qi] = true;
                  witness = true;
                }
              }
            if (witness) {
#pragma omp critical(audit_examples)
              if (examples.size() < 100) {
                std::ostringstream s;
                s << std::setprecision(12) << "{\"center\":" << node
                  << ",\"candidate\":" << qe.id << ",\"witness\":" << pe.id
                  << ",\"a1\":" << a1 << ",\"b1\":" << b1 << ",\"a2\":" << a2
                  << ",\"b2\":" << b2 << ",\"center_q_x\":" << dx[qi]
                  << ",\"center_q_y\":" << dy[qi]
                  << ",\"center_p_x\":" << dx[pi]
                  << ",\"center_p_y\":" << dy[pi] << ",\"pair_x\":" << pairx
                  << ",\"pair_y\":" << pairy << "}";
                examples.push_back(s.str());
              }
            }
          }
        }
      ++c.nodes;
      c.edges += nn;
      bool affected = false;
      for (unsigned q = 0; q < nn; q++) {
        const auto &qe = g.edges[offset + q];
        std::vector<ps::Interval> stored;
        for (unsigned z = 0; z < qe.count; z++) {
          auto p = g.ranges[qe.start + z];
          stored.push_back({ p.first, p.second });
        }
        c.stored_width += ps::union_width(stored);
        if (hit[q]) {
          ++c.susceptible_edges;
          affected = true;
          c.potential_excess_width_union += ps::union_width(missing[q]);
        }
        c.endpoint0_edges += ep0[q];
        c.endpoint1_edges += ep1[q];
      }
      c.susceptible_nodes += affected;
      U processed = ++done;
      if (processed % 10000 == 0) {
#pragma omp critical(audit_progress)
        {
          std::ofstream out(std::string(argv[4]) + ".progress.json");
          out << "{\"stage\":\"auditing\",\"processed_nodes\":" << done.load()
              << ",\"total_nodes\":" << (end - first) << ",\"elapsed_seconds\":"
              << std::chrono::duration<double>(Clock::now() - begin).count()
              << "}\n";
        }
      }
    }
    Counts c;
    for (auto t : totals)
      c += t;
    std::ofstream out(argv[4]);
    out << std::setprecision(16)
        << "{\"stage\":\"complete\",\"scope\":\"both orientations of final "
           "retained neighbor pairs, active-range overlap; not historical "
           "candidate comparisons\",\"first_node\":" << first
        << ",\"end_node\":" << end;
#define FIELD(k) out << ",\"" #k "\":" << c.k
    FIELD(nodes);
    FIELD(edges);
    FIELD(unordered_pairs);
    FIELD(directions);
    FIELD(relevant_directions);
    FIELD(first_zero_slope);
    FIELD(first_zero_rhs);
    FIELD(second_zero_slope);
    FIELD(second_zero_rhs);
    FIELD(boundary_directions);
    FIELD(susceptible_edges);
    FIELD(susceptible_nodes);
    FIELD(endpoint0_edges);
    FIELD(endpoint1_edges);
    FIELD(pair_distances);
    FIELD(pair_zero_x);
    FIELD(pair_zero_y);
    FIELD(pair_zero_both);
    FIELD(potential_excess_width_union);
    FIELD(stored_width);
#undef FIELD
    out << ",\"elapsed_seconds\":"
        << std::chrono::duration<double>(Clock::now() - begin).count()
        << ",\"examples\":[";
    for (U i = 0; i < examples.size(); i++) {
      if (i)
        out << ",";
      out << examples[i];
    }
    out << "]}\n";
    if (!out)
      throw std::runtime_error("audit output");
  }
  catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
