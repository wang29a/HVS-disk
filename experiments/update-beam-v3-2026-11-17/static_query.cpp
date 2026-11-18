// External harness: original Engine code is included unchanged from its pinned
// file.
#define main archived_multigraph_main
#include HVS_ORIGINAL_QUERY_SOURCE
#undef main
#include "dual_update.h"
#include <atomic>
#include <thread>
using namespace hvs_update;
struct Sample {
  std::array<unsigned, 10> ids{};
  double us = 0;
  U expanded = 0, topo = 0, coord = 0;
};
int main(int argc, char **argv) {
  try {
    if (argc != 4)
      throw std::runtime_error("fixture GT-directory output-jsonl");
    std::string root = argv[1], gt_dir = argv[2];
    Engine original(root + "/manifest.txt", "ssd", "pq", { 7 });
    DualUpdate executor(root, Options{});
    unsigned n, d, n2, d2;
    auto x = matrix<float>(root + "/query-x.bin", n, d),
         y = matrix<float>(root + "/query-y.bin", n2, d2);
    if (n != n2 || d != executor.dim1() || d2 != executor.dim2())
      throw std::runtime_error("dimensions");
    std::vector<unsigned> cohort;
    std::ifstream f(root + "/report-ids.txt");
    unsigned id;
    while (f >> id)
      cohort.push_back(id);
    if (cohort.size() != 500)
      throw std::runtime_error("cohort");
    std::vector<std::unique_ptr<Workspace> > work;
    for (unsigned i = 0; i < 4; ++i)
      work.emplace_back(new Workspace());
    std::ofstream output(argv[3]);
    output << std::setprecision(17);
    for (float alpha : { .1f, .3f, .5f, .7f, .9f }) {
      std::ostringstream name;
      name << alpha;
      unsigned gn, gd;
      auto gt = matrix<unsigned>(gt_dir + "/gt-1800000-" + name.str() + ".bin",
                                 gn, gd);
      if (gn != n || gd != 10)
        throw std::runtime_error("GT");
      std::map<unsigned, std::vector<Sample> > reference;
      auto run = [&](bool old, unsigned L, int rep, bool emit) {
        std::vector<Sample> samples(cohort.size());
        std::atomic<unsigned> next{ 0 };
        std::vector<std::thread> threads;
        auto begin = Clock::now();
        for (unsigned t = 0; t < 4; ++t)
          threads.emplace_back([&, t] {
            for (;;) {
              unsigned k = next.fetch_add(1);
              if (k >= cohort.size())
                break;
              unsigned q = cohort[k];
              auto tick = Clock::now();
              auto &s = samples[k];
              if (old) {
                Job job{ "static", "DEG_off", L, L, 0, alpha };
                auto a = original.query(job, x.data() + U(q) * d,
                                        y.data() + U(q) * d2, alpha, *work[t],
                                        false);
                s.us = usec(tick);
                s.ids = a.ids;
                s.expanded = a.graph[7].expanded;
                s.topo = a.graph[7].submitted;
                s.coord = a.coord_submitted;
              } else {
                QueryWork w;
                auto a =
                    executor.search(x.data() + U(q) * d, y.data() + U(q) * d2,
                                    alpha, 10, L, L, w);
                s.us = usec(tick);
                if (a.size() != 10)
                  throw std::runtime_error("top10");
                for (unsigned j = 0; j < 10; ++j)
                  s.ids[j] = a[j].id;
                s.expanded = w.expanded;
                s.topo = w.io.read_pages[0];
                s.coord = w.io.read_pages[1];
              }
            }
          });
        for (auto &t : threads)
          t.join();
        double seconds = usec(begin) / 1e6;
        if (!emit)
          return;
        if (old)
          reference[L] = samples;
        else {
          if (!reference.count(L))
            throw std::runtime_error("missing original reference");
          for (unsigned q = 0; q < samples.size(); ++q)
            if (samples[q].ids != reference[L][q].ids ||
                samples[q].expanded != reference[L][q].expanded)
              throw std::runtime_error(
                  "original Engine per-query ID/expansion parity");
        }
        std::vector<double> lat;
        U hits = 0, expanded = 0, topo = 0, coord = 0;
        for (unsigned k = 0; k < samples.size(); ++k) {
          const auto &s = samples[k];
          lat.push_back(s.us);
          expanded += s.expanded;
          topo += s.topo;
          coord += s.coord;
          for (auto nb : s.ids)
            for (unsigned j = 0; j < 10; ++j)
              if (nb == gt[U(cohort[k]) * 10 + j]) {
                ++hits;
                break;
              }
        }
        std::sort(lat.begin(), lat.end());
        double count = samples.size();
        output << "{\"path\":\"" << (old ? "original" : "executor")
               << "\",\"alpha\":" << alpha << ",\"L\":" << L
               << ",\"rerank\":" << L << ",\"rep\":" << rep
               << ",\"queries\":" << samples.size()
               << ",\"threads\":4,\"mean_us\":"
               << std::accumulate(lat.begin(), lat.end(), 0.) / count
               << ",\"p95_us\":" << lat[size_t(.95 * (lat.size() - 1))]
               << ",\"wall_seconds\":" << seconds
               << ",\"qps\":" << count / seconds
               << ",\"recall\":" << hits / (count * 10)
               << ",\"expanded\":" << expanded / count
               << ",\"topology_pages\":" << topo / count
               << ",\"coordinate_pages\":" << coord / count << "}\n";
        output.flush();
      };
      run(true, 96, -1, false);
      run(false, 96, -1, false);
      for (unsigned rep = 0; rep < 3; ++rep)
        for (unsigned L : { 96u }) {
          run(rep % 2 == 0, L, rep, true);
          run(rep % 2 != 0, L, rep, true);
        }
    }
    std::cout
        << "STATIC_QUERY_PASS same_graph=1 weights=5 repetitions=3 threads=4"
        << std::endl;
  }
  catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
