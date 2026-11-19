#include "dual_update.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

using namespace hvs_update;

static void check(bool valid, const char *message) {
  if (!valid)
    throw std::runtime_error(message);
}

static bool reachable(DualUpdate &index,
                      const std::vector<std::vector<uint32_t> > &graph,
                      uint32_t target, bool skip_deleted) {
  auto queue = index.entries();
  std::vector<bool> visited(graph.size());
  for (size_t i = 0; i < queue.size(); ++i) {
    const auto id = queue[i];
    if (visited.at(id) || (skip_deleted && index.deleted(id)))
      continue;
    if (id == target)
      return true;
    visited[id] = true;
    queue.insert(queue.end(), graph[id].begin(), graph[id].end());
  }
  return false;
}

int main(int argc, char **argv) {
  try {
    check(argc == 4, "usage: merge_check DIRECTORY DISTANCE SCENARIO");
    const std::string scenario = argv[3];
    Options options;
    options.exact_reprune = std::string(argv[2]) == "exact";
    DualUpdate index(argv[1], options);
    IOStats io;
    const auto before = index.audit_topology(io);
    auto initial_degree = index.indegree_snapshot();
    const bool retain = scenario == "retain";
    check(initial_degree[50] == (retain ? 2u : 1u) &&
              before[49] == std::vector<uint32_t>{ 50 },
          "unexpected in-edges into 50");
    check(index.erase(49), "cannot delete test vertex");
    check(reachable(index, before, 50, false) &&
              reachable(index, before, 50, true) == retain,
          "unexpected reachability without deleted vertex 49");
#ifdef HVS_MERGE_V4
    index.consolidate(io);
    const auto after = index.audit_topology(io);
    check(index.consolidated(49) && after[49].empty(),
          "deleted row was not cleared");
    check(reachable(index, after, 50, true),
          "repair did not connect vertex 50");
    check(io.merge_transferred_edges == (scenario == "search" ? 0u : 1u) &&
              io.merge_search_edges == (scenario == "search" ? 1u : 0u),
          "wrong repair path");
    if (scenario != "search")
      check(std::find(after[0].begin(), after[0].end(), 50) != after[0].end(),
            "original surviving predecessor did not receive the edge");
    if (retain)
      check(io.write_pages[0] == 2,
            "retaining an edge unexpectedly staged a duplicate write");
    std::vector<uint32_t> scanned_degree(after.size());
    uint64_t scanned_edges = 0;
    for (uint32_t id = 0; id < after.size(); ++id) {
      auto row = after[id];
      check(row.size() <= options.degree, "degree overflow");
      scanned_edges += row.size();
      std::sort(row.begin(), row.end());
      check(std::adjacent_find(row.begin(), row.end()) == row.end(),
            "duplicate edge");
      for (auto target : row) {
        check(target < after.size() && target != id && !index.deleted(target),
              "invalid edge");
        ++scanned_degree[target];
      }
    }
    check(scanned_degree == index.indegree_snapshot(),
          "in-degree audit mismatch");
    for (uint32_t id = 0; id < after.size(); ++id)
      check(index.deleted(id) || !initial_degree[id] || scanned_degree[id],
            "new zero in-degree");
    check(io.merge_live_nodes == 63 && io.merge_live_edges == scanned_edges,
          "merge edge telemetry mismatch");
    std::cout << "MERGE_DIRECTED_PASS " << scenario << ' ' << argv[2] << '\n';
#else
    try {
      index.consolidate(io);
    }
    catch (const std::runtime_error &error) {
      check(std::string(error.what()).find("cannot preserve incoming edges") !=
                std::string::npos,
            "unexpected failure from the frozen executor");
      check(index.neighbors(49, io) == before[49],
            "old plan cleared the sole row");
      std::cout << "OLD_MERGE_REPRODUCED " << scenario << ' ' << argv[2]
                << '\n';
      return 0;
    }
    throw std::runtime_error("old executor did not reproduce the merge stop");
#endif
  }
  catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
