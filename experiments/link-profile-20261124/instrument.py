"""Generate compile-time instrumentation from the frozen unified builder."""

import hashlib
from pathlib import Path


SOURCE_SHA = "fe6be5622a486006ec9bebcd4f027619c3cbb7a30b16af9cb83da6565292bf08"


def change(text, before, after):
    if text.count(before) != 1:
        raise ValueError("Frozen source anchor changed: "+before[:90])
    return text.replace(before, after)


def guarded(text):
    return "#ifdef HVS_LINK_PROFILE\n"+text+"\n#endif\n"


def before(text, anchor, inserted):
    return change(text, anchor, guarded(inserted)+anchor)


def after(text, anchor, inserted):
    return change(text, anchor, anchor+guarded(inserted))


def builder(path):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA:
        raise ValueError("Frozen unified builder hash changed")
    s=raw.decode("utf-8")
    s=guarded('#include "link_profile.h"')+s
    s=after(s, "        pool[k].flag = false;\n        unsigned n = pool[k].id;\n",
            "        if (auto *w = hvs_link_profile::worker) ++w->current_expanded;")
    s=after(s, "    if (prune_needed) {\n",
            "      if (auto *w = hvs_link_profile::worker) ++w->current_overflows;")
    s=after(s, "  pipeann::Timer link_timer;\n",
            "  namespace hp = hvs_link_profile;\n  hp::Session profile(omp_get_max_threads());")
    s=before(s, "#pragma omp parallel for schedule(dynamic)\n  for (int64_t node = 1; node < n_vecs_to_visit; node++)",
             "  profile.insertion_begin = hp::now();")
    s=after(s, "  for (int64_t node = 1; node < n_vecs_to_visit; node++) {\n",
            """    auto &profile_worker = profile.workers[omp_get_thread_num()];
    hp::worker = &profile_worker;
    if (profile_worker.inserted++ == 0) ++profile_worker.initialized;
    profile_worker.current_expanded = profile_worker.current_overflows = profile_worker.current_layers = 0;
    auto profile_start = hp::now();""")
    s=after(s, "    get_expanded_nodes(node, L, local_init_ids, pool, visited, ids_rw_mutex);\n",
            "    auto profile_search_end = hp::now();")
    s=before(s, "    prune_neighbors_no_opt(pool, parameters, pruned_list);\n",
             """    auto profile_candidates = pool.size();
    profile_worker.read_new_layers = true;
    auto profile_prune_start = hp::now();""")
    s=after(s, "    prune_neighbors_no_opt(pool, parameters, pruned_list);\n",
            """    auto profile_prune_end = hp::now();
    profile_worker.read_new_layers = false;""")
    s=before(s, "    inter_insert(node, pruned_list, parameters);\n",
             "    auto profile_reverse_start = hp::now();")
    s=after(s, "    inter_insert(node, pruned_list, parameters);\n",
            "    auto profile_reverse_end = hp::now();")
    s=after(s, "    UpdateEnterpointSet(enterpoints_skyline, _init_ids, emb_center.get(),\n                        loc_center.get(), node, ids_rw_mutex);\n",
            """    auto profile_end = hp::now();
    const auto search_ns = hp::elapsed(profile_start, profile_search_end);
    const auto prune_ns = hp::elapsed(profile_prune_start, profile_prune_end);
    const auto reverse_ns = hp::elapsed(profile_reverse_start, profile_reverse_end);
    profile_worker.time[hp::search] += search_ns;
    profile_worker.time[hp::prune] += prune_ns;
    profile_worker.time[hp::reverse] += reverse_ns;
    profile_worker.time[hp::bookkeeping] += hp::elapsed(profile_start, profile_end) -
                                           search_ns - prune_ns - reverse_ns;
    profile_worker.add(hp::visited, visited.size());
    profile_worker.add(hp::expanded, profile_worker.current_expanded);
    profile_worker.add(hp::candidates, profile_candidates);
    profile_worker.add(hp::skyline, profile_worker.current_layers);
    profile_worker.add(hp::retained, pruned_list.size());
    profile_worker.add(hp::overflow, profile_worker.current_overflows);
    profile_worker.reverse_attempts += pruned_list.size();
    hp::worker = nullptr;""")
    s=before(s, "  size_t size = n_vecs_to_visit;\n  std::cerr << \"K:\"",
             "  profile.insertion_end = hp::now();")
    start=s.index('#pragma omp parallel for schedule(dynamic, 65536)')
    end=s.index('  LOG(INFO) << "init ids size: "',start)
    block=s[start:end]
    block=guarded("  profile.cleanup_begin = hp::now();")+block
    block=after(block,"  for (_s64 node_ctr = 0; node_ctr < n_vecs_to_visit; node_ctr++) {\n",
                """    auto &profile_worker = profile.workers[omp_get_thread_num()];
    auto profile_cleanup_start = hp::now();""")
    block=after(block,"    if (_final_graph[node].size() > range) {\n",
                "      ++profile_worker.cleaned_rows;")
    pos=block.rindex("  }\n")
    block=block[:pos]+guarded("    profile_worker.time[hp::cleanup] += hp::elapsed(profile_cleanup_start, hp::now());")+block[pos:]
    block+=guarded("  profile.cleanup_end = hp::now();")
    s=s[:start]+block+s[end:]
    s=after(s, '              << ((double)link_timer.elapsed() / (double)1000000) << "s";\n  }\n',
            "  profile.write(n_vecs_to_visit, L, range, link_timer.elapsed());")
    start=s.index("void HybridIndex<T, TagT>::prune_neighbors_no_opt(")
    end=s.index("template <",start)
    core=before(s[start:end],"  pool.swap(result);\n  size_t iter = 0;\n",
                "  if (auto *w = hvs_link_profile::worker; w && w->read_new_layers) w->current_layers = l;")
    s=s[:start]+core+s[end:]
    return s


def driver(path):
    s=path.read_text()
    s=change(s,'void set_params(pipeann::Parameters &p, unsigned r) {',
              """#ifdef HVS_LINK_PROFILE
unsigned profile_L = 200, profile_C = 750;
#endif
void set_params(pipeann::Parameters &p, unsigned r) {""")
    s=change(s,'  p.Set<unsigned>("L",200);\n  p.Set<unsigned>("C",750);',
              """#ifdef HVS_LINK_PROFILE
  p.Set<unsigned>("L",profile_L);
  p.Set<unsigned>("C",profile_C);
#else
  p.Set<unsigned>("L",200);
  p.Set<unsigned>("C",750);
#endif""")
    s=before(s,'    if (argc==7 && mode=="shard")',
             """    if (argc==9 && mode=="shard") {
      profile_L = std::stoul(argv[7]);
      profile_C = std::stoul(argv[8]);
      argc = 7;
    }""")
    return s
