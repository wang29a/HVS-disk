"""Derive the isolated degree-cap builder from the verified Link sources."""

from pathlib import Path
import shutil
import subprocess

from build import read, sha
from instrument import change
from run import save


def patch_builder(source):
    source = '#include "shard_degree.h"\n' + source
    source = change(source, '  const auto range = parameter.Get<unsigned>("R");',
                    '  const auto row_default = parameter.Get<unsigned>("R");')
    source = change(source, '    auto &des_pool = _final_graph[des.id_];',
                    '    const unsigned range = shard_degree::cap(des.id_, row_default);\n'
                    '    auto &des_pool = _final_graph[des.id_];')
    source = change(source, '      prune_neighbors_no_opt(dummy_pool, parameter, new_out_neighbors);',
                    '      const shard_degree::ScopedPruneLimit limit(range);\n'
                    '      prune_neighbors_no_opt(dummy_pool, parameter, new_out_neighbors);')
    source = change(source, '    prune_neighbors_no_opt(pool, parameters, pruned_list);',
                    '    {\n'
                    '      const shard_degree::ScopedPruneLimit limit(shard_degree::cap(node, range));\n'
                    '      prune_neighbors_no_opt(pool, parameters, pruned_list);\n'
                    '    }')
    source = change(source, '    if (_final_graph[node].size() > range) {',
                    '    const unsigned cap = shard_degree::cap(node, range);\n'
                    '    if (_final_graph[node].size() > cap) {')
    source = change(source, '      prune_neighbors_no_opt(dummy_pool, parameters, new_out_neighbors);',
                    '      const shard_degree::ScopedPruneLimit limit(cap);\n'
                    '      prune_neighbors_no_opt(dummy_pool, parameters, new_out_neighbors);')
    signature = "void HybridIndex<T, TagT>::prune_neighbors_no_opt("
    prefix, body = source.split(signature)
    end = body.index("HybridIndex<T, TagT>::prune_neighbors(")
    source = change(body[:end], '''  const unsigned range = parameter.Get<unsigned>("R");
  auto judge_dominate''', '''  const unsigned range = shard_degree::prune_limit
                             ? shard_degree::prune_limit
                             : parameter.Get<unsigned>("R");
  auto judge_dominate''')
    return prefix + signature + source + body[end:]


def patch_driver(source):
    source = '#include "shard_degree.h"\n' + source
    source = change(source, '''#ifdef HVS_LINK_PROFILE
unsigned profile_L = 200, profile_C = 750;
#endif''', '''unsigned shard_L = 200, shard_C = 750;''')
    source = change(source, '''#ifdef HVS_LINK_PROFILE
  p.Set<unsigned>("L",profile_L);
  p.Set<unsigned>("C",profile_C);
#else
  p.Set<unsigned>("L",200);
  p.Set<unsigned>("C",750);
#endif''', '''  p.Set<unsigned>("L",shard_L);
  p.Set<unsigned>("C",shard_C);''')
    source = change(source, '''void build_shard(const char *x,const char *y,Id n,unsigned r,const std::string &out) {
  pipeann::Parameters p;''', '''void build_shard(const char *x, const char *y, Id n, unsigned r, const std::string &out,
                 const char *caps_path) {
  std::vector<uint8_t> caps;
  if (caps_path) {
    std::ifstream file(caps_path, std::ios::binary);
    caps.resize(n);
    check(bool(file.read(reinterpret_cast<char *>(caps.data()), n)) && file.peek() == EOF,
          "degree-cap file length");
    const auto invalid = std::find_if(caps.begin(), caps.end(), [r](uint8_t cap) {
      return cap != r && !(r == 40 && cap == 27);
    });
    check(invalid == caps.end(), "unexpected degree cap");
    shard_degree::caps = &caps;
  }
  pipeann::Parameters p;''')
    source = change(source, 'index._final_graph[id].size()<=r',
                    'index._final_graph[id].size() <= shard_degree::cap(id, r)')
    source = change(source, '''  check(bool(f),"shard graph output");''', '''  check(bool(f),"shard graph output");
  shard_degree::caps = nullptr;''')
    source = change(source, '''#ifdef HVS_LINK_PROFILE
    if (argc==9 && mode=="shard") {
      profile_L = std::stoul(argv[7]);
      profile_C = std::stoul(argv[8]);
      argc = 7;
    }
#endif
    if (argc==7 && mode=="shard")
      build_shard(argv[2],argv[3],std::stoul(argv[4]),std::stoul(argv[5]),argv[6]);''',
                    '''    if ((argc == 7 || argc == 9 || argc == 10) && mode == "shard") {
      if (argc >= 9) {
        shard_L = std::stoul(argv[7]);
        shard_C = std::stoul(argv[8]);
      }
      build_shard(argv[2], argv[3], std::stoul(argv[4]), std::stoul(argv[5]), argv[6],
                  argc == 10 ? argv[9] : nullptr);
    }''')
    return source


def compile_builder(root, original):
    proof = read(original / "build/identity.json")
    off = proof["binaries"]["off"]
    previous = original / "build"
    inputs = {}
    for name, key in (("hybrid_index.cpp", "generated_builder_sha256"),
                      ("driver.cpp", "generated_driver_sha256")):
        path = previous / name
        inputs[name] = sha(path)
        if inputs[name] != proof[key]:
            raise RuntimeError("frozen generated source changed: " + name)
    build = root / "build"
    build.mkdir(exist_ok=True)
    builder = build / "hybrid_index.cpp"
    driver = build / "degree-driver.cpp"
    builder.write_text(patch_builder((previous / "hybrid_index.cpp").read_text(encoding="utf-8")), encoding="utf-8")
    driver.write_text(patch_driver((previous / "driver.cpp").read_text(encoding="utf-8")), encoding="utf-8")

    archive = build / "libpipeann-unified.a"
    source_archive = next(Path(x) for x in off["link"] if x.endswith(".a"))
    inputs["archive"] = sha(source_archive)
    obj = build / "hybrid_index.cpp.o"
    binary = build / "build-merge"
    include = "-I" + str(root / "source")
    cc = [*off["compile"][:-4], include, "-c", str(builder), "-o", str(obj)]
    substitutions = {str(previous / "driver.cpp"): str(driver),
                     str(source_archive): str(archive), off["binary"]: str(binary)}
    link = [off["link"][0], include, *(substitutions.get(x, x) for x in off["link"][1:])]
    with (root / "logs/compile-degree.log").open("w") as out:
        def execute(argv):
            subprocess.run(argv, check=True, stdout=out, stderr=subprocess.STDOUT)
        execute(cc)
        shutil.copyfile(source_archive, archive)
        execute(["ar", "r", str(archive), str(obj)])
        execute(link)
    save(build / "degree-identity.json",
         dict(inputs=inputs, binary=str(binary), sha256=sha(binary), compile=cc, link=link,
              cap_header_sha256=sha(root / "source/shard_degree.h")))
    return binary
