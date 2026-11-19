"""Generate compile-time variants without importing node3 orchestration."""
import argparse
import ast
import hashlib
import json
import re
import runpy
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace


def load_functions(path, names, scope, constants=()):
    parsed = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [node for node in parsed.body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise ValueError(f"Missing transform in {path}")
    for node in parsed.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in constants:
                    scope[target.id] = ast.literal_eval(node.value)
    parsed.body = selected
    exec(compile(parsed, str(path), "exec"), scope)
    return scope


def once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"Build-source hook changed: {before}")
    return text.replace(before, after, 1)


def emit(out, name, text):
    (out / name).write_text(text, encoding="utf-8")


def generate(repo, out, author):
    ex = repo / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    metric = ex / "metric-consistent-20261111"
    scratch = out / "metric"
    for folder in ("sources", "analysis"):
        (scratch / folder).mkdir(parents=True, exist_ok=True)
    for src in (metric / "sources").glob("*.cpp"):
        shutil.copyfile(src, scratch / "sources" / src.name)
    metric_main = runpy.run_path(str(metric / "scripts/prepare_variant.py"))["main"]
    metric_main.__globals__["ROOT"] = scratch
    metric_main()

    u = ex / "u-shape-20261113/scripts"
    transforms = load_functions(u / "prepare_specialized.py", ["specialize"], {"re": re})
    load_functions(u / "prepare_pruning_v2.py", ["corrected_intervals", "direct_fixed"], transforms)
    shared = (scratch / "variant/hybrid_index.cpp").read_text()
    unified = transforms["corrected_intervals"](shared)
    emit(out, "unified.cpp", unified)
    emit(out, "fixed.cpp", transforms["direct_fixed"](unified))
    emit(out, "fixed-v1.cpp", transforms["specialize"](shared))

    builder = (ex / "laion-six-cell-20261111/scripts/build_graph.cpp").read_text()
    anchor = 'throw std::runtime_error("graph target");'
    build_v2 = 'extern "C" void hvs_write_pruning_stats(const char *);\n' + once(
        builder, anchor, anchor + '\n    hvs_write_pruning_stats((std::string(argv[8])+"/pruning-counters.json").c_str());')
    emit(out, "build-v2.cpp", build_v2)

    ab = ex / "backlink-ab-20261115/scripts"
    ab_text = load_functions(ab / "compile_ab.py", ["variant"], {}, ["B_BODY"])["variant"](unified)[0]
    emit(out, "ab.cpp", ab_text)
    ab_builder = 'extern "C" void hvs_ab_write_stats(const char *);\n' + once(
        build_v2, '    hvs_write_pruning_stats((std::string(argv[8])+"/pruning-counters.json").c_str());',
        '    hvs_write_pruning_stats((std::string(argv[8])+"/pruning-counters.json").c_str());\n'
        '    hvs_ab_write_stats((std::string(argv[8])+"/reverse-insert-stats.json").c_str());')
    emit(out, "build-ab.cpp", ab_builder)
    hook = "prune_neighbors_no_opt(candidates,parameter,selected);"
    emit(out, "capture.cpp", '#include "trace_capture.h"\n' + once(ab_text, hook, '''auto function_input=candidates;
    prune_neighbors_no_opt(candidates,parameter,selected);
    hvs_capture::record(des.id_,n,function_input,selected,[&](unsigned a,unsigned b,unsigned space) {
      auto *data=space?_data2:_data1;auto dim=space?_aligned_dim2:_aligned_dim1;
      return _distance->compare(data+size_t(a)*dim,data+size_t(b)*dim,unsigned(dim));
    });'''))

    edbt = ex / "edbt-evidence-20261115/scripts"
    for variant, counting in (("plain", False), ("counted", True)):
        fn = load_functions(edbt / "compile_native.py", ["transform"], {"re": re})["transform"]
        emit(out, f"native-{variant}.cpp",
             fn((repo / "src/search/beam_search.cpp").read_text(), counting))
    relaxed = load_functions(edbt / "compile_relaxed.py", ["transform"], {})["transform"]
    emit(out, "relaxed.cpp", relaxed(unified))
    member = (u / "test_actual_pruner.cpp").read_text()
    member = once(member, '#include "index.h"', '#include "index.h"\n#include "relaxation.h"')
    member = once(member, "dist(p, q) <= dist(0, q)", "hvs_unified_rho() * dist(p, q) <= dist(0, q)")
    member = once(member, "dist(got[j].id_, got[i].id_) <= dist(0, got[i].id_)",
                  "hvs_unified_rho() * dist(got[j].id_, got[i].id_) <= dist(0, got[i].id_)")
    emit(out, "test-relaxed-member.cpp", member)
    multi = (ex / "multigraph-20261113/scripts/multigraph.cpp").read_text()
    emit(out, "multigraph32.cpp", once(multi, "threads > 8", "threads > 32"))

    v4 = ex / "update-merge-v4"
    prof = out / "merge-profile"
    if prof.is_dir():
        shutil.rmtree(prof)
    load_functions(v4 / "merge_profile.py", ["replace", "prepare"],
                   {"ROOT": v4, "shutil": shutil})["prepare"](prof)

    if author:
        manifest = json.loads((ab.parent / "AUTHOR-UPSTREAM.json").read_text())
        for name, digest in manifest["files"].items():
            if hashlib.sha256((author / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Unexpected pinned DEG source: {name}")
        original = out / "author-original"
        for name in ("src", "include"):
            dest = original / name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(author / name, dest)
        c = SimpleNamespace(RUN=out, sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest(),
                            save=lambda p, d: p.write_text(json.dumps(d, indent=2) + "\n"))
        adapt = load_functions(ab / "function_compile.py", ["author_variant"],
                               {"c": c, "F": out, "shutil": shutil}, ["ZERO", "FIXES"])["author_variant"]
        for mask in range(8):
            adapt(mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--author", type=Path)
    args = parser.parse_args()
    generate(args.source.resolve(), args.output.resolve(), args.author)
