#!/usr/bin/env bash
# Linux/x86-64. No experiment data or pre-existing binaries are needed.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build="${1:-"$root/build-experiments"}"
if (( $# )); then shift; fi
mkdir -p -- "$build"
build="$(cd -- "$build" && pwd)"
jobs="${HVS_BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN)}"
odin_sha=7840ce96ac74a461b8efe2739b6aed5b14c15b9a
deg_sha=728c4128e74039ce212c8c22a42a5185abd68c23

pinned_checkout() (
    local folder="$1" remote="$2" revision="$3"
    mkdir -p -- "$folder"
    cd -- "$folder"
    if [[ ! -d .git ]]; then
        git init -q
        git remote add origin "$remote"
        git fetch --depth 1 origin "$revision"
        git checkout --detach FETCH_HEAD
    fi
    [[ "$(git rev-parse HEAD)" == "$revision" ]]
    [[ -z "$(git status --porcelain -uno)" ]]
)

odin="${HVS_ODIN_SOURCE_DIR:-"$build/upstream/odinann"}"
deg="${HVS_DEG_SOURCE_DIR:-"$build/upstream/deg"}"
pinned_checkout "$odin" https://github.com/thustorage/PipeANN.git "$odin_sha"
if [[ -z "${HVS_DEG_SOURCE_DIR:-}" || -d "$deg/.git" ]]; then
    pinned_checkout "$deg" https://github.com/Heisenberg-Yin/DEG.git "$deg_sha"
else
    # The CMake source generator checks the archived author file manifest.
    [[ -d "$deg/include" && -d "$deg/src" ]]
fi

# Read CBLAS/BLAS choices from the parent configuration, including overrides.
common=(-DCMAKE_BUILD_TYPE=Release -DUSE_AIO=ON)
ADDITIONAL_DEFINITIONS='-DREORDER_COMPUTE_PQ -DUSE_TOPO_DISK' \
    cmake -S "$root" -B "$build" "${common[@]}" "$@" \
    -DHVS_BUILD_EXPERIMENTS=OFF
cblas="$(sed -n 's/^HVS_CBLAS_INCLUDE_DIR:PATH=//p' "$build/CMakeCache.txt")"
[[ -n "$cblas" ]]
ADDITIONAL_DEFINITIONS='-DBG_IO_THREAD -DDELTA_PRUNING' \
    cmake -S "$odin" -B "$build/odinann" "${common[@]}" "$@" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_CXX_FLAGS="${CXXFLAGS:-} -I\"$cblas\" -fno-fast-math -ffp-contract=off"
cmake --build "$build/odinann" --parallel "$jobs" --target build_disk_index

ADDITIONAL_DEFINITIONS='-DREORDER_COMPUTE_PQ -DUSE_TOPO_DISK' \
    cmake -S "$root" -B "$build" "${common[@]}" "$@" \
    -DHVS_BUILD_EXPERIMENTS=ON \
    -DHVS_DEG_SOURCE_DIR="$deg" -DHVS_ODIN_SOURCE_DIR="$odin" \
    -DHVS_ODIN_LIBRARY="$build/odinann/src/libpipeann.a"
cmake --build "$build" --parallel "$jobs"
