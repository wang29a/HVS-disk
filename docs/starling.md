# Starling：生成页面重排顺序

Starling 是外部图分区实现。本项目使用 [wang29a/starling](https://github.com/wang29a/starling) 的 custom topology 路径：读取 hvs-disk 的 metadata 和拓扑，生成 partition，再按页面重新排列 topology。它不负责构图、PQ 编码、alpha 压缩或向量生成。

本文保留历史 node3 验证配置。固定版本是 `7437c4848a83e7bf62558c5eab1eb999feed8537`，安装示例为 `/mnt/nvme3/wz/starling-wang29a`。其他机器需按实际编译器、Boost ABI 和依赖路径适配。

## 安装与编译

```bash
export STARLING_ROOT=/mnt/nvme3/wz/starling-wang29a

git clone --recurse-submodules \
  https://github.com/wang29a/starling.git "$STARLING_ROOT"
cd "$STARLING_ROOT"
git checkout 7437c4848a83e7bf62558c5eab1eb999feed8537
git submodule update --init --recursive
```

Node3 的旧 Boost 使用旧 libstdc++ ABI，因此历史验证配置带 `_GLIBCXX_USE_CXX11_ABI=0`：

```bash
cmake -S . -B release-gcc11-abi0 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc-11 \
  -DCMAKE_CXX_COMPILER=g++-11 \
  -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0 \
  -DTBB_ENABLE_IPO=OFF
cmake --build release-gcc11-abi0 -j8 --target partitioner
```

该机器的旧汇编器不支持 oneTBB 自动启用的 `tpause`。若新克隆出现此错误，历史兼容处理是从 `graph_partition/oneTBB/cmake/compilers/GNU.cmake` 移除 `-mwaitpkg`，并在 `graph_partition/oneTBB/include/oneapi/tbb/detail/_config.h` 将 `__TBB_WAITPKG_INTRINSICS_PRESENT` 设为 0，再重新构建。已有 node3 克隆做过这两处本地修改；它们不改变 partition 算法。新环境没有该错误时无需照搬。

Custom relayout 的历史编译命令为：

```bash
mkdir -p release-gcc11-abi0/tests/utils
g++-11 -std=c++14 -O2 -mavx2 -include immintrin.h \
  -Iinclude tests/utils/index_relayout.cpp \
  -Wl,--unresolved-symbols=ignore-all \
  -o release-gcc11-abi0/tests/utils/index_relayout_custom
```

这是 node3 旧 Boost/MKL 环境下的兼容办法，链接时忽略了同一文件中标准 DiskANN 分支的符号，因此这里只能调用已验证的 `--custom` 分支。默认 benchmark 脚本和标准 DiskANN relayout 不能代替它；移植时优先在外部仓库拆出独立 custom target，而不是推广这个链接选项。

## 对每种格式独立生成 partition

先完成[构建与搜索流程](workflow.md)中的 Full 构建、Merge/No-alpha 转换，并保留其中的 `HVS_ROOT`、`EXP_ROOT`、`STARLING_ROOT` 环境变量。然后运行：

```bash
export STARLING_BUILD="$STARLING_ROOT/release-gcc11-abi0"
export LD_LIBRARY_PATH="/opt/gcc-11.4/lib64:$STARLING_BUILD/gnu_11.4_cxx17_64_release:$LD_LIBRARY_PATH"
set -euo pipefail

for kind in full merge no-alpha; do
  META="$EXP_ROOT/$kind/_disk.index"
  TOPO="$EXP_ROOT/$kind/disk_index_graph"
  PART="$EXP_ROOT/$kind/partition.bin"

  ENTRYPOINT_COUNT=$(python3 - "$META" <<'PY'
import struct, sys
with open(sys.argv[1], 'rb') as stream:
    header = stream.read(32)
entrypoints = struct.unpack('<5IQI', header)[-1]
assert entrypoints > 0, 'metadata 中没有入口点'
print(entrypoints)
PY
  )

  "$STARLING_BUILD/graph_partition/partitioner" \
    --data_type float \
    --gp_file "$PART" \
    --custom_graph 1 \
    --graph_file "$TOPO" \
    --meta_file "$META" \
    --ep_size "$ENTRYPOINT_COUNT" \
    -T 8 -L 8

  "$STARLING_BUILD/tests/utils/index_relayout_custom" \
    --custom "$TOPO" "$META" "$PART" \
    "$EXP_ROOT/$kind/disk_index_graph.relayout"

  "$HVS_ROOT/build/tests/pad_partition" "$PART"
done
```

`-T 8` 是分区线程数，`-L 8` 是 LDG 轮数，和 ANN 搜索深度 L 无关。入口点数量从各自 metadata 读取，不能固定抄成 12。上述运行库路径用于 node3，保证加载 GCC 11 的 libstdc++；其他机器应使用自己的路径。

Full、Merge、No-alpha 会有不同的 nodes/page，所以各自生成 partition。不要先重排 Full，再把顺序直接当成另两种格式的原生 Starling 分区；统一顺序只适用于明确声明的受控实验。

## 输出如何交给搜索器

每种方案得到 `partition.bin`、`partition.bin.aligned` 和 `disk_index_graph.relayout`。保留转换阶段生成的原 8192 字节 `_disk.index`：该版本 custom relayout 还会输出一个只复制 4096 字节的 metadata 副本，不能安装它。

随后回到[组装搜索目录](workflow.md#组装搜索目录)，将原 metadata、该方案 relayout topology、aligned partition、共享坐标和 PQ 组合到一起。重排只能改变物理页顺序，应保持所有逻辑节点、邻居 ID、alpha slots 和入口语义；验收时核对 partition 双射、整条记录、空槽与 padding。
