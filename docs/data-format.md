# 数据格式

hvs-disk 通过两份向量描述同一组对象。构建和评估之前，先确认文件内部格式、对象顺序，以及 query、alpha、GT 的对应关系；文件名本身不能证明这些条件成立。

## 输入文件如何对应

| 文件 | 形状 | 含义 |
|---|---|---|
| Base X | `N×D1` | 每个对象的第一空间向量 |
| Base Y | `N×D2` | 相同对象、相同行号的第二空间向量 |
| Query X | `Q×D1` | 第一空间查询 |
| Query Y | `Q×D2` | 与 Query X 逐行对应 |
| Alpha | `Q×1`，固定为 float32 | 每个查询的融合权重，范围 `[0,1]` |
| GT | `Q×K_gt` 的邻居 ID | 对应以上 query 与 alpha，且 `K_gt ≥ 搜索的 K` |

两空间的维度可以不同，但同一空间的 base/query 维度必须一致。Base 的第 `i` 行共同定义对象 ID `i`；只对一个空间 shuffle、过滤或换序会破坏这个对应关系。构建命令对两份 base 使用同一种 `data_type`。

## Base、query 和 alpha 的字节布局

```text
偏移 0：uint32 行数
偏移 4：uint32 维度
偏移 8：按行连续存储的矩阵数据
```

Linux/x86 实验使用小端序。数据区没有行间 padding，也不在每行前重复存维度。文件大小必须等于 `8 + 行数 × 维度 × 元素字节数`。

`float` 使用 float32、每个元素 4 字节；`int8` 和 `uint8` 每个元素 1 字节。Alpha 始终是 float32，不能跟随 int8 base 改成整数。

下面的只读检查适用于 float32 输入。在仓库根目录运行，把路径换成实际文件：

```bash
python3 - /mnt/nvme3/wz/dataset/Openimage_disk/base_img_emb.fvecs <<'PY'
import os, struct, sys
path = sys.argv[1]
with open(path, 'rb') as stream:
    rows, dims = struct.unpack('<II', stream.read(8))
actual = os.path.getsize(path)
expected = 8 + rows * dims * 4
print('rows={}, dims={}, bytes={}, expected={}'.format(rows, dims, actual, expected))
assert actual == expected, '文件大小与 float32 矩阵头不匹配'
PY
```

文件大小匹配只是第一步，仍需核对两空间的 ID 顺序、数值类型，以及 query/GT 的来源。

## GT 的字节布局

```text
[查询数 uint32][K_gt uint32]
[Q × K_gt 个邻居 ID，uint32]
[可选：Q × K_gt 个距离，float32]
[可选：Q × K_gt 个 tag，uint32]
```

ID、distance、tag 是各自完整的矩阵块，不是每个邻居的字段交错排列。允许 header+IDs、header+IDs+distances，或再加 tags 三种长度。GT 中的对象 ID 必须落在 `[0,N)`，查询数与顺序必须和查询矩阵一致。

不同 alpha 对应的排序目标通常不同。固定 `alpha=0.5` 的 GT 不能拿来评估 `alpha=0.1`，mixed-alpha 的 GT 也不能仅修改文件名就充当固定权重 GT。近期预实验沿用已有配套 GT，并额外核验其距离目标，细节见[预实验说明](../experiments/README.md)。

## `.fvecs`、`.ivecs` 与 bin 的区别

经典 fvecs/ivecs 在**每一行**前保存维度，本项目的 bin 只在**整个文件**开头保存一次行数和维度。node3 的 OpenImages 数据虽然沿用 `.fvecs/.ivecs` 后缀，内部已经是项目 bin，不要再次转换。

只有确认输入是经典 vecs 格式后，才使用这些工具：

```bash
./build/tests/utils/fvecs_to_bin input.fvecs output.fbin
./build/tests/utils/bvecs_to_bin input.bvecs output.bin
./build/tests/utils/ivecs_to_bin input.ivecs output.bin
```

转换不会自动解决对象对应、查询权重或 GT 语义问题。格式确认完成后，继续[构建与搜索流程](workflow.md)。
