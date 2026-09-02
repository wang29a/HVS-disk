#!/bin/bash

# =================================================================
# 默认参数设置 (Default Configuration)
# 你可以在这里修改默认值，或者通过命令行参数覆盖它们
# =================================================================
INDEX_PATH="/home/gongwei/deg/Index/CC3M/"
QUERY_IMG="/home/gongwei/deg/dataset/CC3M_disk/query_img_emb.fvecs"
QUERY_TEXT="/home/gongwei/deg/dataset/CC3M_disk/query_text_emb.fvecs"
QUERY_ALPHA="/home/gongwei/deg/dataset/CC3M_disk/base_gt_round_00/query_alpha.fvecs"
RESULT_PATH="/home/gongwei/deg/dataset/CC3M_disk/base_gt_round_00/top10_results.ivecs"

THREADS=32
K=10
L=100

DATATYPE="float"
PARAM_2="4"
METRIC="l2"
EXTRA_PARAMS="3 0 1"

# =================================================================
# 帮助函数 (Help Function)
# =================================================================
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -i <path>    Index path (索引路径)"
    echo "  -m <path>    Query Image path (图像查询路径)"
    echo "  -t <path>    Query Text path (文本查询路径)"
    echo "  -a <path>    Query Alpha path (Alpha参数路径)"
    echo "  -r <path>    Result output path (结果输出路径)"
    echo "  -n <num>     Number of threads (线程数, default: 32)"
    echo "  -k <num>     Top K (default: 10)"
    echo "  -l <num>     Search queue length (L参数, default: 100)"
    echo "  -h           Show this help message"
    exit 1
}

# =================================================================
# 参数解析 (Argument Parsing)
# =================================================================
while getopts "i:m:t:a:r:n:k:l:h" opt; do
  case $opt in
    i) INDEX_PATH="$OPTARG" ;;
    m) QUERY_IMG="$OPTARG" ;;
    t) QUERY_TEXT="$OPTARG" ;;
    a) QUERY_ALPHA="$OPTARG" ;;
    r) RESULT_PATH="$OPTARG" ;;
    n) THREADS="$OPTARG" ;;
    k) K="$OPTARG" ;;
    l) L="$OPTARG" ;;
    h) usage ;;
    *) usage ;;
  esac
done

# =================================================================
# 执行命令 (Execution)
# =================================================================

# 检查可执行文件是否存在
EXE_PATH="./build/tests/search_disk_index"
if [ ! -f "$EXE_PATH" ]; then
    echo "Error: Executable $EXE_PATH not found!"
    exit 1
fi

echo "------------------------------------------------"
echo "Starting Search with the following parameters:"
echo "Index:   $INDEX_PATH"
echo "Threads: $THREADS"
echo "K:       $K"
echo "L:       $L"
echo "Output:  $RESULT_PATH"
echo "------------------------------------------------"

# 运行命令
# 注意：变量加双引号是为了防止路径中包含空格导致错误
set -x # 开启调试模式，打印具体执行的命令
$EXE_PATH $DATATYPE \
    "$INDEX_PATH" \
    "$THREADS" "$PARAM_2" \
    "$QUERY_IMG" \
    "$QUERY_TEXT" \
    "$QUERY_ALPHA" \
    "$RESULT_PATH" \
    "$K" $METRIC $EXTRA_PARAMS "$L"
set +x # 关闭调试模式