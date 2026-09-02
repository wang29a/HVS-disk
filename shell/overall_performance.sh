project_path=/home/loujh/yq_code/PipeANN-main

data_path=/data/dataset_vec_public

output_path=/gauss_yusong/ljh_data/freshdiskann_output

cd $project_path/build

i=${1:-0}

if [ "$i" -eq 1 ]; then
    flag="-DREORDER_COMPUTE_PQ -DUSE_TOPO_DISK  -DUSE_NHOOD_CACHE"
fi

export ADDITIONAL_DEFINITIONS="$flag"
cmake ..
make -j8

cd "$project_path"

data_name=msong # msong, gist, sift100w
batch_num=0

if [ "$data_name" = "sift1b" ]; then
    L="100"
    pq_dim=32
    data="${data_path}/${data_name}/${data_name}_base.bin"
    query="${data_path}/${data_name}/${data_name}_query.bin"
    data_type=uint8
    batch_size=1000000
    truthset_prefix=${data_path}/${data_name}/${data_name}_gt_K10.bin
else
    L="100"
    pq_dim=64
    data="${data_path}/${data_name}/${data_name}_base.fbin"
    query="${data_path}/${data_name}/${data_name}_query.fbin"
    data_type=float
    batch_size=1000
    truthset_prefix=${data_path}/${data_name}/${data_name}_gt_0.00125
fi

mkdir -p "${output_path}/${data_name}/disk_init_pq${pq_dim}"
K=10

if [ "$i" -eq 0 ]; then
    search_mode=0 # 0 base, 1 starling, 2 pipeann, 3 rerank
    pipeline_width=4
    strategy=0
else
    search_mode=3 #0 base, 1 starling, 2 pipeann 3 rerank
    pipeline_width=32
    strategy=63
fi

disk_init_path=${output_path}/${data_name}/disk_init_pq$pq_dim
index_path_prefix=${output_path}/${data_name}/${data_name}
result=${output_path}/${data_name}/result
l_disk=75
# truthset_prefix=${data_path}/${data_name}/${data_name}_query_learn_gt

rm "$output_path"/${data_name}/${data_name}_disk.index.tags
cp $disk_init_path/${data_name}_disk.index "$output_path"/${data_name}/
cp $disk_init_path/${data_name}_pq_compressed.bin "$output_path"/${data_name}/
cp $disk_init_path/${data_name}_pq_pivots.bin "$output_path"/${data_name}/
cp $disk_init_path/${data_name}_pq_compressed_2.bin "$output_path"/${data_name}/
cp $disk_init_path/${data_name}_pq_pivots_2.bin "$output_path"/${data_name}/
cp $disk_init_path/disk_index_graph "$output_path"/${data_name}/
cp $disk_init_path/disk_index_data "$output_path"/${data_name}/
cp $disk_init_path/dram_index_graph "$output_path"/${data_name}/
cp $disk_init_path/reordered_disk_index_data "$output_path"/${data_name}/
cp $disk_init_path/reordered_disk_index_graph "$output_path"/${data_name}/
cp $disk_init_path/reorder_map_data "$output_path"/${data_name}/
cp $disk_init_path/reorder_map_graph "$output_path"/${data_name}/

cmd="${project_path}/build/tests/overall_performance  \
    ${data_type} \
    ${data} \
    ${l_disk}
    ${index_path_prefix} \
    ${query} \
    ${truthset_prefix} \
    ${K} \
    ${pipeline_width} \
    ${batch_num} \
    ${batch_size} \
    ${search_mode} \
    ${strategy} \
    ${L}
    "
echo ${cmd}
eval ${cmd}