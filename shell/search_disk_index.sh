project_path=/home/loujh/yq_code/PipeANN-main

data_path=/data/dataset_vec_public

output_path=/gauss_yusong/ljh_data/freshdiskann_output

cd $project_path/build
cmake ..
make -j
cd $project_path

# exit 0
data_name=sift100w # sift, gist, sift100w
pq_dim=64
K=10
query=${data_path}/${data_name}/${data_name}_query.fbin
gt=${data_path}/${data_name}/${data_name}_gt_0.00125/gt_0.bin 
# query=${data_path}/${data_name}/${data_name}_query.bin
# gt=${data_path}/${data_name}/${data_name}_gt_K10.bin
disk_init_path=${output_path}/${data_name}/disk_init_pq$pq_dim
data_type=float
L="10 20 30 40 50 60 70 80 90 100"
# L="50 100 150 200 250 300 350 400 450 500"
# L="50"
L_mem=10
num_threads=8
search_mode=3 #0 base, 1 starling, 2 pipeann 3 rerank
pipeline_width=32
strategy=1
index_path_prefix=${output_path}/${data_name}/${data_name}

rm "$output_path"/${data_name}/${data_name}_disk.index.tags
cp $disk_init_path/${data_name}_disk.index "$output_path"/${data_name}/
cp $disk_init_path/${data_name}_mem.index "$output_path"/${data_name}/
cp $disk_init_path/${data_name}_mem.index.data "$output_path"/${data_name}/
cp $disk_init_path/${data_name}_mem.index.tags "$output_path"/${data_name}/
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

# for strategy in 0
# do
cmd="${project_path}/build/tests/search_disk_index  \
    ${data_type} \
    ${index_path_prefix} \
    ${num_threads} \
    ${pipeline_width} \
    ${query} \
    ${gt} \
    ${K} \
    l2 \
    ${search_mode} \
    ${L_mem} \
    ${strategy} \
    ${L}
    "
echo ${cmd}
eval ${cmd}
# done
