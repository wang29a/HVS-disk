project_path=/home/loujh/yq_code/PipeANN-main

data_path=/data/dataset_vec_public

output_path=/gauss_yusong/ljh_data/freshdiskann_output

log_path=/home/loujh/yq_code/PipeANN-main/log/fig3

mkdir $log_path

i=${1:-0}

batch_num=-1

for sys in dgai odinann
do

    if [ "$sys" = "dgai" ]; then
        flag="-DREORDER_COMPUTE_PQ -DUSE_TOPO_DISK  -DUSE_NHOOD_CACHE"
        search_mode=3 
        pipeline_width=32
        strategy=63
    else
        flag=""
        search_mode=0 
        pipeline_width=4
        strategy=0
    fi

    cd $project_path/build
    export ADDITIONAL_DEFINITIONS="$flag"
    cmake ..
    make -j8
    cd "$project_path"

    # for update_ratio in 0.001
    for update_ratio in 0.001 0.003 0.007 0.01 0.03 0.07 0.1
    do

        for data_name in sift100w
        do

            if [ "$data_name" = "sift100w" ]; then
                L="50"
            elif [ "$data_name" = "msong" ]; then
                L="100"
            elif [ "$data_name" = "gist" ]; then
                L="250"
            elif [ "$data_name" = "word2vec" ]; then
                L="100"
            fi
            
            pq_dim=64
            data="${data_path}/${data_name}/${data_name}_base.fbin"
            query="${data_path}/${data_name}/${data_name}_query.fbin"
            data_type=float
            batch_size=$update_ratio
            truthset_prefix=${data_path}/${data_name}/${data_name}_gt_${update_ratio}

            K=10

            disk_init_path=${output_path}/${data_name}/disk_init_pq$pq_dim
            index_path_prefix=${output_path}/${data_name}/${data_name}
            l_disk=75

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

            if [ "$sys" = "odinann" ]; then
                sys_name='OdinANN'
            elif [ "$sys" = "dgai" ]; then
                sys_name='DGAI'
            fi

            if [ "$data_name" = "sift1b" ]; then
                dataset_name='SIFT1B'
            elif [ "$data_name" = "sift100w" ]; then
                dataset_name='SIFT1M'
            elif [ "$data_name" = "gist" ]; then
                dataset_name='GIST'
            elif [ "$data_name" = "msong" ]; then
                dataset_name='MSONG'
            elif [ "$data_name" = "word2vec" ]; then
                dataset_name='WORD2VEC'
            fi

            log_file_path=${log_path}/${sys_name}_${dataset_name}_${update_ratio}.log

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
            eval ${cmd} 2>&1 | tee ${log_file_path}

        done
    done
done