project_path=/home/loujh/yq_code/PipeANN-main

data_path=/data/dataset_vec_public

output_path=/gauss_yusong/ljh_data/freshdiskann_output

log_path=/home/loujh/yq_code/PipeANN-main/log/fig8


i=${1:-0}
batch_num=5
batch_size=100000
K=10

# for i in 4 5
# do
    # log_path=/home/loujh/yq_code/PipeANN-main/log/fig1_${i}
    mkdir $log_path
# for sys in odinann dgai
    for sys in dgai odinann
    do

        if [ "$sys" = "dgai" ]; then
            # flag="-DREORDER_COMPUTE_PQ -DUSE_TOPO_DISK -DUSE_NHOOD_CACHE"
            flag="-DREORDER_COMPUTE_PQ -DUSE_TOPO_DISK "
            search_mode=3 
            pipeline_width=32
            strategy=31
            # strategy=27
        else
            flag="-DBG_IO_THREAD"
            # flag=""
            search_mode=0 
            pipeline_width=4
            strategy=0
        fi

        cd $project_path/build
        export ADDITIONAL_DEFINITIONS="$flag"
        cmake ..
        make -j
        cd "$project_path"

    # exit 0
        # for data_name in sift100w msong gist
        for data_name in deep1m deep10m deep100m
        do
            L='100'
            pq_dim=64
            if [ "$data_name" = "sift100w" ]; then
                L="50"
            elif [ "$data_name" = "msong" ]; then
                L="100"
            elif [ "$data_name" = "gist" ]; then
                L="250"
            elif [ "$data_name" = "deep1m" ]; then
                L="100"
                batch_size=1000
            elif [ "$data_name" = "deep10m" ]; then
                L="200"
                batch_size=10000
            elif [ "$data_name" = "deep100m" ]; then
                L="300"
                batch_size=100000
            fi
            data="${data_path}/${data_name}/${data_name}_base.fbin"
            query="${data_path}/${data_name}/${data_name}_query.fbin"
            data_type=float
            truthset_prefix=${data_path}/${data_name}/${data_name}_gt_0.00125

            disk_init_path=${output_path}/${data_name}/disk_init_pq$pq_dim
            index_path_prefix=${output_path}/${data_name}/${data_name}
            l_disk=75

            rm "$output_path"/${data_name}/${data_name}_disk.index.tags
            # if [ "$sys" = "odinann" ]; then
            cp $disk_init_path/${data_name}_disk.index "$output_path"/${data_name}/
            # fi
            cp $disk_init_path/${data_name}_pq_compressed.bin "$output_path"/${data_name}/
            cp $disk_init_path/${data_name}_pq_pivots.bin "$output_path"/${data_name}/

            # cp $disk_init_path/disk_index_graph "$output_path"/${data_name}/
            # cp $disk_init_path/dram_index_graph "$output_path"/${data_name}/
            if [ "$sys" = "dgai" ]; then
                # truncate -s 4096 $output_path/${data_name}_disk.index
                cp $disk_init_path/${data_name}_pq_compressed_2.bin "$output_path"/${data_name}/
                cp $disk_init_path/${data_name}_pq_pivots_2.bin "$output_path"/${data_name}/
                # cp $disk_init_path/reordered_disk_index_data "$output_path"/${data_name}/
                cp $disk_init_path/disk_index_data "$output_path"/${data_name}/
                cp $disk_init_path/reordered_disk_index_data_2 "$output_path"/${data_name}/
                # cp $disk_init_path/reordered_disk_index_graph "$output_path"/${data_name}/
                cp $disk_init_path/reordered_disk_index_graph_2 "$output_path"/${data_name}/
                # cp $disk_init_path/reorder_map_data "$output_path"/${data_name}/
                cp $disk_init_path/reorder_map_data_2 "$output_path"/${data_name}/
                # cp $disk_init_path/reorder_map_graph "$output_path"/${data_name}/
                cp $disk_init_path/reorder_map_graph_2 "$output_path"/${data_name}/
            fi
            # exit 0
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
            elif [ "$data_name" = "deep100m" ]; then
                dataset_name='DEEP100M'
            elif [ "$data_name" = "deep10m" ]; then
                dataset_name='DEEP10M'
            elif [ "$data_name" = "deep1m" ]; then
                dataset_name='DEEP1M'
            fi

            # log_file_path=${log_path}/_.log
            log_file_path=${log_path}/${sys_name}_${dataset_name}.log

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
# done