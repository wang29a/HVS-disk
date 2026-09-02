project_path=/home/loujh/yq_code/PipeANN-main

data_path=/data/dataset_vec_public

output_path=/gauss_yusong/ljh_data/freshdiskann_output

log_path=/home/loujh/yq_code/PipeANN-main/log/fig4

linux_aligned_file_reader=${project_path}/src/utils/linux_aligned_file_reader.cpp

mkdir $log_path

mkdir -p $log_path
K=10

for sys in dgai
do

    if [ "$sys" = "dgai" ]; then
        flag="-DREORDER_COMPUTE_PQ -DUSE_TOPO_DISK -DUSE_NHOOD_CACHE"
        L_mem=0
        search_mode=3 #0 base, 1 starling, 2 pipeann 3 rerank
        pipeline_width=32
        strategy=47
    else
        flag="-DREAD_ONLY_TESTS -DNO_MAPPING"
        L_mem=10
        search_mode=0 
        pipeline_width=4
        strategy=0
    fi

    cd $project_path/build
    export ADDITIONAL_DEFINITIONS="$flag"
    cmake ..
    make -j8
    cd "$project_path"

    for num_threads in 1 2 4 8 16
    do
        for data_name in sift100w
        do

            pq_dim=64

            query=${data_path}/${data_name}/${data_name}_query.fbin
            gt=${data_path}/${data_name}/${data_name}_gt_0.00125/gt_0.bin 
            data_type=float

            disk_init_path=${output_path}/${data_name}/disk_init_pq$pq_dim


            if [ "$data_name" = "sift100w" ]; then
                L=" 90 99" # recall99
                trunc_len=19
                if [ "$sys" = "dgai" ]; then
                    L=" 90 97"
                fi
                L="60 62 " #recall98
            elif [ "$data_name" = "msong" ]; then
                L=" 240 270"
                if [ "$sys" = "dgai" ]; then
                    L="240 300"
                fi
            elif [ "$data_name" = "gist" ]; then
                L=" 200 213" #90
                if [ "$sys" = "dgai" ]; then
                    L="200 249" #90
                fi
            fi

            index_path_prefix=${output_path}/${data_name}/${data_name}


            sed -i 's/\(trunc_len *= *\)[0-9]\+/\1'"$trunc_len"'/' $linux_aligned_file_reader 
            
            cd $project_path/build
            export ADDITIONAL_DEFINITIONS="$flag"
            cmake ..
            make -j8
            cd "$project_path"

            # rm "$output_path"/${data_name}/${data_name}_disk.index.tags
            # if [ "$sys" = "odinann" ]; then
            #     cp $disk_init_path/${data_name}_disk.index "$output_path"/${data_name}/
            #     cp $disk_init_path/${data_name}_mem.index "$output_path"/${data_name}/
            #     cp $disk_init_path/${data_name}_mem.index.data "$output_path"/${data_name}/
            #     cp $disk_init_path/${data_name}_mem.index.tags "$output_path"/${data_name}/
            # fi
            # cp $disk_init_path/${data_name}_pq_compressed.bin "$output_path"/${data_name}/
            # cp $disk_init_path/${data_name}_pq_pivots.bin "$output_path"/${data_name}/

            # cp $disk_init_path/disk_index_graph "$output_path"/${data_name}/
            # cp $disk_init_path/disk_index_data "$output_path"/${data_name}/
            # cp $disk_init_path/dram_index_graph "$output_path"/${data_name}/
            # if [ "$sys" = "dgai" ]; then
            #     cp $disk_init_path/${data_name}_pq_compressed_2.bin "$output_path"/${data_name}/
            #     cp $disk_init_path/${data_name}_pq_pivots_2.bin "$output_path"/${data_name}/
            #     cp $disk_init_path/reordered_disk_index_data "$output_path"/${data_name}/
            #     cp $disk_init_path/reordered_disk_index_data_2 "$output_path"/${data_name}/
            #     cp $disk_init_path/reordered_disk_index_graph_2 "$output_path"/${data_name}/
            #     cp $disk_init_path/reordered_disk_index_graph "$output_path"/${data_name}/
            #     cp $disk_init_path/reorder_map_data "$output_path"/${data_name}/
            #     cp $disk_init_path/reorder_map_data_2 "$output_path"/${data_name}/
            #     cp $disk_init_path/reorder_map_graph "$output_path"/${data_name}/
            #     cp $disk_init_path/reorder_map_graph_2 "$output_path"/${data_name}/
            # fi

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
            fi

            log_file_path=${log_path}/${sys_name}_${dataset_name}_${num_threads}.log

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
                ${L} 2>&1 | tee ${log_file_path}
                "
            echo ${cmd}
            eval ${cmd}

        done
    done
done
