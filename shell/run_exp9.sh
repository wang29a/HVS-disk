project_path=/home/loujh/yq_code/PipeANN-main

data_path=/data/dataset_vec_public

output_path=/gauss_yusong/ljh_data/freshdiskann_output

log_path=/home/loujh/yq_code/PipeANN-main/log/fig9

linux_aligned_file_reader=${project_path}/src/utils/linux_aligned_file_reader.cpp
mkdir $log_path

K=10
num_threads=8

# for strategy in 3
# do
    for sys in odinann
    do

        if [ "$sys" = "dgai" ]; then
            # flag="-DREORDER_COMPUTE_PQ -DUSE_TOPO_DISK -DUSE_NHOOD_CACHE"
            flag="-DREORDER_COMPUTE_PQ -DUSE_TOPO_DISK -DUSE_NHOOD_CACHE"
            L_mem=0
            search_mode=3
            pipeline_width=32
            strategy=63
        else
            flag="-DREAD_ONLY_TESTS -DNO_MAPPING"
            L_mem=10
            search_mode=0
            pipeline_width=4
            strategy=0
        fi

        # cd $project_path/build
        # export ADDITIONAL_DEFINITIONS="$flag"
        # cmake ..
        # make -j8
        # cd "$project_path"

        # for data_name in sift100w gist msong deep100m
        for data_name in deep1m deep10m 
        do

            # data_name=sift1b # sift, gist, sift100w
            pq_dim=64

            # query=${data_path}/${data_name}/${data_name}_query.bin
            # gt=${data_path}/${data_name}/${data_name}_gt_K10.bin
            # data_type=uint8
            query=${data_path}/${data_name}/${data_name}_query.fbin
            gt=${data_path}/${data_name}/${data_name}_gt_0.00125/gt_0.bin 
            # gt=${data_path}/${data_name}/${data_name}_gt_0.01/gt_0.bin 
            data_type=float

            if [ "$data_name" = "sift100w" ]; then
                # L="10 20 30 40 50 60 70 80 90 100"
                L="10 20 30 40 50 60 70 80 90 100 110 120 130 140 150"
                trunc_len=18
            elif [ "$data_name" = "msong" ]; then
                L="10 30 70 100 130 160 190 210 240 270 300 350 400"
                trunc_len=168
            elif [ "$data_name" = "gist" ]; then
                L="50 100 150 200 250 300 350 400 450 500 600 700 800 900 1000"
                # L="50 100 150 200 250 300 350 400 450 500 600"
                trunc_len=511
            elif [ "$data_name" = "deep100m" ]; then
                L="40 70 100 130 160 200 250 300 350 400 450 500 550 600 650 700"
                trunc_len=21
            elif [ "$data_name" = "deep10m" ]; then
                L="40 70 100 130 160 170 180 190 200 250 300 350 400"
                trunc_len=21
            elif [ "$data_name" = "deep1m" ] || [ "$data_name" = "deep50w" ]; then
                L="10 20 30 40 50 60 70 75 80 90 100 110 120 130 140 150"
                trunc_len=21
            fi
            
            # if [ "$data_name" = "sift100w" ]; then
            #     L="10 15 20 25 30 40 50 60 70 80 90 100"
            #     trunc_len=18
            # elif [ "$data_name" = "msong" ]; then
            #     L="10 30 50 75 100 130 160 190 210 240 300 400"
            #     trunc_len=168
            # elif [ "$data_name" = "gist" ]; then
            #     L="50 100 150 200 250 300 350 400 450 500 550 600 700 800 900 1000"
            #     # L="50 100 150 200 250 300 350 400 450 500 600"
            #     trunc_len=511
            # elif [ "$data_name" = "deep100m" ]; then
            #     L="40 70 100 130 160 200 250 300 350 400 450 500 550 600 650 700"
            #     trunc_len=21
            # elif [ "$data_name" = "deep1m" ] || [ "$data_name" = "deep50w" ]; then
            #     L="40 70 100 130 160 200 250 300 350 400"
            # fi


            sed -i 's/\(trunc_len *= *\)[0-9]\+/\1'"$trunc_len"'/' $linux_aligned_file_reader 
            # exit 0
            cd $project_path/build
            export ADDITIONAL_DEFINITIONS="$flag"
            cmake ..
            make -j8
            cd "$project_path"

            disk_init_path=${output_path}/${data_name}/disk_init_pq$pq_dim

            index_path_prefix=${output_path}/${data_name}/${data_name}

            rm "$output_path"/${data_name}/${data_name}_disk.index.tags
            cp $disk_init_path/${data_name}_disk.index "$output_path"/${data_name}/
            if [ "$sys" = "odinann" ]; then
                cp $disk_init_path/${data_name}_mem.index "$output_path"/${data_name}/
                cp $disk_init_path/${data_name}_mem.index.data "$output_path"/${data_name}/
                cp $disk_init_path/${data_name}_mem.index.tags "$output_path"/${data_name}/
            fi
            cp $disk_init_path/${data_name}_pq_compressed.bin "$output_path"/${data_name}/
            cp $disk_init_path/${data_name}_pq_pivots.bin "$output_path"/${data_name}/

            # cp $disk_init_path/disk_index_graph "$output_path"/${data_name}/
            # cp $disk_init_path/disk_index_data "$output_path"/${data_name}/
            # cp $disk_init_path/dram_index_graph "$output_path"/${data_name}/
            if [ "$sys" = "dgai" ]; then
                cp $disk_init_path/${data_name}_pq_compressed_2.bin "$output_path"/${data_name}/
                cp $disk_init_path/${data_name}_pq_pivots_2.bin "$output_path"/${data_name}/
                # cp $disk_init_path/reordered_disk_index_data "$output_path"/${data_name}/
                cp $disk_init_path/reordered_disk_index_data_2 "$output_path"/${data_name}/
                if [ "$data_name" = "gist" ]; then
                    cp $disk_init_path/disk_index_data "$output_path"/${data_name}/
                fi

                cp $disk_init_path/reordered_disk_index_graph_2 "$output_path"/${data_name}/
                # cp $disk_init_path/reordered_disk_index_graph "$output_path"/${data_name}/
                # cp $disk_init_path/reorder_map_data "$output_path"/${data_name}/
                cp $disk_init_path/reorder_map_data_2 "$output_path"/${data_name}/
                # cp $disk_init_path/reorder_map_graph "$output_path"/${data_name}/
                cp $disk_init_path/reorder_map_graph_2 "$output_path"/${data_name}/
            fi


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
            elif [ "$data_name" = "deep1m" ]; then
                dataset_name='DEEP1M'
            fi

            # log_file_path=${log_path}/_.log
            log_file_path=${log_path}/${sys_name}_${dataset_name}.log

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
# done
